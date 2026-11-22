"""The real fall classifier in the harness, with a fake scorer in place of the LiteRT model."""

import numpy as np
import pytest
from e3_edge.anomaly import FrameStream
from e3_edge.anomaly.simulate import FALL, HouseholdSimulator

from src.anomaly.evaluate import Tally, evaluate, report
from src.anomaly.fall_model import FrameScores, RealClassifier, notifications, select_folds, split_candidates
from src.anomaly.inject import Planted, Planter, clips
from src.anomaly.methods import EdgeBaseline

DAY = 86400.0


@pytest.fixture(scope="module")
def data():
    sim = HouseholdSimulator(dim=24, hop=0.96, seed=4)
    t = np.concatenate([10 * DAY + i * 60 + np.array([0.0, 0.96]) for i in range(20)])
    falls = sim.frames(t, np.full(len(t), FALL, np.int8))
    return sim.stream(0, 2 * DAY), sim.stream(2 * DAY, DAY), falls


def _fake_scores(falls: FrameStream, false_rows: dict[bytes, float] | None = None):
    """0.9 for SAFE fall frames, a chosen score for some household frames, 0 otherwise."""
    fall_rows = {row.tobytes() for row in falls.embeddings}
    calls = []

    def score(embeddings):
        calls.append(len(embeddings))
        return np.array([0.9 if row.tobytes() in fall_rows else (false_rows or {}).get(row.tobytes(), 0.0)
                         for row in embeddings])

    return FrameScores(score), calls


def test_one_notification_per_burst_after_its_first_window():
    t = np.arange(20) * 0.48
    s = np.zeros(20)
    s[[2, 3, 5]] = [0.7, 0.9, 0.8]  # one burst: 2.4 s from first to last frame above threshold
    s[18] = 0.95  # 6.2 s after the last: a new burst
    out = notifications(t, s, 0.66, frame_seconds=0.96, rearm_seconds=5.0)
    assert out == [(pytest.approx(2 * 0.48 + 0.96), 0.7), (pytest.approx(18 * 0.48 + 0.96), 0.95)]
    assert notifications(t, s, 0.85, 0.96, 5.0)[0][1] == 0.9  # a higher threshold starts later


def test_planted_copy_rescored_only_where_planting_changed_it(data):
    _, test, falls = data
    scores, calls = _fake_scores(falls)
    planter = Planter(test, falls, seed=0)
    planter.fall(2 * DAY + 3 * 3600, quiet_seconds=120)
    clf = RealClassifier(scores, 0.66)
    assert clf.candidates(test) == []
    assert calls == [len(test)]
    planted = clf.candidates(planter.target, test, planter.touched())
    assert calls[1] == len(planter.touched()) < len(test)
    assert len(planted) == 1 and 0 < planted[0][0] - planter.planted[0].start <= 2.0
    clf.candidates(test)
    assert len(calls) == 2  # the test days' scores are cached


def test_false_triggers_are_the_notifications_not_for_a_planted_fall():
    candidates = [(100.5, 0.9), (104.0, 0.8), (300.0, 0.7), (50.0, 0.7)]
    assert split_candidates(candidates, [100.0, 290.0]) == [50.0]
    assert split_candidates(candidates, []) == [100.5, 104.0, 300.0, 50.0]


def test_tally_times_the_classifier_notification_of_each_fall():
    planted = [Planted("fall", 1000.0, 600.0), Planted("fall", 5000.0, 600.0),
               Planted("impact_burst", 9000.0, 160.0)]
    tally = Tally()
    tally.add(planted, [], [9001.0], candidates=[(1001.5, 0.9), (9001.0, 0.8)])
    out = tally.summary(days=1.0)["fall:classifier_notified"]
    assert (out["planted"], out["detected"], out["latency_s_max"]) == (2, 1, 1.5)


def test_folds_come_from_the_sorted_wav_names(data, tmp_path):
    _, _, falls = data
    names = [f"{fold:02d}-{k:03d}-00-001-01.wav" for k in range(20) for fold in [(k % 10) + 1]]
    for name in names + ["01-999-00-001-02.wav"]:  # a non-fall, ignored for the -01 suffix
        (tmp_path / name).write_bytes(b"")
    kept = select_folds(falls, tmp_path, "-01", (9, 10))
    parts = clips(falls)
    want = [i for i, name in enumerate(sorted(names)) if int(name[:2]) in (9, 10)]
    assert len(clips(kept)) == len(want) == 4
    np.testing.assert_array_equal(kept.embeddings, falls.embeddings[np.concatenate([parts[i] for i in want])])
    (tmp_path / "10-999-00-001-01.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="exported from"):
        select_folds(falls, tmp_path, "-01", (9, 10))


def test_evaluation_with_the_real_classifier(data):
    calib, test, falls = data
    # Two household frames the classifier mistakes for falls, 6 h apart.
    false_at = [int(np.searchsorted(test.timestamps, 2 * DAY + h * 3600)) for h in (4, 10)]
    scores, _ = _fake_scores(falls, {test.embeddings[i].tobytes(): 0.7 for i in false_at})
    clf = RealClassifier(scores, 0.66)
    results = evaluate(calib, test, falls, methods=[EdgeBaseline("category")], per_kind=2,
                       parts=("behavioural", "classifier"), fall_classifier=clf, fall_thresholds=(0.8,),
                       info={"fall_classifier": {"threshold": 0.66}})
    real = results["behavioural"]["category"]["real_classifier"]
    assert real["fall:classifier_notified"]["detection_rate"] == 1.0
    assert real["fall:classifier_notified"]["latency_s_max"] <= 2.0
    assert real["fall:post_impact_silence"]["detection_rate"] == 1.0
    untouched = real["untouched_test_days"]
    assert untouched["false_triggers"] == 2 and len(untouched["triggers"]) == 2
    assert {"local_time", "score", "top_class", "became_alert"} <= set(untouched["triggers"][0])
    sweep = results["classifier"]
    assert set(sweep) >= {"ideal", "real_0.66", "real_0.8"}
    assert sweep["real_0.8"]["untouched_test_days"]["false_triggers"] == 0  # 0.7 is below 0.8
    text = report(results, synthetic=True)
    assert "Real fall classifier on the untouched test days" in text and "real_0.8" in text
