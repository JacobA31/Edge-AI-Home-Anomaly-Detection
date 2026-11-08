"""The evaluation harness, run end to end on small synthetic data."""

import json

import numpy as np
import pytest
from e3_edge.anomaly import AnomalyConfig, FrameStream
from e3_edge.anomaly.categories import DEFAULT_CATEGORIES, CategoryMapper
from e3_edge.anomaly.simulate import FALL, HouseholdSimulator
from e3_edge.anomaly.temporal import Finding

from src.anomaly import metrics
from src.anomaly.evaluate import (
    ClassifierStandIn,
    Tally,
    concat,
    evaluate,
    main,
    report,
    sound_rows,
    with_overrides,
)
from src.anomaly.inject import Planted, Planter, clips
from src.anomaly.methods import (
    EdgeBaseline,
    Euclidean,
    KMeansSelective,
    KNNScorer,
    default_methods,
    parse_context,
)

DAY = 86400.0


def test_default_edge_methods_share_the_requested_local_timezone():
    cfg = AnomalyConfig(timezone="Europe/Brussels")
    methods = default_methods(True, cfg)
    edge_methods = [method for method in methods if isinstance(method, EdgeBaseline)]
    assert len(edge_methods) == 4
    assert all(method.cfg.timezone == "Europe/Brussels" for method in edge_methods)


@pytest.fixture(scope="module")
def data():
    sim = HouseholdSimulator(dim=24, hop=0.96, seed=4)
    t = np.concatenate([10 * DAY + i * 60 + np.array([0.0, 0.96]) for i in range(20)])
    falls = sim.frames(t, np.full(len(t), FALL, np.int8))
    return sim.stream(0, 2 * DAY), sim.stream(2 * DAY, DAY), falls


def test_metrics_on_known_scores():
    normal, anomalous = np.arange(100.0), np.arange(100.0) + 1000
    assert metrics.auc(normal, anomalous) == 1.0
    assert metrics.partial_auc(normal, anomalous, 0.1) == pytest.approx(1.0)
    assert metrics.partial_auc(normal, normal.copy(), 0.1) == pytest.approx(0.05, abs=0.01)  # chance
    assert metrics.tpr_at_fpr(normal, np.array([50.0, 99.5]), 0.01) == 0.5
    out = metrics.at_threshold(normal, np.array([10.0, 500.0]), 89.5, hours=2.0)
    assert out == {"false_alarm_frame_share": 0.1, "false_alarm_frames_per_hour": 5.0,
                   "anomalies_caught": 0.5, "anomalies_missed": 0.5}


def test_planting_uses_the_homes_own_frames(data):
    calib, test, falls = data
    assert len(clips(falls)) == 20
    original = test.embeddings.copy()
    planter = Planter(test, falls, seed=1)
    assert all(planter.available().values())
    planter.prolonged_water(2 * DAY + 3 * 3600, 1200)
    planter.impact_burst(2 * DAY + 5 * 3600, count=6, spacing=10)
    planter.fall(2 * DAY + 6 * 3600, quiet_seconds=300)
    assert [p.kind for p in planter.planted] == ["prolonged_water", "impact_burst", "fall"]
    np.testing.assert_array_equal(test.embeddings, original)  # planting works on a copy
    assert not np.array_equal(planter.target.embeddings, original)
    rows = (test.timestamps >= 2 * DAY + 3 * 3600) & (test.timestamps < 2 * DAY + 3 * 3600 + 1200)
    donors = {tuple(r) for r in test.embeddings[planter.water]}
    assert all(tuple(r) in donors for r in planter.target.embeddings[rows])
    # the fall's quiet comes from the original stream, never from frames already planted (water)
    t = planter.target.timestamps
    after_fall = (t > 2 * DAY + 6 * 3600 + 3) & (t < 2 * DAY + 6 * 3600 + 300)
    categories = CategoryMapper(DEFAULT_CATEGORIES).scores_batch(planter.target.full_scores()[after_fall])
    assert categories.max() < 0.1


def test_methods_rank_synthetic_falls_above_household_frames(data):
    calib, test, falls = data
    for method in (EdgeBaseline("category"), EdgeBaseline("global", n_components=8), KMeansSelective(k=4),
                   KNNScorer(pca_dims=8, bank=5000), Euclidean()):
        method.fit(calib, 0.999)
        assert metrics.auc(method.score(test.slice(0, 20000)), method.score(falls)) > 0.9, method.name


def test_evaluation_end_to_end(data, tmp_path):
    calib, test, falls = data
    results = evaluate(calib, test, falls, non_falls=falls, methods=[EdgeBaseline("category"), Euclidean()],
                       per_kind=2, curve_hours=(12, 48), contexts=("category", "global"),
                       parts=("frame_level", "behavioural", "calibration_curve"))
    assert [r["method"] for r in results["frame_level"]] == ["edge_category", "euclidean"]
    edge = results["frame_level"][0]
    assert edge["edge_ready"] is True
    assert edge["auc_falls_vs_non_falls"] == 0.5  # the control: identical clips can't be told apart
    assert 0 <= edge["flagged_impact"] <= 1
    assert set(results["behavioural"]) == {"category", "global"}
    with_classifier = results["behavioural"]["category"]["ideal_classifier"]
    assert with_classifier["fall:post_impact_silence"]["detection_rate"] == 1.0
    assert with_classifier["fall:post_impact_inactivity"]["detection_rate"] == 1.0
    assert with_classifier["fall:post_impact_silence"]["latency_s_max"] <= 10  # NF1
    assert with_classifier["prolonged_water:prolonged_water"]["detection_rate"] == 1.0
    assert with_classifier["impact_burst:repeated_impacts"]["detection_rate"] == 1.0
    assert [r["calibration_hours"] for r in results["calibration_curve"]] == [12, 48]
    assert with_classifier["stray_alerts_per_day"] >= 0 and "side_alerts" in with_classifier
    weaker = results["behavioural"]["category"]["weaker_classifier"]
    assert weaker["fall:post_impact_silence"]["planted"] == 2
    text = report(results, synthetic=True)
    assert "Synthetic data" in text and "edge_category" in text and "By embedding context" in text


def test_classifier_stand_in_misses_delays_and_fires_on_household_sounds(data):
    _, test, falls = data
    planter = Planter(test, falls, seed=2)
    for k in range(4):
        planter.fall(2 * DAY + (2 + 4 * k) * 3600, quiet_seconds=300)
    sounds = sound_rows(test)
    assert len(sounds)
    standin = ClassifierStandIn(recall=0.5, delay_s=3.0, false_per_day=20.0)
    candidates, false = standin.candidates(planter.planted, test, sounds, 1.0, np.random.default_rng(0))
    hits = [t for t, _ in candidates if t not in false]
    assert len(hits) == 2
    assert all(any(t == p.start + 3.0 for p in planter.planted) for t in hits)
    assert 0 < len(false) and set(false) <= set(test.timestamps[sounds])
    for p in planter.planted:  # never inside a planted fall or the 10 min around it
        assert all(not p.start - 600 <= t <= p.start + p.length + 600 for t in false)


def _finding(kind, started, detected, role="alert", **extra):
    return Finding(kind, "anomaly", role, detected, started, 0.9, 1.0, 1.0, "s", extra)


def test_tally_separates_detections_side_alerts_and_stray_alerts():
    planted = [Planted("impact_burst", 1000.0, 160.0), Planted("fall", 5000.0, 600.0)]
    findings = [
        _finding("repeated_impacts", 1000.0, 1080.0),  # detects the burst
        _finding("post_impact_inactivity", 1150.0, 2900.0),  # caused by the burst: side alert
        _finding("post_impact_silence", 5003.0, 5010.0, role="evidence"),
        _finding("post_impact_inactivity", 5003.0, 5125.0),
        _finding("post_impact_inactivity", 9000.0, 9120.0, trigger="fall_candidate"),  # false trigger
        _finding("repeated_impacts", 20000.0, 20050.0),  # stray
    ]
    tally = Tally()
    tally.add(planted, findings, false_triggers=[9000.5])
    out = tally.summary(days=1.0)
    assert out["impact_burst:repeated_impacts"]["detected"] == 1
    assert out["fall:post_impact_silence"]["latency_s_median"] == 10.0
    assert out["fall:post_impact_inactivity"]["detected"] == 1
    assert out["side_alerts"] == {"post_impact_inactivity": 1}
    assert out["stray_alerts"]["by_kind"] == {"post_impact_inactivity": 1, "repeated_impacts": 1}
    assert (out["false_triggers"], out["false_trigger_alerts"]) == (1, 1)


def test_overrides_and_context_names():
    cfg = with_overrides(AnomalyConfig(), {"post_impact.short_seconds": 4.0, "categories.impact.on": 0.1,
                                           "timezone": "UTC"})
    assert cfg.post_impact.short_seconds == 4.0 and cfg.categories["impact"].on == 0.1
    assert parse_context("global_pca64") == ("global", 64) and parse_context("category") == ("category", None)
    with pytest.raises(ValueError):
        parse_context("global_pcax")
    with pytest.raises(ValueError):
        parse_context("nope")


def test_cli_on_files_with_decimation(data, tmp_path):
    calib, test, falls = data
    calib.slice(0, len(calib) // 2).save(tmp_path / "c1.npz")
    calib.slice(len(calib) // 2, len(calib)).save(tmp_path / "c2.npz")
    test.save(tmp_path / "t.npz")
    falls.save(tmp_path / "f.npz")
    parts = [FrameStream.load(tmp_path / "c2.npz"), FrameStream.load(tmp_path / "c1.npz")]
    assert concat(parts).timestamps[0] == 0  # sorted by time, not by argument order
    assert main(["--calibration", str(tmp_path / "c1.npz"), str(tmp_path / "c2.npz"), "--test",
                 str(tmp_path / "t.npz"), "--falls", str(tmp_path / "f.npz"), "--decimate", "2",
                 "--parts", "frame_level", "--timezone", "Europe/Brussels",
                 "--out", str(tmp_path / "out")]) == 0
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert results["data"]["calibration_hours"] == pytest.approx(48, abs=0.1)
    assert results["data"]["timezone"] == "Europe/Brussels"
    assert (tmp_path / "out" / "report.md").read_text().startswith("# Anomaly detection evaluation")
