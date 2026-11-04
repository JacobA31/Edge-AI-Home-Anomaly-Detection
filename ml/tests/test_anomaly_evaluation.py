"""The evaluation harness, run end to end on small synthetic data."""

import json

import numpy as np
import pytest
from e3_edge.anomaly import FrameStream
from e3_edge.anomaly.categories import DEFAULT_CATEGORIES, CategoryMapper
from e3_edge.anomaly.simulate import FALL, HouseholdSimulator

from src.anomaly import metrics
from src.anomaly.evaluate import concat, evaluate, main, report
from src.anomaly.inject import Planter, clips
from src.anomaly.methods import EdgeBaseline, Euclidean, KMeansSelective, KNNScorer

DAY = 86400.0


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
    results = evaluate(calib, test, falls, methods=[EdgeBaseline("category"), Euclidean()], per_kind=2,
                       curve_hours=(12, 48), parts=("frame_level", "behavioural", "calibration_curve"))
    assert [r["method"] for r in results["frame_level"]] == ["edge_category", "euclidean"]
    assert results["frame_level"][0]["edge_ready"] is True
    with_classifier = results["behavioural"]["with_fall_classifier"]
    assert with_classifier["fall:post_impact_silence"]["detection_rate"] == 1.0
    assert with_classifier["fall:post_impact_inactivity"]["detection_rate"] == 1.0
    assert with_classifier["fall:post_impact_silence"]["latency_s_max"] <= 10  # NF1
    assert with_classifier["prolonged_water:prolonged_water"]["detection_rate"] == 1.0
    assert with_classifier["impact_burst:repeated_impacts"]["detection_rate"] == 1.0
    assert [r["calibration_hours"] for r in results["calibration_curve"]] == [12, 48]
    text = report(results, synthetic=True)
    assert "Synthetic data" in text and "edge_category" in text


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
                 "--parts", "frame_level", "--out", str(tmp_path / "out")]) == 0
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert results["data"]["calibration_hours"] == pytest.approx(48, abs=0.1)
    assert (tmp_path / "out" / "report.md").read_text().startswith("# Anomaly detection evaluation")
