"""Calibration, persistence, the service, and the detector end to end on synthetic days."""

import json

import numpy as np
import pytest
from conftest import DAY, HOUR, calibrate, calibration_config

from e3_edge.anomaly import (
    AnomalyConfig,
    AnomalyDetector,
    AnomalyService,
    Baseline,
    BaselineCalibrator,
    CalibrationIncomplete,
    FrameStream,
)
from e3_edge.anomaly.__main__ import main as cli
from e3_edge.anomaly.calibration import split_calibration
from e3_edge.anomaly.simulate import WATER


def gaussian_stream(start, seconds, dim=8, hop=0.5, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    t = start + np.arange(int(seconds / hop)) * hop
    return FrameStream(t, rng.normal(size=(len(t), dim)) + shift)


# ---- calibration ---------------------------------------------------------------------------

def test_calibration_phases_follow_observed_time():
    cal = BaselineCalibrator(calibration_config(stats_hours=1, threshold_hours=1))
    frames = list(gaussian_stream(0, 2.5 * HOUR).frames())
    for f in frames[: int(0.9 * HOUR / 0.5)]:
        cal.add(f)
    assert cal.phase == "stats"
    with pytest.raises(CalibrationIncomplete):
        cal.finalize()
    for f in frames[int(0.9 * HOUR / 0.5):]:
        cal.add(f)
    assert cal.done and cal.progress()["fraction"] == 1.0
    assert cal.n_frames == pytest.approx(2 * HOUR / 0.5, abs=2)  # frames after "done" are ignored


def test_threshold_phase_is_a_whole_day_when_possible():
    assert split_calibration(4 * DAY) == (3 * DAY, DAY)
    assert split_calibration(2 * DAY) == (DAY, DAY)
    assert split_calibration(8 * HOUR) == (6 * HOUR, 2 * HOUR)


def test_mute_gaps_do_not_count_as_calibration_time():
    cal = BaselineCalibrator(calibration_config(stats_hours=1, threshold_hours=1))
    for s in (gaussian_stream(0, 0.5 * HOUR), gaussian_stream(5 * HOUR, 0.5 * HOUR, seed=1)):
        for f in s.frames():
            cal.add(f)
    assert cal.observed == pytest.approx(HOUR, abs=1.0)
    with pytest.raises(ValueError, match="time order"):
        cal.add(next(gaussian_stream(0, 1).frames()))


def test_threshold_is_the_held_out_quantile():
    cfg = calibration_config(stats_hours=1, threshold_hours=2, embedding={"frame_quantile": 0.99})
    cal = BaselineCalibrator(cfg)
    for f in gaussian_stream(0, 3 * HOUR).frames():
        cal.add(f)
    baseline = cal.finalize()
    fresh = gaussian_stream(10 * HOUR, 2 * HOUR, seed=9)
    distances = baseline.embedding.score_batch(fresh.embeddings)
    share = np.mean(distances > baseline.distance_threshold())
    assert 0.005 < share < 0.02  # about 1%
    assert baseline.level(baseline.distance_threshold()) == pytest.approx(0.99, abs=0.001)
    assert baseline.level(1e9) == 1.0 and baseline.level(0.0) == 0.0


def test_baseline_file_round_trip_and_size(tmp_path, sim):
    from e3_edge.anomaly.simulate import HouseholdSimulator

    big = HouseholdSimulator(dim=1024, hop=0.96, seed=2)
    cal = calibrate(big, calibration_config(stats_hours=1.5, threshold_hours=0.5,
                                            calibration={"min_frames": 500, "min_threshold_frames": 200}),
                    start=8 * HOUR, seconds=2 * HOUR)
    baseline = cal.finalize()
    size = baseline.save(tmp_path / "b.npz")
    assert size < 20e6  # NF5: deployed models under 20 MB
    loaded = Baseline.load(tmp_path / "b.npz")
    probe = big.stream(11 * HOUR, 60).embeddings
    np.testing.assert_allclose(loaded.embedding.score_batch(probe), baseline.embedding.score_batch(probe),
                               rtol=1e-6)
    assert loaded.summary()["embedding"] == baseline.summary()["embedding"]
    assert set(loaded.embedding.components) == set(baseline.embedding.components)
    assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves nothing behind


def test_checkpoint_resume_gives_the_same_model(tmp_path, sim):
    cfg = calibration_config(stats_hours=2, threshold_hours=1, calibration={"min_frames": 500})
    frames = list(sim.stream(6 * HOUR, 3 * HOUR).frames())
    whole = BaselineCalibrator(cfg)
    for f in frames:
        whole.add(f)
    half = BaselineCalibrator(cfg)
    for f in frames[: len(frames) // 2]:
        half.add(f)
    half.checkpoint(tmp_path / "ckpt.npz")
    resumed = BaselineCalibrator.resume(tmp_path / "ckpt.npz", cfg)
    assert resumed.progress() == half.progress()
    for f in frames[len(frames) // 2:]:
        resumed.add(f)
    a, b = whole.finalize(), resumed.finalize()
    assert set(a.embedding.components) == set(b.embedding.components)
    for name, comp in a.embedding.components.items():
        np.testing.assert_allclose(b.embedding.components[name].mean, comp.mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(b.embedding.components[name].eigvals, comp.eigvals, rtol=1e-3)
    with pytest.raises(ValueError, match="settings changed"):
        BaselineCalibrator.resume(tmp_path / "ckpt.npz", calibration_config(stats_hours=5))


# ---- service ---------------------------------------------------------------------------------

def test_service_calibrates_resumes_after_restart_then_detects(tmp_path, sim):
    cfg = calibration_config(stats_hours=1.5, threshold_hours=0.5,
                             calibration={"min_frames": 500, "checkpoint_seconds": 600},
                             baseline_path=str(tmp_path / "baseline.npz"),
                             checkpoint_path=str(tmp_path / "calibration.npz"))
    frames = list(sim.stream(8 * HOUR, 2.5 * HOUR).frames())
    service = AnomalyService(cfg, mic_id="near")
    assert all(service.process(f) is None for f in frames[:3000])
    assert (tmp_path / "calibration-near.npz").exists()

    service = AnomalyService(cfg, mic_id="near")  # "restart"
    assert service.state == "calibrating" and service.status()["frames"] > 2000
    outputs = [service.process(f) for f in frames[3000:]]
    assert service.state == "detecting"
    assert (tmp_path / "baseline-near.npz").exists() and not (tmp_path / "calibration-near.npz").exists()
    assert outputs[-1] is not None

    service = AnomalyService(cfg, mic_id="near")
    assert service.state == "detecting" and service.process(frames[-1]) is not None


# ---- detector on a scenario day ------------------------------------------------------------

TEST_DAY = 2 * DAY  # the baseline saw days 0-1
FALL_AT, WATER_AT, BURST_AT = TEST_DAY + 14.5 * HOUR, TEST_DAY + 10.2 * HOUR, TEST_DAY + 15.5 * HOUR


@pytest.fixture(scope="module")
def scenario_day(sim, day_baseline):
    day = sim.stream(TEST_DAY, DAY)
    sim.put_fall(day, FALL_AT, 600)
    sim.put(day, WATER_AT, 2400, WATER)
    sim.put_impacts(day, BURST_AT, 8, 20)
    detector = AnomalyDetector(day_baseline)
    outputs = [detector.process(f) for f in day.frames()]
    return detector, outputs, [f for o in outputs for f in o.findings]


def test_prolonged_water_is_flagged_at_the_learned_threshold(scenario_day):
    detector, _, findings = scenario_day
    water = [f for f in findings if f.kind == "prolonged_water"]
    assert [f.extra["stage"] for f in water] == [1, 2]
    limit = detector.thresholds.prolonged_seconds
    assert water[0].started_at == pytest.approx(WATER_AT, abs=1)
    assert water[0].detected_at - water[0].started_at <= limit + 1  # NF1: well within 5 minutes


def test_impact_burst_is_flagged(scenario_day):
    _, _, findings = scenario_day
    burst = [f for f in findings if f.kind == "repeated_impacts"]
    assert len(burst) == 1 and BURST_AT <= burst[0].detected_at <= BURST_AT + 160


def test_fall_gets_silence_evidence_in_time_and_an_inactivity_alert(scenario_day):
    _, outputs, findings = scenario_day
    fall_frame = next(o for o in outputs if o.t >= FALL_AT)
    assert fall_frame.is_anomalous and fall_frame.level > 0.999 and "impact" in fall_frame.categories
    evidence = [f for f in findings if f.kind == "post_impact_silence" and abs(f.started_at - FALL_AT) < 2]
    assert evidence and evidence[0].detected_at - FALL_AT <= 10  # NF1 fall budget
    alert = [f for f in findings if f.kind == "post_impact_inactivity" and abs(f.started_at - FALL_AT) < 2]
    assert alert and alert[0].confidence > 0.9 and alert[0].detected_at - FALL_AT <= 125


def test_no_confident_alerts_on_ordinary_parts_of_the_day(scenario_day):
    _, outputs, findings = scenario_day
    injected = [(WATER_AT, WATER_AT + 2400), (FALL_AT, FALL_AT + 700), (BURST_AT, BURST_AT + 200)]
    stray = [f for f in findings if f.role == "alert" and f.confidence >= 0.5
             and not any(a <= f.started_at <= b for a, b in injected)]
    assert stray == []
    assert np.mean([o.is_anomalous for o in outputs]) < 0.01


def test_mute_is_not_mistaken_for_silence(sim, day_baseline):
    day = sim.stream(TEST_DAY + 14 * HOUR, HOUR)
    sim.put_fall(day, FALL_AT, 600)
    detector = AnomalyDetector(day_baseline)
    findings = []
    for f in day.frames():
        out = detector.process(f)
        findings += out.findings
        if FALL_AT + 3 <= f.t < FALL_AT + 4:
            detector.pause()  # resident presses mute right after the thud
    assert not [f for f in findings if f.kind.startswith("post_impact") and abs(f.started_at - FALL_AT) < 2]


def test_runtime_config_changes_thresholds_without_recalibrating(day_baseline):
    strict = AnomalyDetector(day_baseline, AnomalyConfig.from_dict(
        {**day_baseline.meta["config"], "embedding": {"frame_quantile": 0.99}}))
    default = AnomalyDetector(day_baseline)
    assert strict.threshold < default.threshold
    with pytest.raises(ValueError, match="dims"):
        default.process(next(gaussian_stream(0, 1, dim=5).frames()))


def test_adaptation_learns_gated_frames_and_keeps_the_previous_baseline(tmp_path, sim, day_baseline):
    path = tmp_path / "baseline.npz"
    day_baseline.save(path)
    raw = dict(day_baseline.meta["config"])
    raw["adaptation"] = {"enabled": True, "interval_seconds": 2 * HOUR, "weight": 0.5}
    raw["calibration"] = {**raw["calibration"], "min_frames": 500, "min_threshold_frames": 200}
    detector = AnomalyDetector(day_baseline, AnomalyConfig.from_dict(raw), baseline_path=path)
    hours = sim.stream(3 * DAY + 9 * HOUR, 2.5 * HOUR)
    sim.put_fall(hours, 3 * DAY + 9.5 * HOUR, 60)
    for f in hours.frames():
        detector.process(f)
    assert detector.baseline_updates == 1
    assert detector.baseline.meta["adapted_from"] == day_baseline.meta["created_at"]
    assert path.with_suffix(".prev.npz").exists()
    assert Baseline.load(path).meta["adapted_from"] == day_baseline.meta["created_at"]


def test_adapting_on_unchanged_sounds_keeps_the_threshold():
    cfg = calibration_config(stats_hours=1, threshold_hours=2,
                             adaptation={"enabled": True, "interval_seconds": 4 * HOUR, "weight": 0.5})
    cal = BaselineCalibrator(cfg)
    for f in gaussian_stream(0, 3 * HOUR).frames():
        cal.add(f)
    detector = AnomalyDetector(cal.finalize())
    before = detector.threshold
    for f in gaussian_stream(4 * HOUR, 4.2 * HOUR, seed=3).frames():
        detector.process(f)
    assert detector.baseline_updates == 1
    # same distribution, so the threshold must stay put (gating the threshold sample cut it ~12%)
    assert detector.threshold == pytest.approx(before, rel=0.06)


# ---- command line ----------------------------------------------------------------------------

def test_cli_calibrate_replay_info_bench(tmp_path, sim, capsys):
    for i, start in enumerate((0, 12 * HOUR, DAY)):
        sim.stream(start, 12 * HOUR).save(tmp_path / f"part{i}.npz")
    assert cli(["calibrate", str(tmp_path / "part0.npz"), str(tmp_path / "part1.npz"),
                "--out", str(tmp_path / "b.npz")]) == 0
    calibrated = json.loads(capsys.readouterr().out)
    assert calibrated["observed_hours"] == pytest.approx(24, abs=0.1)

    assert cli(["replay", str(tmp_path / "part2.npz"), "--baseline", str(tmp_path / "b.npz"),
                "--findings", str(tmp_path / "f.jsonl"), "--scores", str(tmp_path / "s.npz")]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["frames"] == len(FrameStream.load(tmp_path / "part2.npz"))
    assert replayed["latency_ms"]["p99"] < 200
    lines = (tmp_path / "f.jsonl").read_text().splitlines()
    assert all(json.loads(line)["event_type"] in ("anomaly", "inactivity") for line in lines)
    assert len(np.load(tmp_path / "s.npz")["distance"]) == replayed["frames"]

    assert cli(["info", str(tmp_path / "b.npz")]) == 0
    assert json.loads(capsys.readouterr().out)["embedding"]["mode"] == "category"

    assert cli(["bench", "--dim", "16", "--frames", "300", "--calibration-hours", "1"]) == 0
    bench = json.loads(capsys.readouterr().out)
    assert bench["budgets"] == {"frame_under_200ms": True, "model_under_20mb": True}
