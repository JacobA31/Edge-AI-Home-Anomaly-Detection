import json
import math
from pathlib import Path

import numpy as np
import pytest

from e3_edge.anomaly.categories import CategoryMapper, CategorySpec, resolve
from e3_edge.anomaly.config import (
    AnomalyConfig,
    ImpactRateConfig,
    PostImpactConfig,
    ProlongedSoundConfig,
    SustainedConfig,
)
from e3_edge.anomaly.temporal import (
    Clock,
    EpisodeTracker,
    Finding,
    ImpactRate,
    PostImpactMonitor,
    ProlongedSound,
    Reservoir,
    SustainedUnusual,
    TemporalEngine,
    TemporalStats,
    derive_thresholds,
    hours_by_period,
    poisson_limit,
)
from e3_edge.events import make_event

SCHEMA = Path(__file__).resolve().parents[2] / "shared" / "event.schema.json"
HOP = 0.5


def test_episode_tracker_hysteresis_and_gap_tolerance():
    tr = EpisodeTracker(CategorySpec(("Water",), on=0.5, off=0.2, gap_seconds=2.0), frame_seconds=1.0)
    scores = [0.1, 0.6, 0.3, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]  # one dip of 1.5 s is bridged
    events = [tr.update(i * HOP, s) for i, s in enumerate(scores)]
    assert [i for i, (onset, _) in enumerate(events) if onset] == [1]
    ended = [(i, d) for i, (_, d) in enumerate(events) if d is not None]
    # ends once more than 2 s pass without the score reaching "off" (t=5.0 > last_on 2.5 + 2);
    # lasted from t=0.5 to last_on=2.5, plus one frame length
    assert ended == [(10, pytest.approx(2.5 - 0.5 + 1.0))]
    assert not tr.active


def test_episode_needs_the_on_threshold_to_start():
    tr = EpisodeTracker(CategorySpec(("Water",), on=0.5, off=0.2), frame_seconds=1.0)
    assert not any(tr.update(i * HOP, 0.4)[0] for i in range(10))


def run_water(threshold, seconds, cfg=None):
    tr = EpisodeTracker(CategorySpec(("Water",), on=0.5, off=0.2, gap_seconds=5), frame_seconds=HOP)
    det = ProlongedSound(cfg or ProlongedSoundConfig(), threshold)
    out = []
    for i in range(int(seconds / HOP)):
        tr.update(i * HOP, 0.8)
        f = det.update(i * HOP, tr)
        if f:
            out.append(f)
    return out


def test_prolonged_water_fires_at_the_threshold_then_escalates_once():
    findings = run_water(threshold=60, seconds=200)
    assert [f.extra["stage"] for f in findings] == [1, 2]
    assert findings[0].value == pytest.approx(60, abs=HOP)
    assert findings[0].confidence == pytest.approx(0.5, abs=0.01)
    assert findings[1].value == pytest.approx(120, abs=HOP) and findings[1].confidence == pytest.approx(1.0)
    assert findings[0].detected_at - findings[0].started_at <= 60  # one frame after the crossing
    assert run_water(threshold=60, seconds=59) == []


def test_impact_rate_window_cooldown_and_period_limits():
    det = ImpactRate(ImpactRateConfig(window_seconds=60, cooldown_seconds=120), {"evening": 5}, default=3)
    hits = [det.update(t, True, "morning") for t in (0, 10, 20)]
    assert hits[:2] == [None, None] and hits[2].value == 3 and hits[2].threshold == 3
    assert det.update(30, True, "morning") is None  # cooldown
    assert det.update(200, True, "morning") is None  # old onsets left the window
    det = ImpactRate(ImpactRateConfig(window_seconds=60), {"evening": 5}, default=3)
    assert [det.update(t, True, "evening") is not None for t in range(0, 50, 10)] == [False] * 4 + [True]


def test_impact_rate_learns_window_maxima_including_quiet_windows():
    samples = {"day": Reservoir(100, np.random.default_rng(0))}
    det = ImpactRate(ImpactRateConfig(window_seconds=60), samples=samples)
    impacts = {5.0, 7.0, 9.0, 130.0}
    for i in range(int(250 / HOP)):
        t = i * HOP
        det.update(t, t in impacts, "day")
    assert samples["day"].items == [3, 0, 1, 0]


def test_poisson_limit():
    # 8 impacts/h, 5-minute window: expected 0.667 others; 24 onsets a day; target 0.05/day
    assert poisson_limit(8 / 12, 24, 0.05) == 6
    assert poisson_limit(0.0, 24, 0.05) == 2  # a lone impact never alerts while impacts happen daily
    assert poisson_limit(5.0, 0, 0.05) == 1


def quiet_run(monitor, start, seconds, active=lambda t: False):
    out = []
    t = start
    while t < start + seconds:
        out += monitor.update(t, HOP, active(t))
        t += HOP
    return out


def test_post_impact_silence_then_inactivity():
    mon = PostImpactMonitor(PostImpactConfig(), extended_seconds=300)
    mon.trigger(0.0, 0.3, "impact")  # weak trigger: learned limit applies
    findings = quiet_run(mon, 0.0, 400)
    assert [f.kind for f in findings] == ["post_impact_silence", "post_impact_inactivity"]
    silence, inactivity = findings
    assert silence.role == "evidence" and silence.detected_at <= 1.5 + 6.0 + HOP  # within NF1's 10 s
    assert inactivity.role == "alert" and inactivity.threshold == 300 and inactivity.confidence == 0.3
    assert mon.run is None


def test_strong_trigger_uses_the_short_cap():
    mon = PostImpactMonitor(PostImpactConfig(strong_max_seconds=120), extended_seconds=3600)
    mon.trigger(0.0, 0.95, "impact")
    alert = [f for f in quiet_run(mon, 0.0, 200) if f.role == "alert"][0]
    assert alert.threshold == 120 and 120 <= alert.detected_at <= 1.5 + 120 + HOP  # grace + limit


def test_activity_ends_the_watch_and_is_learned():
    samples = Reservoir(10, np.random.default_rng(0))
    mon = PostImpactMonitor(PostImpactConfig(), samples=samples)
    mon.trigger(0.0, 0.3, "impact")
    findings = quiet_run(mon, 0.0, 120, active=lambda t: t > 30)
    assert [f.kind for f in findings] == ["post_impact_silence"]
    assert mon.run is None and 30 < samples.items[0] < 45


def test_quiet_is_measured_only_once_the_impact_is_over():
    mon = PostImpactMonitor(PostImpactConfig(), extended_seconds=3600)
    mon.trigger(0.0, 0.9, "impact")
    out = []
    for i in range(40):
        t = i * HOP
        settling = t < 4.0  # a long impact tail: loud and still "impact" for 4 s
        out += mon.update(t, HOP, active=settling, settling=settling)
    silence = [f for f in out if f.kind == "post_impact_silence"]
    assert silence and silence[0].detected_at == pytest.approx(4.0 + 5.0 - HOP, abs=HOP)


def test_a_fall_candidate_waits_for_the_fall_sound_to_end():
    # Like a planted SAFE fall: the classifier fires at the start of a 3 s clip that YAMNet
    # never calls "impact" (so no settling flag), its category lingers ~1 s, then silence.
    mon = PostImpactMonitor(PostImpactConfig(), extended_seconds=3600)
    mon.trigger(0.0, 0.9, "fall_candidate")
    out = quiet_run(mon, 0.0, 200, active=lambda t: t < 4.0)
    assert [f.kind for f in out] == ["post_impact_silence", "post_impact_inactivity"]
    assert out[0].detected_at == pytest.approx(4.0 + 5.0 - HOP, abs=HOP)  # inside NF1's 10 s
    assert out[1].threshold == 120  # strong trigger: short cap


def test_a_trigger_that_never_goes_quiet_ends_the_watch():
    samples = Reservoir(10, np.random.default_rng(0))
    mon = PostImpactMonitor(PostImpactConfig(), samples=samples)
    mon.trigger(0.0, 0.9, "fall_candidate")
    assert quiet_run(mon, 0.0, 60, active=lambda t: True) == []
    assert mon.run is None and samples.items == [0.0]  # learned: no quiet after this impact


def test_moving_about_late_in_the_watch_ends_it():
    # 90 s of quiet, then 20 s of walking: only 18% of the whole watch, but someone is up.
    mon = PostImpactMonitor(PostImpactConfig(strong_max_seconds=120), extended_seconds=3600)
    mon.trigger(0.0, 0.95, "fall_candidate")
    out = quiet_run(mon, 0.0, 200, active=lambda t: 90 <= t < 110)
    assert [f.kind for f in out] == ["post_impact_silence"] and mon.run is None


def test_a_short_noise_does_not_end_the_watch():
    mon = PostImpactMonitor(PostImpactConfig(strong_max_seconds=120), extended_seconds=3600)
    mon.trigger(0.0, 0.95, "fall_candidate")
    out = quiet_run(mon, 0.0, 200, active=lambda t: 60 <= t < 63)  # a cough, a car outside
    assert [f.kind for f in out] == ["post_impact_silence", "post_impact_inactivity"]


def engine(learn=True, **sections):
    cfg = AnomalyConfig.from_dict(sections)
    return TemporalEngine(cfg, CategoryMapper(resolve(cfg.categories)), Clock(cfg.timezone, cfg.periods),
                          learn=learn)


def test_clock_stepping_back_is_a_gap_not_negative_time():
    eng = engine()
    for i in range(20):
        eng.process(1000.0 + i * HOP, None, -70.0)
    eng.notify_fall_candidate(1009.0, 0.9)
    eng.process(1010.0, None, -70.0)
    observed = sum(eng.stats.hourly_observed)
    eng.process(900.0, None, -70.0)  # NTP steps the clock back 110 s
    assert eng.post.run is None  # the watch is dropped, not fed 110 s of negative quiet
    assert sum(eng.stats.hourly_observed) == pytest.approx(observed + eng.cfg.frame_seconds)


def test_learning_measures_activity_against_the_floor_so_far():
    # A room with a fan: -40 dBFS is its silence. With the -60 dBFS default floor every frame of
    # calibration counted as activity, unlike in detection, which uses the learned floor.
    eng = engine()
    rng = np.random.default_rng(0)
    results = [eng.process(i * HOP, None, -40.0 + rng.normal(0, 0.5)) for i in range(1200)]
    assert eng.noise_floor == pytest.approx(-40.6, abs=0.3)
    assert results[0].active and not any(r.active for r in results[-500:])
    detection_floor = derive_thresholds(eng.snapshot_stats(), eng.cfg).noise_floor_dbfs
    assert detection_floor == pytest.approx(eng.noise_floor, abs=0.3)


def test_a_weaker_follow_up_thud_keeps_the_fall_strength():
    mon = PostImpactMonitor(PostImpactConfig(), extended_seconds=3600)
    mon.trigger(0.0, 0.95, "fall_candidate")
    mon.trigger(4.0, 0.3, "impact")
    assert mon.run.t0 == 4.0 and mon.run.strength == 0.95 and mon.run.source == "fall_candidate"
    mon.trigger(60.0, 0.3, "impact")
    assert mon.run.strength == 0.3


def test_fall_candidate_upgrades_a_nearby_impact_and_interrupt_forgets():
    mon = PostImpactMonitor(PostImpactConfig(), extended_seconds=3600)
    mon.trigger(10.0, 0.3, "impact")
    mon.trigger(10.8, 0.9, "fall_candidate")
    assert mon.run.t0 == 10.0 and mon.run.strength == 0.9 and mon.run.source == "fall_candidate"
    mon.interrupt()
    assert mon.run is None and quiet_run(mon, 20, 100) == []


def test_sustained_unusual_sound():
    det = SustainedUnusual(SustainedConfig(window_seconds=10, min_fraction=0.5, min_frames=5))
    flags = [i % 4 != 0 for i in range(40)]  # 75% anomalous
    hits = [det.update(i * HOP, a, 0.99) for i, a in enumerate(flags)]
    found = [h for h in hits if h]
    assert len(found) == 1  # cooldown
    assert found[0].confidence == pytest.approx(0.99) and found[0].value >= 0.5
    det = SustainedUnusual(SustainedConfig(window_seconds=10, min_fraction=0.5, min_frames=5))
    assert not any(det.update(i * HOP, i % 3 == 0, 0.99) for i in range(40))  # 33%


def test_thresholds_fall_back_to_defaults_without_data_and_learn_with_it():
    cfg = AnomalyConfig()
    stats = TemporalStats.empty(["water", "impact"], ["night", "morning", "afternoon", "evening"])
    th = derive_thresholds(stats, cfg)
    assert th.prolonged_seconds == cfg.water.default_seconds
    assert th.impact_default == cfg.impacts.default_count and th.impact_counts == {}
    assert th.post_impact_seconds == cfg.post_impact.default_seconds
    assert all(v.startswith("default") for v in th.sources.values())

    stats.durations["water"] = {"items": [30.0] * 95 + [400.0] * 5, "seen": 100}
    stats.levels = {"items": list(np.linspace(-70, -30, 200)), "seen": 200}
    stats.impact_counts["evening"] = {"items": [0] * 20 + [2], "seen": 21}
    stats.post_impact_quiet = {"items": [6.0] * 50, "seen": 50}
    th = derive_thresholds(stats, cfg)
    assert th.prolonged_seconds == pytest.approx(400 * 1.5)
    assert th.noise_floor_dbfs == pytest.approx(-66, abs=0.1)
    assert th.impact_counts["evening"] == max(cfg.impacts.min_count, 2 + 1)
    assert th.post_impact_seconds == cfg.post_impact.min_seconds  # 6 s * 1.5, clipped up


def test_clock_periods_follow_the_local_timezone():
    assert hours_by_period({"night": 0, "day": 7})["day"] == list(range(7, 24))
    clock = Clock("America/Toronto", {"night": 0, "day": 7})
    noon_utc = 1_793_102_400.0  # 2026-10-27 12:00 UTC = 08:00 in Toronto (EDT)
    assert clock.hour(noon_utc) == 8 and clock.period(noon_utc) == "day"
    assert clock.period(noon_utc - 3 * 3600) == "night"


def test_findings_become_valid_events():
    finding = Finding("prolonged_water", "anomaly", "alert", 1_793_102_400.0, 1_793_101_500.0, 0.75,
                      900.0, 600.0, "s", {"stage": 1})
    event = make_event("pi-1", "kitchen", **finding.to_event_fields())
    schema = json.loads(SCHEMA.read_text())
    assert set(event) <= set(schema["properties"])
    assert event["event_type"] in schema["properties"]["event_type"]["enum"]
    assert len(event["details"]) <= schema["properties"]["details"]["maxProperties"]
    assert event["details"]["started_at"] == "2026-10-27T11:45:00Z"
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(event, schema)
    assert not math.isnan(event["confidence"])
