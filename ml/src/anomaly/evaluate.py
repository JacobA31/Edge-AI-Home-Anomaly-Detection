"""Offline evaluation of the anomaly module, following the SINS replay protocol (proposal 1.6.1).

  cd ml
  python -m src.anomaly.evaluate --calibration sins/day1.npz sins/day2.npz sins/day3.npz \\
      --test sins/day5.npz sins/day6.npz --falls safe/falls.npz --non-falls safe/non_falls.npz \\
      --out reports/anomaly
  python -m src.anomaly.evaluate --synthetic --out reports/anomaly-synthetic   # harness smoke run

Inputs are frame-stream .npz files (see edge/src/e3_edge/anomaly/frames.py); `--falls` and
`--non-falls` hold SAFE clips' frames with gaps of >5 s between clips. Writes results.json and
report.md.

What it measures (all against held-out days, never the calibration days):
1. frame level: each method's score separates SAFE falls from SINS frames (AUC, pAUC) and what
   its calibrated threshold does (falls caught/missed, normal frames flagged per hour, the home's
   own impact/water frames flagged). SAFE non-falls, recorded the same way as the falls, show how
   much of the separation is the recording setup rather than the fall;
2. behavioural, per embedding context: falls, running water and impact bursts planted into the
   test days (inject.py) -> detection rate and latency; and false alerts per day on the
   untouched test days. The fall classifier isn't built yet, so stand-ins notify the planted
   falls: an ideal one, and weaker ones that miss falls, are late or fire falsely;
3. how much calibration audio is needed; 4. static vs adapting baseline;
5. (`--parts classifier`) the behavioural results against the stand-in's delay, recall and false
   triggers; 6. (`--parts tuning`) the same against threshold settings, recalibrating for each.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from e3_edge.anomaly import AnomalyConfig, AnomalyDetector, Baseline, FrameStream
from e3_edge.anomaly.categories import DEFAULT_CATEGORIES, CategoryMapper
from e3_edge.anomaly.frames import observed_seconds
from e3_edge.anomaly.simulate import FALL, HouseholdSimulator

from . import metrics
from .inject import EXPECT, Planted, Planter
from .methods import Scorer, calibrate_edge, default_methods, parse_context

DAY = 86400.0
PARTS = ("frame_level", "behavioural", "calibration_curve", "adaptation", "classifier", "tuning")
DEFAULT_PARTS = PARTS[:4]  # classifier and tuning replay the test days many times
CATEGORY_COLUMNS = ("impact", "water", "household")  # the home's own sounds, flagged share reported


def concat(streams: list[FrameStream]) -> FrameStream:
    streams = sorted((s for s in streams if len(s)), key=lambda s: s.timestamps[0])
    classes = {tuple(s.score_classes) for s in streams if s.scores is not None}
    if len(classes) > 1:
        raise ValueError("streams store different YAMNet class columns")
    has_scores = all(s.scores is not None for s in streams)
    has_level = all(s.level_dbfs is not None for s in streams)
    return FrameStream(np.concatenate([s.timestamps for s in streams]),
                       np.concatenate([s.embeddings for s in streams]),
                       np.concatenate([s.scores for s in streams]) if has_scores else None,
                       streams[0].score_classes if has_scores else None,
                       np.concatenate([s.level_dbfs for s in streams]) if has_level else None)


def hours(stream: FrameStream, cfg: AnomalyConfig) -> float:
    return observed_seconds(stream.timestamps, cfg.frame_seconds, cfg.max_gap_seconds) / 3600


def _sample(stream: FrameStream, n: int, seed: int) -> FrameStream:
    if len(stream) <= n:
        return stream
    return _rows(stream, np.sort(np.random.default_rng(seed).choice(len(stream), n, replace=False)))


# ---- 1. frame level -----------------------------------------------------------------------------

def category_rows(stream: FrameStream) -> dict[str, np.ndarray]:
    """Rows of a stream where YAMNet hears each category (score at or above its `on` level)."""
    if stream.scores is None:
        return {}
    mapper = CategoryMapper(DEFAULT_CATEGORIES)
    cats = mapper.scores_batch(stream.full_scores())
    return {name: np.flatnonzero(cats[:, i] >= DEFAULT_CATEGORIES[name].on)
            for i, name in enumerate(mapper.names) if name in CATEGORY_COLUMNS}


def frame_level(methods: list[Scorer], calib: FrameStream, test: FrameStream, falls: FrameStream | None,
                cfg: AnomalyConfig, quantile: float, max_test_frames: int, seed: int,
                non_falls: FrameStream | None = None) -> list[dict]:
    sample = _sample(test, max_test_frames, seed)
    sample_hours = hours(test, cfg) * len(sample) / len(test)
    # ADR 0002's question: are the home's own rare sounds (its thuds, its tap) flagged as unusual?
    own = {name: _rows(test, r) for name, r in category_rows(test).items()}
    rows = []
    for method in methods:
        method.fit(calib, quantile, seed)
        started = time.perf_counter()
        normal = method.score(sample)
        per_frame_us = (time.perf_counter() - started) / len(sample) * 1e6
        size = method.size_bytes()
        row: dict[str, Any] = {"method": method.name, "edge_ready": method.edge_ready,
                               "fit_s": round(method.fit_seconds, 1),
                               "score_us_per_frame": round(per_frame_us, 1),
                               "model_mb": round(size / 1e6, 2) if size else None}
        anomalous = method.score(falls) if falls is not None and len(falls) else np.array([])
        if len(anomalous):
            row |= {"auc": round(metrics.auc(normal, anomalous), 4),
                    "pauc_0.1": round(metrics.partial_auc(normal, anomalous, 0.1), 4),
                    "tpr_at_fpr_1%": round(metrics.tpr_at_fpr(normal, anomalous, 0.01), 4),
                    "tpr_at_fpr_0.1%": round(metrics.tpr_at_fpr(normal, anomalous, 0.001), 4)}
        row |= metrics.at_threshold(normal, anomalous, method.threshold, sample_hours)
        if non_falls is not None and len(non_falls):
            # Same recording setup as the falls, so a score that only reacts to the setup gives
            # non-falls the same AUC against SINS as falls, and ~0.5 falls vs non-falls.
            other = method.score(non_falls)
            row |= {"auc_non_falls": round(metrics.auc(normal, other), 4),
                    "non_falls_flagged": round(float(np.mean(other > method.threshold)), 4)}
            if len(anomalous):
                row["auc_falls_vs_non_falls"] = round(metrics.auc(other, anomalous), 4)
        for name, frames in own.items():
            if len(frames):
                row[f"flagged_{name}"] = round(float(np.mean(method.score(frames) > method.threshold)), 4)
        rows.append(row)
    return rows


def _rows(stream: FrameStream, rows: np.ndarray) -> FrameStream:
    return FrameStream(stream.timestamps[rows], stream.embeddings[rows],
                       None if stream.scores is None else stream.scores[rows], stream.score_classes,
                       None if stream.level_dbfs is None else stream.level_dbfs[rows])


# ---- 2. behavioural -----------------------------------------------------------------------------

@dataclass
class Replay:
    findings: list
    detector: AnomalyDetector
    flagged_by_day: list[float]  # share of frames the detector flagged as it ran, per day


def run_detector(baseline: Baseline, stream: FrameStream, cfg: AnomalyConfig | None = None,
                 fall_candidates: list[tuple[float, float]] = ()) -> Replay:
    """Replays a stream; `fall_candidates` (time, confidence) emulate the fall classifier."""
    detector = AnomalyDetector(baseline, cfg)
    pending = sorted(fall_candidates)
    findings = []
    flagged = np.zeros(len(stream), dtype=bool)
    for i, frame in enumerate(stream.frames()):
        out = detector.process(frame)
        flagged[i] = out.is_anomalous
        findings += out.findings
        while pending and pending[0][0] <= frame.t:
            detector.notify_fall_candidate(*pending.pop(0))
    day = ((stream.timestamps - stream.timestamps[0]) // DAY).astype(int)
    by_day = [round(float(flagged[day == d].mean()), 5) for d in range(day.max() + 1)] if len(stream) else []
    return Replay(findings, detector, by_day)


def _per_day(findings: list, days: float) -> dict[str, float]:
    alerts = [f for f in findings if f.role == "alert"]
    return {"alerts_per_day": round(len(alerts) / days, 2),
            "confident_alerts_per_day": round(sum(f.confidence >= 0.5 for f in alerts) / days, 2),
            "evidence_per_day": round(sum(f.role == "evidence" for f in findings) / days, 2),
            "alerts_by_kind": dict(Counter(f.kind for f in alerts))}


@dataclass(frozen=True)
class ClassifierStandIn:
    """Plays the fall classifier, which isn't built yet: it notices `recall` of the planted falls,
    `delay_s` after each one starts, and fires `false_per_day` times on other sounds of the home.

    A real classifier's false alarms come from real sounds (a door, a dropped pan), not from
    silence, so false triggers land on frames where YAMNet hears an impact or household sound,
    away from anything planted. Whether quiet follows them is up to the home.
    """

    recall: float = 1.0
    delay_s: float = 0.0
    false_per_day: float = 0.0
    confidence: float = 0.9

    def candidates(self, planted: list[Planted], stream: FrameStream, sound_rows: np.ndarray,
                   days: float, rng: np.random.Generator) -> tuple[list[tuple[float, float]], list[float]]:
        """(time, confidence) notifications, and the times of the false ones."""
        falls = [p for p in planted if p.kind == "fall"]
        caught = rng.permutation(len(falls))[: round(self.recall * len(falls))]
        hits = [(falls[i].start + self.delay_s, self.confidence) for i in sorted(caught)]
        t = stream.timestamps[sound_rows]
        free = np.ones(len(t), dtype=bool)
        for p in planted:  # keep false triggers out of planted events and the quiet after them
            free &= (t < p.start - 600) | (t > p.start + p.length + 600)
        n_false = min(int(rng.poisson(self.false_per_day * days)), int(free.sum()))
        false = sorted(float(x) for x in rng.choice(t[free], n_false, replace=False)) if n_false else []
        return sorted(hits + [(x, self.confidence) for x in false]), false


SCENARIOS: dict[str, ClassifierStandIn | None] = {
    "impact_trigger_only": None,  # no classifier: only YAMNet's impact category starts a watch
    "ideal_classifier": ClassifierStandIn(),
    "weaker_classifier": ClassifierStandIn(recall=0.8, delay_s=2.0, false_per_day=5.0),
}
CLASSIFIER_SWEEP: dict[str, ClassifierStandIn | None] = {
    "ideal": ClassifierStandIn(),
    "delay_2s": ClassifierStandIn(delay_s=2.0),
    "delay_5s": ClassifierStandIn(delay_s=5.0),
    "recall_0.8": ClassifierStandIn(recall=0.8),
    "recall_0.5": ClassifierStandIn(recall=0.5),
    "false_1_per_day": ClassifierStandIn(false_per_day=1.0),
    "false_5_per_day": ClassifierStandIn(false_per_day=5.0),
    "false_20_per_day": ClassifierStandIn(false_per_day=20.0),
}


def sound_rows(stream: FrameStream) -> np.ndarray:
    """Frames where YAMNet hears an impact or a household sound: where a classifier's false alarm lands."""
    rows = category_rows(stream)
    return np.union1d(rows.get("impact", []), rows.get("household", [])).astype(int)


def behavioural(baseline: Baseline, test: FrameStream, falls: FrameStream | None, cfg: AnomalyConfig,
                per_kind: int, seeds: tuple[int, ...] = (0,),
                scenarios: dict[str, ClassifierStandIn | None] | None = None) -> dict[str, Any]:
    """Untouched test days, then planted anomalies replayed once per seed and scenario."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    days = hours(test, cfg) / 24
    clean = run_detector(baseline, test).findings
    out: dict[str, Any] = {"test_days": round(days, 2), "untouched_test_days": _per_day(clean, days)}
    if test.scores is None:
        out["planted"] = "skipped: test frames have no YAMNet class scores"
        return out
    sounds = sound_rows(test)
    tallies = {name: Tally() for name in scenarios}
    for seed in seeds:
        planter = Planter(test, falls, seed)
        planter.schedule(per_kind)
        out["available"] = planter.available()
        for k, (name, standin) in enumerate(scenarios.items()):
            candidates, false = [], []
            if standin is not None:
                rng = np.random.default_rng([seed, k])
                candidates, false = standin.candidates(planter.planted, test, sounds, days, rng)
            replay = run_detector(baseline, planter.target, fall_candidates=candidates)
            tallies[name].add(planter.planted, replay.findings, false)
        out["thresholds"] = replay.detector.thresholds.to_dict()
        del planter
    out["seeds"] = list(seeds)
    out |= {name: tally.summary(days * len(seeds)) for name, tally in tallies.items()}
    return out


class Tally:
    """Matches findings to planted anomalies over one or more replays.

    A finding belongs to a planted anomaly if it comes after the planting starts and either began
    in the planted interval or was raised during it (a burst's first counted impact may be a real
    one from just before). It *detects* it if it is of an expected kind; otherwise it is a side
    alert (e.g. inactivity after a planted burst at 5 a.m., when the home really is quiet). Alerts
    that belong to nothing planted are stray: false alerts on real audio.
    """

    def __init__(self) -> None:
        self.latencies: dict[str, list[float | None]] = {}
        self.stray: list = []
        self.side: list = []
        self.false_triggers = 0
        self.false_trigger_alerts = 0

    def add(self, planted: list[Planted], findings: list, false_triggers: list[float] = ()) -> None:
        caused = set()
        for p in planted:
            end = p.start + p.length
            mine = [i for i, f in enumerate(findings) if f.detected_at >= p.start
                    and (p.start - 5 <= f.started_at <= end or f.detected_at <= end + 5)]
            caused.update(mine)
            for kind in EXPECT[p.kind]:
                hits = [i for i in mine if findings[i].kind == kind]
                latency = findings[hits[0]].detected_at - p.start if hits else None
                self.latencies.setdefault(f"{p.kind}:{kind}", []).append(latency)
            self.side += [findings[i] for i in mine
                          if findings[i].role == "alert" and findings[i].kind not in EXPECT[p.kind]]
        stray = [f for i, f in enumerate(findings) if i not in caused and f.role == "alert"]
        self.stray += stray
        self.false_triggers += len(false_triggers)
        false = np.asarray(false_triggers)
        self.false_trigger_alerts += sum(
            1 for f in stray if f.extra.get("trigger") == "fall_candidate" and len(false)
            and np.min(np.abs(false - f.started_at)) <= 2.0)

    def summary(self, days: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, latencies in self.latencies.items():
            found = [x for x in latencies if x is not None]
            out[key] = {"planted": len(latencies), "detected": len(found),
                        "detection_rate": round(len(found) / len(latencies), 3),
                        "latency_s_median": round(float(np.median(found)), 1) if found else None,
                        "latency_s_max": round(float(np.max(found)), 1) if found else None}
        out["stray_alerts_per_day"] = round(len(self.stray) / days, 2)
        out["stray_alerts"] = {"by_kind": dict(Counter(f.kind for f in self.stray)),
                               "confident": sum(f.confidence >= 0.5 for f in self.stray)}
        out["side_alerts"] = dict(Counter(f.kind for f in self.side))
        if self.false_triggers:
            out["false_triggers"] = self.false_triggers
            out["false_trigger_alerts"] = self.false_trigger_alerts
        return out


# ---- 3. calibration length, 4. adaptation ----------------------------------------------------------

def calibration_curve(calib: FrameStream, test: FrameStream, falls: FrameStream | None, cfg: AnomalyConfig,
                      hour_steps: list[float], seed: int) -> list[dict]:
    available = hours(calib, cfg)
    sample = _sample(test, 50_000, seed)
    rows = []
    for h in [h for h in hour_steps if h <= available + 1e-6]:
        prefix = calib.slice(0, int(np.searchsorted(calib.timestamps, calib.timestamps[0] + h * 3600)))
        baseline = calibrate_edge(prefix, cfg)
        normal = baseline.embedding.score_batch(sample.embeddings)
        row = {"calibration_hours": h, "components": len(baseline.embedding.components),
               "test_frames_flagged": round(float(np.mean(normal > baseline.distance_threshold())), 5),
               "target_flagged": round(1 - baseline.frame_quantile, 5)}
        if falls is not None and len(falls):
            row["auc"] = round(metrics.auc(normal, baseline.embedding.score_batch(falls.embeddings)), 4)
        row |= _per_day(run_detector(baseline, test).findings, hours(test, cfg) / 24)
        rows.append(row)
    return rows


def adaptation(calib: FrameStream, test: FrameStream, cfg: AnomalyConfig) -> dict[str, Any]:
    """Static baseline vs one that adapts every 2 days (1 day learning + 1 whole day threshold)."""
    if hours(test, cfg) < 3 * 24 - 0.5:  # a few frames short of 72 h still counts
        return {"skipped": "needs at least three test days"}
    baseline = calibrate_edge(calib, cfg)
    adaptive_cfg = replace(baseline_config(baseline), adaptation=replace(cfg.adaptation, enabled=True,
                                                                         interval_seconds=2 * DAY))
    out = {}
    for name, run_cfg in (("static", None), ("adaptive", adaptive_cfg)):
        replay = run_detector(baseline, test, run_cfg)
        out[name] = {**_per_day(replay.findings, hours(test, cfg) / 24),
                     "baseline_updates": replay.detector.baseline_updates,
                     "frames_flagged_by_day": replay.flagged_by_day,
                     "target_flagged": round(1 - baseline.frame_quantile, 5)}
    return out


def baseline_config(baseline: Baseline) -> AnomalyConfig:
    return AnomalyConfig.from_dict(baseline.meta["config"])


# ---- 5. classifier stand-ins, 6. threshold tuning ---------------------------------------------------

# Each variant changes the defaults in one way. Calibration learns with these settings too (the
# post-impact quiet it learns depends on them), so every variant is recalibrated, as a device
# with that config would be.
TUNING_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("defaults", {}),
    ("short_4s", {"post_impact.short_seconds": 4.0}),
    ("short_3s", {"post_impact.short_seconds": 3.0}),
    ("strong_max_60s", {"post_impact.strong_max_seconds": 60.0}),
    ("strong_max_300s", {"post_impact.strong_max_seconds": 300.0}),
    ("quantile_0.995", {"embedding.frame_quantile": 0.995}),
    ("quantile_0.9995", {"embedding.frame_quantile": 0.9995}),
    ("impact_on_0.1", {"categories.impact.on": 0.1, "categories.impact.off": 0.05}),
    ("impact_on_0.3", {"categories.impact.on": 0.3, "categories.impact.off": 0.15}),
    ("impacts_target_0.01", {"impacts.target_alerts_per_day": 0.01}),
    ("impacts_min_6", {"impacts.min_count": 6}),
    ("activity_margin_3db", {"activity.margin_db": 3.0}),
    ("activity_margin_10db", {"activity.margin_db": 10.0}),
    ("timezone_utc", {"timezone": "UTC"}),  # is the result sensitive to the assumed SINS clock?
]


def with_overrides(cfg: AnomalyConfig, overrides: dict[str, Any]) -> AnomalyConfig:
    """`cfg` with dotted settings replaced, e.g. {"post_impact.short_seconds": 4}."""
    raw = cfg.to_dict()
    for key, value in overrides.items():
        *path, last = key.split(".")
        node = raw
        for part in path:
            node = node.setdefault(part, {})
        node[last] = value
    return AnomalyConfig.from_dict(raw)


def tuning(calib: FrameStream, test: FrameStream, falls: FrameStream | None, cfg: AnomalyConfig,
           context: str, per_kind: int, seeds: tuple[int, ...],
           variants: list[tuple[str, dict[str, Any]]] | None = None) -> list[dict]:
    rows = []
    for label, overrides in variants or TUNING_VARIANTS:
        variant = with_overrides(cfg, overrides)
        baseline = calibrate_edge(calib, variant, *parse_context(context))
        b = behavioural(baseline, test, falls, variant, per_kind, seeds,
                        {k: SCENARIOS[k] for k in ("impact_trigger_only", "ideal_classifier")})
        rows.append({"variant": label, "overrides": overrides, **_tuning_row(b)})
    return rows


def _tuning_row(b: dict[str, Any]) -> dict[str, Any]:
    """The numbers the threshold choice is made on, flattened for one table row."""
    clean = b["untouched_test_days"]
    row = {"alerts_per_day": clean["alerts_per_day"], "evidence_per_day": clean["evidence_per_day"],
           "alerts_by_kind": clean["alerts_by_kind"]}
    if "ideal_classifier" not in b:
        return row
    ideal, alone = b["ideal_classifier"], b["impact_trigger_only"]

    def get(summary: dict, key: str, field: str) -> Any:
        return summary.get(key, {}).get(field)

    return row | {
        "fall_silence_rate": get(ideal, "fall:post_impact_silence", "detection_rate"),
        "fall_silence_s_median": get(ideal, "fall:post_impact_silence", "latency_s_median"),
        "fall_silence_s_max": get(ideal, "fall:post_impact_silence", "latency_s_max"),
        "fall_inactivity_rate": get(ideal, "fall:post_impact_inactivity", "detection_rate"),
        "fall_inactivity_s_median": get(ideal, "fall:post_impact_inactivity", "latency_s_median"),
        "fall_silence_rate_no_classifier": get(alone, "fall:post_impact_silence", "detection_rate"),
        "water_rate": get(ideal, "prolonged_water:prolonged_water", "detection_rate"),
        "water_s_median": get(ideal, "prolonged_water:prolonged_water", "latency_s_median"),
        "burst_rate": get(ideal, "impact_burst:repeated_impacts", "detection_rate"),
        "burst_s_median": get(ideal, "impact_burst:repeated_impacts", "latency_s_median"),
        "stray_alerts_per_day": ideal["stray_alerts_per_day"],
    }


# ---- data, report, CLI ------------------------------------------------------------------------------

def synthetic_data(dim: int, seed: int) -> tuple[FrameStream, FrameStream, FrameStream]:
    """3 calibration days, 3 test days and 40 two-frame fall clips from the simulator."""
    sim = HouseholdSimulator(dim=dim, hop=0.96, seed=seed)
    t = np.concatenate([10 * DAY + i * 60 + np.array([0.0, 0.96]) for i in range(40)])
    return sim.stream(0, 3 * DAY), sim.stream(3 * DAY, 3 * DAY), sim.frames(t, np.full(len(t), FALL, np.int8))


def evaluate(calib: FrameStream, test: FrameStream, falls: FrameStream | None, *,
             non_falls: FrameStream | None = None, cfg: AnomalyConfig | None = None,
             methods: list[Scorer] | None = None, contexts: tuple[str, ...] | None = None,
             quantile: float = 0.999, per_kind: int = 10, max_test_frames: int = 50_000,
             curve_hours: tuple = (6, 12, 24, 48, 72, 120), seed: int = 0,
             plant_seeds: tuple[int, ...] | None = None, parts: tuple = DEFAULT_PARTS) -> dict:
    """`contexts`: embedding contexts the behavioural part compares (default: the config's); the
    classifier and tuning parts use the first. The calibration curve and adaptation use `cfg`."""
    cfg = cfg or AnomalyConfig()
    has_scores = calib.scores is not None
    if has_scores and test.scores is None:
        raise ValueError("calibration has class scores but test frames don't")
    contexts = contexts or ((cfg.embedding.contexts if has_scores else "global"),)
    plant_seeds = plant_seeds or (seed,)
    results: dict[str, Any] = {"data": {
        "calibration_hours": round(hours(calib, cfg), 2), "test_hours": round(hours(test, cfg), 2),
        "fall_frames": 0 if falls is None else len(falls),
        "non_fall_frames": 0 if non_falls is None else len(non_falls), "embedding_dim": calib.dim,
        "class_scores": has_scores, "levels": calib.level_dbfs is not None,
        "timezone": cfg.timezone, "contexts": list(contexts), "plant_seeds": list(plant_seeds)}}
    if "frame_level" in parts:
        results["frame_level"] = frame_level(
            methods or default_methods(has_scores, cfg), calib, test, falls, cfg,
            quantile, max_test_frames, seed, non_falls,
        )
    if "behavioural" in parts:
        results["behavioural"] = {
            name: behavioural(calibrate_edge(calib, cfg, *parse_context(name)), test, falls, cfg,
                              per_kind, plant_seeds)
            for name in contexts}
    if "calibration_curve" in parts:
        results["calibration_curve"] = calibration_curve(calib, test, falls, cfg, list(curve_hours), seed)
    if "adaptation" in parts:
        results["adaptation"] = adaptation(calib, test, cfg)
    if "classifier" in parts:
        baseline = calibrate_edge(calib, cfg, *parse_context(contexts[0]))
        results["classifier"] = behavioural(baseline, test, falls, cfg, per_kind, plant_seeds,
                                            CLASSIFIER_SWEEP)
    if "tuning" in parts:
        results["tuning"] = tuning(calib, test, falls, cfg, contexts[0], per_kind, plant_seeds)
    return results


def report(results: dict, synthetic: bool) -> str:
    lines = ["# Anomaly detection evaluation", ""]
    if synthetic:
        lines += ["> **Synthetic data.** These numbers come from simulated frames and only show that the "
                  "harness and code paths work. They say nothing about real accuracy.", ""]
    lines += ["## Data", "", "```", json.dumps(results["data"], indent=2), "```", ""]
    if "frame_level" in results:
        rows = results["frame_level"]
        cols = ["method", "edge_ready", "auc", "pauc_0.1", "tpr_at_fpr_1%", "anomalies_caught",
                "false_alarm_frames_per_hour", "model_mb", "score_us_per_frame"]
        lines += ["## 1. Frame level: SAFE falls vs held-out household frames", "", _table(rows, cols), ""]
        if any("auc_non_falls" in r for r in rows):
            lines += ["SAFE non-falls (same recordings, no fall) as a control: if non-falls separate from "
                      "SINS about as well as falls do, the separation is mostly the recording setup.", "",
                      _table(rows, ["method", "auc", "auc_non_falls", "auc_falls_vs_non_falls",
                                    "anomalies_caught", "non_falls_flagged"]), ""]
        own = [f"flagged_{c}" for c in CATEGORY_COLUMNS if any(f"flagged_{c}" in r for r in rows)]
        if own:
            lines += ["The home's own sounds flagged as anomalous (share of test frames where YAMNet hears "
                      "the category; lower is better, ADR 0002):", "", _table(rows, ["method", *own]), ""]
    if "behavioural" in results:
        by_context = results["behavioural"]
        lines += ["## 2. Behavioural detectors", ""]
        if len(by_context) > 1:
            lines += ["By embedding context (ideal classifier stand-in):", "",
                      _table([{"context": name, **_tuning_row(b)} for name, b in by_context.items()],
                             ["context", "alerts_per_day", "alerts_by_kind", "fall_silence_rate",
                              "fall_inactivity_rate", "water_rate", "burst_rate",
                              "stray_alerts_per_day"]), ""]
        name, b = next(iter(by_context.items()))
        lines += [f"### `{name}`: untouched test days ({b['test_days']})", "",
                  "```", json.dumps(b["untouched_test_days"], indent=2), "```", ""]
        lines += _scenario_tables(b, f"`{name}`, planted anomalies (seeds {b.get('seeds')})")
    if "calibration_curve" in results:
        cols = ["calibration_hours", "components", "test_frames_flagged", "target_flagged", "auc",
                "alerts_per_day", "confident_alerts_per_day"]
        lines += ["## 3. Calibration length", "", _table(results["calibration_curve"], cols), ""]
    if "adaptation" in results:
        lines += ["## 4. Static vs adapting baseline", "",
                  "```", json.dumps(results["adaptation"], indent=2), "```", ""]
    if "classifier" in results:
        c = results["classifier"]
        scenarios = [k for k, v in c.items() if isinstance(v, dict) and "stray_alerts_per_day" in v]
        rows = []
        for k in scenarios:
            s = c[k]
            rows.append({"stand_in": k, **{f"{f}_{m}": s.get(f"fall:{f}", {}).get(m)
                                          for f in ("post_impact_silence", "post_impact_inactivity")
                                          for m in ("detection_rate", "latency_s_max")},
                         "stray_alerts_per_day": s["stray_alerts_per_day"],
                         "false_triggers": s.get("false_triggers"),
                         "false_trigger_alerts": s.get("false_trigger_alerts")})
        lines += ["## 5. Fall classifier stand-ins", "",
                  "Each stand-in changes the ideal one in one way. False triggers land on the home's "
                  "own impact/household sounds.", "", _table(rows, list(rows[0]) if rows else []), ""]
    if "tuning" in results:
        cols = ["variant", "alerts_per_day", "alerts_by_kind", "evidence_per_day", "fall_silence_rate",
                "fall_silence_s_median", "fall_silence_s_max", "fall_inactivity_rate",
                "fall_inactivity_s_median", "fall_silence_rate_no_classifier", "water_rate", "water_s_median",
                "burst_rate", "burst_s_median", "stray_alerts_per_day"]
        lines += ["## 6. Threshold tuning (each variant recalibrated, ideal classifier stand-in)", "",
                  _table(results["tuning"], cols), ""]
    return "\n".join(lines)


def _scenario_tables(b: dict[str, Any], title: str) -> list[str]:
    lines = []
    for key in SCENARIOS:
        s = b.get(key)
        if not isinstance(s, dict):
            continue
        rows = [{"planted:finding": k, **v} for k, v in s.items() if isinstance(v, dict) and "planted" in v]
        lines += [f"### {title}, {key.replace('_', ' ')}", "",
                  _table(rows, ["planted:finding", "planted", "detected", "detection_rate",
                                "latency_s_median", "latency_s_max"]), "",
                  f"Stray alerts per day: {s['stray_alerts_per_day']} ({json.dumps(s['stray_alerts'])}); "
                  f"side alerts caused by planted events: {json.dumps(s['side_alerts'])}"
                  + (f"; {s['false_trigger_alerts']} of {s['false_triggers']} false triggers became alerts"
                     if "false_triggers" in s else ""), ""]
    return lines


def _table(rows: list[dict], cols: list[str]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return ""
        return json.dumps(value) if isinstance(value, dict) else str(value)

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(cell(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", nargs="+", type=Path, default=[])
    p.add_argument("--test", nargs="+", type=Path, default=[])
    p.add_argument("--falls", type=Path)
    p.add_argument("--non-falls", type=Path, help="SAFE non-fall clips: control for the recording setup")
    p.add_argument("--contexts", nargs="+",
                   help="embedding contexts the behavioural part compares, e.g. category global "
                        "global_pca64 period (default: the config's); the first is used by classifier/tuning")
    p.add_argument("--plant-seeds", nargs="+", type=int,
                   help="plant anomalies once per seed and pool the results (default: --seed)")
    p.add_argument("--synthetic", action="store_true", help="simulated data (smoke test of the harness)")
    p.add_argument("--dim", type=int, default=128, help="embedding size for --synthetic")
    p.add_argument("--decimate", type=int, default=1,
                   help="keep every n-th frame (e.g. 2 turns a 0.48 s hop into 0.96 s)")
    p.add_argument("--quantile", type=float, default=0.999)
    p.add_argument("--per-kind", type=int, default=10, help="planted anomalies of each kind")
    p.add_argument("--parts", nargs="+", default=list(DEFAULT_PARTS), choices=PARTS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlflow", action="store_true", help="also log metrics to MLflow")
    p.add_argument("--timezone", default="UTC", help="Local timezone for routines, e.g. Europe/Brussels")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    non_falls = None
    if args.synthetic:
        calib, test, falls = synthetic_data(args.dim, args.seed)
    else:
        if not args.calibration or not args.test:
            p.error("give --calibration and --test frame streams, or --synthetic")
        calib = concat([FrameStream.load(f) for f in args.calibration])
        test = concat([FrameStream.load(f) for f in args.test])
        falls = FrameStream.load(args.falls) if args.falls else None
        non_falls = FrameStream.load(args.non_falls) if args.non_falls else None
    if args.decimate > 1:
        calib, test = _every(calib, args.decimate), _every(test, args.decimate)
    for name in args.contexts or []:
        try:
            parse_context(name)
        except ValueError as e:
            p.error(str(e))
    results = evaluate(calib, test, falls, non_falls=non_falls, cfg=AnomalyConfig(timezone=args.timezone),
                       contexts=tuple(args.contexts) if args.contexts else None,
                       quantile=args.quantile, per_kind=args.per_kind, parts=tuple(args.parts),
                       seed=args.seed, plant_seeds=tuple(args.plant_seeds) if args.plant_seeds else None)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    (args.out / "report.md").write_text(report(results, args.synthetic))
    if args.mlflow:
        _log_mlflow(results, args)
    print(f"Wrote {args.out / 'report.md'}")
    return 0


def _every(stream: FrameStream, n: int) -> FrameStream:
    return _rows(stream, np.arange(0, len(stream), n))


def _log_mlflow(results: dict, args: argparse.Namespace) -> None:
    import mlflow

    with mlflow.start_run(run_name="anomaly-evaluation"):
        mlflow.log_params({"quantile": args.quantile, "synthetic": args.synthetic, **results["data"]})
        for row in results.get("frame_level", []):
            for key in ("auc", "pauc_0.1", "anomalies_missed", "false_alarm_frames_per_hour"):
                if row.get(key) is not None:
                    mlflow.log_metric(f"{row['method']}.{key}", row[key])
        mlflow.log_artifact(str(args.out / "report.md"))


if __name__ == "__main__":
    raise SystemExit(main())
