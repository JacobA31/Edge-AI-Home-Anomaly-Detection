"""Offline evaluation of the anomaly module, following the SINS replay protocol (proposal 1.6.1).

  cd ml
  python -m src.anomaly.evaluate --calibration sins/day1.npz sins/day2.npz sins/day3.npz \\
      --test sins/day5.npz sins/day6.npz --falls safe/falls.npz --out reports/anomaly
  python -m src.anomaly.evaluate --synthetic --out reports/anomaly-synthetic   # harness smoke run

Inputs are frame-stream .npz files (see edge/src/e3_edge/anomaly/frames.py); `--falls` holds the
SAFE fall clips' frames with gaps of >5 s between clips. Writes results.json and report.md.

What it measures (all against held-out days, never the calibration days):
1. frame level: each method's score separates SAFE falls from SINS frames (AUC, pAUC) and what
   its calibrated threshold does (falls caught/missed, normal frames flagged per hour);
2. behavioural: falls, running water and impact bursts planted into the test days (inject.py)
   -> detection rate and latency; and false alerts per day on the untouched test days;
3. how much calibration audio is needed; 4. static vs adapting baseline.
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
from e3_edge.anomaly.frames import observed_seconds
from e3_edge.anomaly.simulate import FALL, HouseholdSimulator

from . import metrics
from .inject import EXPECT, Planter
from .methods import Scorer, calibrate_edge, default_methods

DAY = 86400.0
PARTS = ("frame_level", "behavioural", "calibration_curve", "adaptation")


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
    rows = np.sort(np.random.default_rng(seed).choice(len(stream), n, replace=False))
    return FrameStream(stream.timestamps[rows], stream.embeddings[rows],
                       None if stream.scores is None else stream.scores[rows], stream.score_classes,
                       None if stream.level_dbfs is None else stream.level_dbfs[rows])


# ---- 1. frame level -----------------------------------------------------------------------------

def frame_level(methods: list[Scorer], calib: FrameStream, test: FrameStream, falls: FrameStream | None,
                cfg: AnomalyConfig, quantile: float, max_test_frames: int, seed: int) -> list[dict]:
    sample = _sample(test, max_test_frames, seed)
    sample_hours = hours(test, cfg) * len(sample) / len(test)
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
        rows.append(row)
    return rows


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


def behavioural(baseline: Baseline, test: FrameStream, falls: FrameStream | None, cfg: AnomalyConfig,
                per_kind: int, seed: int) -> dict[str, Any]:
    days = hours(test, cfg) / 24
    clean = run_detector(baseline, test).findings
    out: dict[str, Any] = {"test_days": round(days, 2), "untouched_test_days": _per_day(clean, days)}
    if test.scores is None:
        out["planted"] = "skipped: test frames have no YAMNet class scores"
        return out
    planter = Planter(test, falls, seed)
    planter.schedule(per_kind)
    out["available"] = planter.available()
    for oracle in (False, True):
        candidates = [(p.start, 0.9) for p in planter.planted if p.kind == "fall"] if oracle else []
        replay = run_detector(baseline, planter.target, fall_candidates=candidates)
        key = "with_fall_classifier" if oracle else "impact_trigger_only"
        out[key] = _match(planter.planted, replay.findings, days)
        out["thresholds"] = replay.detector.thresholds.to_dict()
    return out


def _match(planted: list, findings: list, days: float) -> dict[str, Any]:
    """A finding detects a planted anomaly if it comes after the planting starts and either began
    in the planted interval or was raised during it (a burst's first counted impact may be a real
    one from just before)."""
    used = set()
    results: dict[str, list] = {}
    for p in planted:
        end = p.start + p.length
        for kind in EXPECT[p.kind]:
            hits = [i for i, f in enumerate(findings) if f.kind == kind and f.detected_at >= p.start
                    and (p.start - 5 <= f.started_at <= end or f.detected_at <= end + 5)]
            used.update(hits)
            latency = findings[hits[0]].detected_at - p.start if hits else None
            results.setdefault(f"{p.kind}:{kind}", []).append(latency)
    summary = {}
    for key, latencies in results.items():
        found = [x for x in latencies if x is not None]
        summary[key] = {"planted": len(latencies), "detected": len(found),
                        "detection_rate": round(len(found) / len(latencies), 3),
                        "latency_s_median": round(float(np.median(found)), 1) if found else None,
                        "latency_s_max": round(float(np.max(found)), 1) if found else None}
    stray = [f for i, f in enumerate(findings) if i not in used and f.role == "alert"]
    summary["other_alerts_per_day"] = round(len(stray) / days, 2)
    summary["other_alerts"] = {"by_kind": dict(Counter(f.kind for f in stray)),
                               "confident": sum(f.confidence >= 0.5 for f in stray)}
    return summary


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


# ---- data, report, CLI ------------------------------------------------------------------------------

def synthetic_data(dim: int, seed: int) -> tuple[FrameStream, FrameStream, FrameStream]:
    """3 calibration days, 3 test days and 40 two-frame fall clips from the simulator."""
    sim = HouseholdSimulator(dim=dim, hop=0.96, seed=seed)
    t = np.concatenate([10 * DAY + i * 60 + np.array([0.0, 0.96]) for i in range(40)])
    return sim.stream(0, 3 * DAY), sim.stream(3 * DAY, 3 * DAY), sim.frames(t, np.full(len(t), FALL, np.int8))


def evaluate(calib: FrameStream, test: FrameStream, falls: FrameStream | None, *,
             cfg: AnomalyConfig | None = None, methods: list[Scorer] | None = None,
             quantile: float = 0.999, per_kind: int = 10, max_test_frames: int = 50_000,
             curve_hours: tuple = (6, 12, 24, 48, 72, 120), seed: int = 0, parts: tuple = PARTS) -> dict:
    cfg = cfg or AnomalyConfig()
    has_scores = calib.scores is not None
    if has_scores and test.scores is None:
        raise ValueError("calibration has class scores but test frames don't")
    results: dict[str, Any] = {"data": {
        "calibration_hours": round(hours(calib, cfg), 2), "test_hours": round(hours(test, cfg), 2),
        "fall_frames": 0 if falls is None else len(falls), "embedding_dim": calib.dim,
        "class_scores": has_scores, "levels": calib.level_dbfs is not None}}
    if "frame_level" in parts:
        results["frame_level"] = frame_level(methods or default_methods(has_scores), calib, test, falls, cfg,
                                             quantile, max_test_frames, seed)
    if "behavioural" in parts:
        baseline = calibrate_edge(calib, cfg, "category" if has_scores else "global")
        results["behavioural"] = behavioural(baseline, test, falls, cfg, per_kind, seed)
    if "calibration_curve" in parts:
        results["calibration_curve"] = calibration_curve(calib, test, falls, cfg, list(curve_hours), seed)
    if "adaptation" in parts:
        results["adaptation"] = adaptation(calib, test, cfg)
    return results


def report(results: dict, synthetic: bool) -> str:
    lines = ["# Anomaly detection evaluation", ""]
    if synthetic:
        lines += ["> **Synthetic data.** These numbers come from simulated frames and only show that the "
                  "harness and code paths work. They say nothing about real accuracy.", ""]
    lines += ["## Data", "", "```", json.dumps(results["data"], indent=2), "```", ""]
    if "frame_level" in results:
        cols = ["method", "edge_ready", "auc", "pauc_0.1", "tpr_at_fpr_1%", "anomalies_caught",
                "anomalies_missed", "false_alarm_frames_per_hour", "model_mb", "score_us_per_frame"]
        lines += ["## 1. Frame level: SAFE falls vs held-out household frames", "",
                  _table(results["frame_level"], cols), ""]
    if "behavioural" in results:
        b = results["behavioural"]
        lines += ["## 2. Behavioural detectors", "", f"Untouched test days ({b['test_days']}):", "",
                  "```", json.dumps(b["untouched_test_days"], indent=2), "```", ""]
        for key in ("impact_trigger_only", "with_fall_classifier"):
            if isinstance(b.get(key), dict):
                rows = [{"planted:finding": k, **v} for k, v in b[key].items() if isinstance(v, dict)]
                lines += [f"### Planted anomalies, {key.replace('_', ' ')}", "",
                          _table(rows, ["planted:finding", "planted", "detected", "detection_rate",
                                        "latency_s_median", "latency_s_max"]),
                          "", f"Other alerts per day: {b[key]['other_alerts_per_day']} "
                          f"({json.dumps(b[key]['other_alerts'])})", ""]
    if "calibration_curve" in results:
        cols = ["calibration_hours", "components", "test_frames_flagged", "target_flagged", "auc",
                "alerts_per_day", "confident_alerts_per_day"]
        lines += ["## 3. Calibration length", "", _table(results["calibration_curve"], cols), ""]
    if "adaptation" in results:
        lines += ["## 4. Static vs adapting baseline", "",
                  "```", json.dumps(results["adaptation"], indent=2), "```", ""]
    return "\n".join(lines)


def _table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", nargs="+", type=Path, default=[])
    p.add_argument("--test", nargs="+", type=Path, default=[])
    p.add_argument("--falls", type=Path)
    p.add_argument("--synthetic", action="store_true", help="simulated data (smoke test of the harness)")
    p.add_argument("--dim", type=int, default=128, help="embedding size for --synthetic")
    p.add_argument("--decimate", type=int, default=1,
                   help="keep every n-th frame (e.g. 2 turns a 0.48 s hop into 0.96 s)")
    p.add_argument("--quantile", type=float, default=0.999)
    p.add_argument("--per-kind", type=int, default=10, help="planted anomalies of each kind")
    p.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlflow", action="store_true", help="also log metrics to MLflow")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if args.synthetic:
        calib, test, falls = synthetic_data(args.dim, args.seed)
    else:
        if not args.calibration or not args.test:
            p.error("give --calibration and --test frame streams, or --synthetic")
        calib = concat([FrameStream.load(f) for f in args.calibration])
        test = concat([FrameStream.load(f) for f in args.test])
        falls = FrameStream.load(args.falls) if args.falls else None
    if args.decimate > 1:
        calib, test = _every(calib, args.decimate), _every(test, args.decimate)
    results = evaluate(calib, test, falls, quantile=args.quantile, per_kind=args.per_kind,
                       parts=tuple(args.parts), seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    (args.out / "report.md").write_text(report(results, args.synthetic))
    if args.mlflow:
        _log_mlflow(results, args)
    print(f"Wrote {args.out / 'report.md'}")
    return 0


def _every(stream: FrameStream, n: int) -> FrameStream:
    rows = np.arange(0, len(stream), n)
    return FrameStream(stream.timestamps[rows], stream.embeddings[rows],
                       None if stream.scores is None else stream.scores[rows], stream.score_classes,
                       None if stream.level_dbfs is None else stream.level_dbfs[rows])


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
