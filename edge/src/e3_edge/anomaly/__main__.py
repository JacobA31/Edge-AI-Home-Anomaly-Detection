"""Command line for the anomaly module (works on a laptop or on the Pi).

  python -m e3_edge.anomaly calibrate day1.npz day2.npz ... --out baseline.npz
  python -m e3_edge.anomaly replay day4.npz ... --baseline baseline.npz --findings findings.jsonl
  python -m e3_edge.anomaly info baseline.npz
  python -m e3_edge.anomaly bench [--soak-days 7]

Frame stream files are described in frames.py. `--config` reads the `anomaly:` section of an
edge config.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .baseline import Baseline
from .calibration import THRESHOLD, BaselineCalibrator, split_calibration
from .config import AnomalyConfig
from .detector import AnomalyDetector
from .frames import FrameStream, iter_streams, observed_seconds
from .simulate import HouseholdSimulator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m e3_edge.anomaly", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="edge config.yaml with an `anomaly:` section")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="learn a baseline from frame streams, in time order")
    p.add_argument("streams", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stats-hours", type=float, help="phase 1 length (default: all but the last day)")
    p.add_argument("--threshold-hours", type=float,
                   help="phase 2 length (default: the last whole day; 25%% with under 2 days)")
    p.add_argument("--mic-id")

    p = sub.add_parser("replay", help="run the detector over frame streams")
    p.add_argument("streams", nargs="+", type=Path)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--findings", type=Path, help="write findings as JSON lines")
    p.add_argument("--scores", type=Path, help="write per-frame distance/level/active as .npz")

    p = sub.add_parser("info", help="summarise a baseline file")
    p.add_argument("baseline", type=Path)

    p = sub.add_parser("bench", help="latency, memory and model size on synthetic frames")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--frames", type=int, default=5000, help="frames to time")
    p.add_argument("--calibration-hours", type=float, default=2.0)
    p.add_argument("--soak-days", type=float, default=0.0, help="also run this many simulated days")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text()).get("anomaly", {}) if args.config else {}
    cfg = AnomalyConfig.from_dict(raw)
    result = {"calibrate": _calibrate, "replay": _replay, "info": _info, "bench": _bench}[args.command](
        args, cfg)
    print(json.dumps(result, indent=2, default=_jsonable))
    return 0


def _load(paths: list[Path]) -> list[FrameStream]:
    return [FrameStream.load(p) for p in paths]


def _calibrate(args: argparse.Namespace, cfg: AnomalyConfig) -> dict[str, Any]:
    streams = _load(args.streams)
    total = sum(observed_seconds(s.timestamps, cfg.frame_seconds, cfg.max_gap_seconds) for s in streams)
    stats, threshold = split_calibration(total)
    if args.stats_hours is not None:
        stats = args.stats_hours * 3600
    if args.threshold_hours is not None:
        threshold = args.threshold_hours * 3600
    cfg = replace(cfg, calibration=replace(cfg.calibration, stats_seconds=stats, threshold_seconds=threshold))
    cal = BaselineCalibrator(cfg, mic_id=args.mic_id)
    started = time.perf_counter()
    for frame in iter_streams(streams):
        cal.add(frame)
    if cal.phase not in (THRESHOLD, "done"):
        raise SystemExit(f"not enough audio: {cal.progress()}")
    baseline = cal.finalize()
    size = baseline.save(args.out)
    return {"baseline": str(args.out), "size_mb": round(size / 1e6, 2),
            "seconds": round(time.perf_counter() - started, 1), **baseline.summary()}


def _replay(args: argparse.Namespace, cfg: AnomalyConfig) -> dict[str, Any]:
    baseline = Baseline.load(args.baseline)
    detector = AnomalyDetector(baseline, cfg if args.config else None)
    streams = _load(args.streams)
    rows = {"t": [], "distance": [], "level": [], "anomalous": [], "active": []}
    findings: list[dict[str, Any]] = []
    latency = []
    for frame in iter_streams(streams):
        start = time.perf_counter()
        out = detector.process(frame)
        latency.append(time.perf_counter() - start)
        for key, value in (("t", out.t), ("distance", out.distance), ("level", out.level),
                           ("anomalous", out.is_anomalous), ("active", out.active)):
            rows[key].append(value)
        for f in out.findings:
            findings.append({"detected_at": f.detected_at, **f.to_event_fields()})
    if args.findings:
        args.findings.write_text("".join(json.dumps(f) + "\n" for f in findings))
    if args.scores:
        np.savez(args.scores, **{k: np.asarray(v) for k, v in rows.items()})
    hours = sum(observed_seconds(s.timestamps, cfg.frame_seconds, cfg.max_gap_seconds)
                for s in streams) / 3600
    return {"frames": len(latency), "observed_hours": round(hours, 2),
            "anomalous_share": round(float(np.mean(rows["anomalous"])), 5) if latency else None,
            "findings": dict(Counter(f["details"]["kind"] for f in findings)),
            "findings_per_day": round(len(findings) / (hours / 24), 2) if hours else None,
            "latency_ms": _percentiles(latency)}


def _info(args: argparse.Namespace, cfg: AnomalyConfig) -> dict[str, Any]:
    return {"file_mb": round(args.baseline.stat().st_size / 1e6, 2), **Baseline.load(args.baseline).summary()}


def _bench(args: argparse.Namespace, cfg: AnomalyConfig) -> dict[str, Any]:
    sim = HouseholdSimulator(dim=args.dim, seed=1)
    hours = args.calibration_hours
    stats, threshold = split_calibration(hours * 3600)
    cfg = replace(cfg, calibration=replace(cfg.calibration, stats_seconds=stats, threshold_seconds=threshold,
                                           min_frames=500, min_threshold_frames=200))
    start = 8 * 3600.0  # 08:00 on day 0
    cal = BaselineCalibrator(cfg)
    began = time.perf_counter()
    for chunk in sim.chunks(start, hours * 3600):
        for frame in chunk.frames():
            cal.add(frame)
    calibration_s = time.perf_counter() - began
    baseline = cal.finalize()
    with tempfile.TemporaryDirectory() as tmp:
        size = baseline.save(Path(tmp) / "baseline.npz")

    detector = AnomalyDetector(baseline, cfg)
    frames = list(sim.stream(start + hours * 3600, args.frames * sim.hop).frames())
    tracemalloc.start()
    latency = []
    for frame in frames:
        began = time.perf_counter()
        detector.process(frame)
        latency.append(time.perf_counter() - began)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report: dict[str, Any] = {
        "dim": args.dim,
        "calibration": {"frames": cal.n_frames, "frames_per_s": round(cal.n_frames / calibration_s)},
        "detection": {"frames": len(latency), "latency_ms": _percentiles(latency),
                      "realtime_factor": round(sim.hop / float(np.mean(latency)), 1),
                      "python_heap_peak_mb": round(peak / 1e6, 2)},
        "baseline_file_mb": round(size / 1e6, 2),
        "process_max_rss_mb": round(_max_rss_bytes() / 1e6, 1),
        "budgets": {"frame_under_200ms": bool(np.max(latency) < 0.2), "model_under_20mb": size < 20e6},
    }
    if args.soak_days:
        report["soak"] = _soak(detector, sim, start + hours * 3600 + args.frames * sim.hop, args.soak_days)
    return report


def _soak(detector: AnomalyDetector, sim: HouseholdSimulator, start: float, days: float) -> dict[str, Any]:
    """Runs simulated days through the detector, sampling memory once per simulated day."""
    tracemalloc.start()
    memory, frames = [], 0
    alerts, confident, evidence = Counter(), Counter(), Counter()
    next_mark = start
    began = time.perf_counter()
    for chunk in sim.chunks(start, days * 86400):
        for frame in chunk.frames():
            out = detector.process(frame)
            frames += 1
            for f in out.findings:
                (alerts if f.role == "alert" else evidence)[f.kind] += 1
                confident[f.kind] += f.role == "alert" and f.confidence >= 0.5
            if frame.t >= next_mark:
                memory.append(round(tracemalloc.get_traced_memory()[0] / 1e6, 3))
                next_mark += 86400
    tracemalloc.stop()
    return {"days": days, "frames": frames, "wall_s": round(time.perf_counter() - began, 1),
            "heap_mb_each_day": memory, "heap_growth_mb": round(memory[-1] - memory[0], 3) if memory else 0,
            "on_normal_synthetic_days": {"alerts": dict(alerts), "confident_alerts": dict(+confident),
                                         "evidence": dict(evidence)}}


def _percentiles(seconds: list[float]) -> dict[str, float] | None:
    if not seconds:
        return None
    ms = np.asarray(seconds) * 1000
    return {"p50": round(float(np.percentile(ms, 50)), 3), "p99": round(float(np.percentile(ms, 99)), 3),
            "max": round(float(ms.max()), 3)}


def _max_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # Linux reports KiB


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


if __name__ == "__main__":
    sys.exit(main())
