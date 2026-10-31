"""Learning the baseline from a calibration period (FR4, proposal 1.7.3.3).

Calibration runs in two phases over observed audio time:

1. "stats": accumulate the mean and covariance of embeddings (plus the temporal statistics).
2. "threshold": score frames the model has *not* seen against it, to pick the anomaly threshold
   from honest held-out distances rather than optimistic in-sample ones.

Nothing is stored per frame, so memory stays flat however long calibration runs. Progress is
checkpointed atomically, so a power cut or crash after five days of a seven-day calibration
resumes instead of starting over (NF4).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import Baseline, _atomic_savez, with_prefix
from .categories import CategoryMapper, resolve
from .config import AnomalyConfig
from .frames import Frame
from .gaussian import EmbeddingModel, GaussianModel, RunningMoments, oas_shrink
from .temporal import Clock, Reservoir, TemporalEngine, TemporalStats, derive_thresholds

log = logging.getLogger(__name__)

STATS, THRESHOLD, DONE = "stats", "threshold", "done"


class CalibrationIncomplete(RuntimeError):
    pass


class BaselineCalibrator:
    def __init__(self, cfg: AnomalyConfig, *, prior: Baseline | None = None, prior_weight: float = 0.0,
                 mic_id: str | None = None, checkpoint_path: str | Path | None = None):
        self.cfg = cfg
        self.prior = prior
        self.prior_weight = prior_weight
        self.mic_id = mic_id
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.rng = np.random.default_rng(cfg.calibration.seed)
        self.mapper = CategoryMapper(resolve(cfg.categories))
        self.clock = Clock(cfg.timezone, cfg.periods)
        self.temporal = TemporalEngine(cfg, self.mapper, self.clock, learn=True, rng=self.rng)
        self.moments: dict[str, RunningMoments] = {}
        self.model: EmbeddingModel | None = None
        self.scores = Reservoir(cfg.calibration.score_reservoir, self.rng)
        self.phase = STATS
        self.observed = 0.0
        self.n_frames = 0
        self.first_t: float | None = None
        self.last_t: float | None = None
        self._checkpointed_at = 0.0

    @property
    def stats_seconds(self) -> float:
        return self.cfg.calibration.stats_seconds

    @property
    def total_seconds(self) -> float:
        return self.cfg.calibration.stats_seconds + self.cfg.calibration.threshold_seconds

    @property
    def done(self) -> bool:
        return self.phase == DONE

    def progress(self) -> dict[str, Any]:
        return {"phase": self.phase, "frames": self.n_frames,
                "observed_hours": round(self.observed / 3600, 2),
                "needed_hours": round(self.total_seconds / 3600, 2),
                "fraction": round(min(1.0, self.observed / self.total_seconds), 4)}

    def add(self, frame: Frame) -> None:
        if self.phase == DONE:
            return
        t = frame.t
        if self.last_t is not None and t <= self.last_t:
            raise ValueError("frames must arrive in time order")
        gap = self.last_t is None or t - self.last_t > self.cfg.max_gap_seconds
        self.observed += self.cfg.frame_seconds if gap else t - self.last_t
        self.last_t = t
        self.first_t = t if self.first_t is None else self.first_t
        self.n_frames += 1
        self.temporal.process(t, frame.scores, frame.level_dbfs)

        c = self.cfg.calibration
        if self.phase == STATS:
            for name in self._contexts(frame):
                if name not in self.moments:
                    self.moments[name] = RunningMoments(len(frame.embedding), c.batch_size)
                self.moments[name].add(frame.embedding)
            if self.observed >= self.stats_seconds and self.moments["global"].n >= c.min_frames:
                self._fit()
                self.phase = THRESHOLD
                log.info("Calibration: model fitted on %d frames, now setting the threshold",
                         self.model.components["global"].n_frames)
        else:
            self.scores.add(self.model.score(frame.embedding)[0])
            if self.observed >= self.total_seconds:
                self.phase = DONE
                log.info("Calibration complete after %.1f h of audio", self.observed / 3600)

        if self.checkpoint_path and self.observed - self._checkpointed_at >= c.checkpoint_seconds:
            self.checkpoint()

    def _contexts(self, frame: Frame) -> list[str]:
        mode = self.cfg.embedding.contexts
        if mode == "period":
            return ["global", self.clock.period(frame.t)]
        if mode == "category" and frame.scores is not None:
            category = self.mapper.dominant(frame.scores)
            if category is not None:
                return ["global", category]
        return ["global"]

    def _fit(self) -> None:
        e = self.cfg.embedding
        components = {}
        for name, moments in self.moments.items():
            if name == "global":
                dims = e.n_components
            elif moments.n < e.min_component_frames:
                log.info("Calibration: only %d frames of %r, no separate component", moments.n, name)
                continue
            else:  # few frames: keep only as many directions as they can support
                dims = min(e.component_dims, max(4, moments.n // 20))
            prior = self.prior.embedding.components.get(name) if self.prior else None
            if prior is not None and prior.dim == moments.dim:
                cov, _ = oas_shrink(moments.covariance(), moments.n)
                components[name] = prior.blend(moments.mean, cov, self.prior_weight, moments.n, dims, e.ridge)
            else:
                components[name] = GaussianModel.fit(moments, dims, e.ridge)
        if self.prior is not None:  # adapting: sounds absent from this round keep their old model
            for name, comp in self.prior.embedding.components.items():
                components.setdefault(name, comp)
        self.model = EmbeddingModel(components, e.contexts)
        self.moments.clear()  # the covariance accumulators are the biggest objects; free them

    def finalize(self) -> Baseline:
        if self.phase == STATS:
            raise CalibrationIncomplete(
                f"still learning the mean and covariance ({self.observed / 3600:.1f} h of "
                f"{self.stats_seconds / 3600:.1f} h, {self.n_frames} frames)")
        if len(self.scores.items) < self.cfg.calibration.min_threshold_frames:
            raise CalibrationIncomplete(
                f"only {len(self.scores.items)} held-out frames to set the threshold "
                f"(need {self.cfg.calibration.min_threshold_frames})")
        stats = self.temporal.snapshot_stats()
        if self.prior is not None:
            stats = self.prior.temporal.merged(stats, self.prior_weight, self.rng,
                                               self.cfg.calibration.sample_reservoir)
        meta = {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mic_id": self.mic_id,
            "n_frames": self.n_frames,
            "observed_seconds": self.observed,
            "first_frame": self.first_t,
            "last_frame": self.last_t,
            "adapted_from": self.prior.meta.get("created_at") if self.prior else None,
            "config": self.cfg.to_dict(),
        }
        return Baseline(self.model, np.asarray(self.scores.items), self.cfg.embedding.frame_quantile,
                        stats, derive_thresholds(stats, self.cfg), meta)

    def checkpoint(self, path: str | Path | None = None) -> int:
        path = Path(path) if path else self.checkpoint_path
        if path is None:
            raise ValueError("no checkpoint path")
        arrays: dict[str, np.ndarray] = {"scores": np.asarray(self.scores.items, dtype=np.float64)}
        for name, moments in self.moments.items():
            for key, value in moments.state().items():
                # float32 halves the file (8 accumulators x 4 MB) for one rounding per checkpoint
                arrays[f"mom__{name}__{key}"] = value.astype(np.float32) if key == "m2" else value
        if self.model is not None:
            for name, comp in self.model.components.items():
                for key, value in comp.arrays().items():
                    arrays[f"emb__{name}__{key}"] = value
        state = {
            "phase": self.phase, "observed": self.observed, "n_frames": self.n_frames,
            "first_t": self.first_t, "last_t": self.last_t, "mic_id": self.mic_id,
            "scores_seen": self.scores.seen, "moments": list(self.moments),
            "components": list(self.model.components) if self.model else [],
            "temporal": self.temporal.snapshot_stats().to_dict(),
            "settings": _calibration_fingerprint(self.cfg),
        }
        arrays["state"] = np.array(json.dumps(state))
        size = _atomic_savez(path, arrays)
        self._checkpointed_at = self.observed
        return size

    @classmethod
    def resume(cls, path: str | Path, cfg: AnomalyConfig,
               checkpoint_path: str | Path | None = None) -> BaselineCalibrator:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        state = json.loads(str(arrays["state"]))
        if state["settings"] != _calibration_fingerprint(cfg):
            raise ValueError("calibration settings changed since the checkpoint; delete it to restart")
        cal = cls(cfg, mic_id=state["mic_id"], checkpoint_path=checkpoint_path or path)
        cal.phase, cal.observed, cal.n_frames = state["phase"], state["observed"], state["n_frames"]
        cal.first_t, cal.last_t = state["first_t"], state["last_t"]
        cal._checkpointed_at = cal.observed
        cal.scores = Reservoir(cfg.calibration.score_reservoir, cal.rng,
                               arrays["scores"].tolist(), state["scores_seen"])
        for name in state["moments"]:
            cal.moments[name] = RunningMoments.from_state(with_prefix(arrays, f"mom__{name}__"),
                                                          cfg.calibration.batch_size)
        if state["components"]:
            cal.model = EmbeddingModel(
                {name: GaussianModel.from_arrays(with_prefix(arrays, f"emb__{name}__"))
                 for name in state["components"]}, cfg.embedding.contexts)
        cal.temporal = TemporalEngine(cfg, cal.mapper, cal.clock, learn=True, rng=cal.rng,
                                      stats=TemporalStats.from_dict(state["temporal"]))
        return cal


def _calibration_fingerprint(cfg: AnomalyConfig) -> dict[str, Any]:
    """Settings that must not change in the middle of a calibration."""
    c = cfg.calibration
    return {"embedding": cfg.to_dict()["embedding"], "timezone": cfg.timezone, "periods": cfg.periods,
            "stats_seconds": c.stats_seconds, "threshold_seconds": c.threshold_seconds}


def split_calibration(total_seconds: float) -> tuple[float, float]:
    """(stats, threshold) seconds for a calibration of this length.

    The threshold phase is the last whole day whenever there are two days or more, so every hour
    of the routine is in it: a threshold set on evenings only flags too much of every morning.
    """
    threshold = 86400.0 if total_seconds >= 2 * 86400 else 0.25 * total_seconds
    return total_seconds - threshold, threshold


def shadow_config(cfg: AnomalyConfig) -> AnomalyConfig:
    """Calibration settings for an adaptation round over `adaptation.interval_seconds`."""
    stats, threshold = split_calibration(cfg.adaptation.interval_seconds)
    calibration = replace(cfg.calibration, stats_seconds=stats, threshold_seconds=threshold)
    return replace(cfg, calibration=calibration)
