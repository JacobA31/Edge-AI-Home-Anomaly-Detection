"""The baseline file: everything learned during calibration, in one .npz (proposal 1.7.3.3).

It stays on the device (it is derived from household audio), is written atomically so a power
cut can't leave half a file, and loads with allow_pickle=False. Size is a few MB for the full
1024-dimensional model, well under the 20 MB model budget (NF5).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .gaussian import EmbeddingModel, GaussianModel
from .temporal import TemporalStats, TemporalThresholds

FORMAT = "e3-anomaly-baseline"
VERSION = 1


@dataclass
class Baseline:
    embedding: EmbeddingModel
    score_sample: np.ndarray  # sorted distances of held-out calibration frames
    frame_quantile: float
    temporal: TemporalStats
    thresholds: TemporalThresholds  # as derived at calibration time (for reports)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score_sample = np.sort(np.asarray(self.score_sample, dtype=np.float32))
        if len(self.score_sample) == 0:
            raise ValueError("baseline has no calibration scores")

    def distance_threshold(self, quantile: float | None = None) -> float:
        return float(np.quantile(self.score_sample, self.frame_quantile if quantile is None else quantile))

    def level(self, distance: float) -> float:
        """Share of held-out baseline frames closer to normal than this one (0..1)."""
        return float(np.searchsorted(self.score_sample, distance, side="right")) / len(self.score_sample)

    def save(self, path: str | Path) -> int:
        """Atomically writes the baseline; returns its size in bytes."""
        arrays: dict[str, np.ndarray] = {"score_sample": self.score_sample}
        for name, comp in self.embedding.components.items():
            for key, value in comp.arrays().items():
                arrays[f"emb__{name}__{key}"] = value
        header = {
            "format": FORMAT,
            "version": VERSION,
            "mode": self.embedding.mode,
            "components": list(self.embedding.components),
            "frame_quantile": self.frame_quantile,
            "temporal": self.temporal.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "meta": self.meta,
        }
        arrays["header"] = np.array(json.dumps(header))
        return _atomic_savez(Path(path), arrays)

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        header = json.loads(str(arrays["header"]))
        if header.get("format") != FORMAT:
            raise ValueError(f"{path} is not an anomaly baseline")
        if header.get("version") != VERSION:
            raise ValueError(f"{path}: baseline version {header.get('version')} is not supported")
        components = {name: GaussianModel.from_arrays(with_prefix(arrays, f"emb__{name}__"))
                      for name in header["components"]}
        return cls(EmbeddingModel(components, header["mode"]), arrays["score_sample"],
                   header["frame_quantile"], TemporalStats.from_dict(header["temporal"]),
                   TemporalThresholds.from_dict(header["thresholds"]), header["meta"])

    def summary(self) -> dict[str, Any]:
        t = self.temporal
        observed = np.asarray(t.hourly_observed)
        days = observed.sum() / 86400
        total = float(observed.sum())
        per_category = {}
        for i, name in enumerate(t.categories):
            durations = t.durations[name]["items"]
            onsets = float(sum(row[i] for row in t.hourly_onsets))
            seconds = float(sum(row[i] for row in t.hourly_category))
            per_category[name] = {
                "episodes_per_day": round(onsets / days, 1) if days else None,
                "median_duration_s": round(float(np.median(durations)), 1) if durations else None,
                "share_of_time": round(seconds / total, 4) if total else None,
            }
        with np.errstate(invalid="ignore", divide="ignore"):
            hourly = np.where(observed > 0, np.asarray(t.hourly_active) / observed, np.nan)
        return {
            "created_at": self.meta.get("created_at"),
            "mic_id": self.meta.get("mic_id"),
            "frames": self.meta.get("n_frames"),
            "observed_hours": round(self.meta.get("observed_seconds", 0) / 3600, 2),
            "embedding": {
                "mode": self.embedding.mode,
                "dim": self.embedding.dim,
                "components": {n: {"frames": c.n_frames, "kept_dims": len(c.eigvals),
                                   "shrinkage": round(c.shrinkage, 4)}
                               for n, c in self.embedding.components.items()},
                "frame_quantile": self.frame_quantile,
                "distance_threshold": round(self.distance_threshold(), 3),
                "calibration_scores": len(self.score_sample),
            },
            "thresholds": self.thresholds.to_dict(),
            "categories": per_category,
            "active_share_by_hour": [None if np.isnan(v) else round(float(v), 3) for v in hourly],
        }


def with_prefix(arrays: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    """The arrays saved under `prefix`, with the prefix removed from their names."""
    return {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path.stat().st_size
