"""Planting known anomalies into normal recordings.

SINS has no falls or emergencies, so detection rates are measured by planting the proposal's
three temporal anomalies into held-out SINS days, built from real frames ("donors"):

* fall:            frames of a SAFE fall clip, then `quiet_seconds` of the home's own quietest
                   frames (no one moving afterwards)
* prolonged_water: `seconds` of the home's own running-water frames
* impact_burst:    `count` of the home's own impact frames, `spacing` seconds apart

Everything around the planted part stays real, so the background noise, TV, etc. are SINS's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from e3_edge.anomaly import FrameStream
from e3_edge.anomaly.categories import DEFAULT_CATEGORIES, CategoryMapper

EXPECT = {
    "fall": ("post_impact_silence", "post_impact_inactivity"),
    "prolonged_water": ("prolonged_water",),
    "impact_burst": ("repeated_impacts",),
}


@dataclass
class Planted:
    kind: str
    start: float
    length: float
    params: dict = field(default_factory=dict)


def copy_stream(s: FrameStream) -> FrameStream:
    return FrameStream(s.timestamps.copy(), s.embeddings.copy(),
                       None if s.scores is None else s.scores.copy(), s.score_classes,
                       None if s.level_dbfs is None else s.level_dbfs.copy())


def clips(stream: FrameStream, gap: float = 5.0) -> list[np.ndarray]:
    """Row indices of each clip in a stream of clips separated by time gaps (e.g. SAFE falls)."""
    breaks = np.flatnonzero(np.diff(stream.timestamps) > gap) + 1
    return [c for c in np.split(np.arange(len(stream)), breaks) if len(c)]


class Planter:
    def __init__(self, target: FrameStream, falls: FrameStream | None, seed: int = 0):
        if target.scores is None:
            raise ValueError("planting needs YAMNet class scores in the target stream")
        self.original = target  # donors always come from here, never from already-planted frames
        self.target = copy_stream(target)
        self.falls = falls
        self.fall_clips = clips(falls) if falls is not None else []
        self.rng = np.random.default_rng(seed)
        mapper = CategoryMapper(DEFAULT_CATEGORIES)
        cats = mapper.scores_batch(target.full_scores())
        on = {n: DEFAULT_CATEGORIES[n].on for n in mapper.names}
        col = {n: i for i, n in enumerate(mapper.names)}
        self.water = np.flatnonzero(cats[:, col["water"]] >= on["water"])
        self.impact = np.flatnonzero(cats[:, col["impact"]] >= on["impact"])
        # quiet donors: no category present, and among the quietest 30% of such frames
        silent = np.all(cats < 0.1, axis=1)
        level = target.level_dbfs if target.level_dbfs is not None else np.zeros(len(target))
        self.quiet = np.array([], dtype=int)
        if silent.any():
            self.quiet = np.flatnonzero(silent & (level <= np.quantile(level[silent], 0.3)))
        self.hop = float(np.median(np.diff(target.timestamps)))
        self.planted: list[Planted] = []

    def available(self) -> dict[str, bool]:
        return {"fall": bool(self.fall_clips) and len(self.quiet) > 0, "prolonged_water": len(self.water) > 0,
                "impact_burst": len(self.impact) > 0}

    def _rows(self, start: float, seconds: float) -> np.ndarray:
        t = self.target.timestamps
        return np.flatnonzero((t >= start) & (t < start + seconds))

    def _copy(self, rows: np.ndarray, source: FrameStream, donor_rows: np.ndarray) -> None:
        tgt = self.target
        tgt.embeddings[rows] = source.embeddings[donor_rows]
        if source.scores is not None:
            full = source.full_scores()[donor_rows]
            tgt.scores[rows] = full[:, tgt.score_classes]
        if source.level_dbfs is not None and tgt.level_dbfs is not None:
            tgt.level_dbfs[rows] = source.level_dbfs[donor_rows]

    def fall(self, start: float, quiet_seconds: float = 600.0) -> None:
        clip = self.fall_clips[self.rng.integers(len(self.fall_clips))]
        rows = self._rows(start, 1e9)[: len(clip)]
        self._copy(rows, self.falls, clip[: len(rows)])
        after = self._rows(self.target.timestamps[rows[-1]] + 1e-3, quiet_seconds)
        self._copy(after, self.original, self.rng.choice(self.quiet, len(after)))
        self.planted.append(Planted("fall", start, quiet_seconds, {"clip_frames": len(clip)}))

    def prolonged_water(self, start: float, seconds: float) -> None:
        rows = self._rows(start, seconds)
        donors = np.resize(self.water, len(rows))  # the home's water frames, cycled in time order
        self._copy(rows, self.original, donors)
        self.planted.append(Planted("prolonged_water", start, seconds))

    def impact_burst(self, start: float, count: int = 8, spacing: float = 20.0) -> None:
        for k in range(count):
            rows = self._rows(start + k * spacing, 0.999 * self.hop)  # exactly one frame
            self._copy(rows, self.original, self.rng.choice(self.impact, len(rows)))
        self.planted.append(Planted("impact_burst", start, count * spacing, {"count": count}))

    def schedule(self, per_kind: int, margin: float = 3600.0, water_seconds: float = 2700.0) -> None:
        """Plants `per_kind` of each available anomaly at random, non-overlapping times."""
        t0, t1 = self.target.timestamps[0] + margin, self.target.timestamps[-1] - margin
        slots = np.arange(t0, t1, 2 * margin)
        kinds = [k for k, ok in self.available().items() if ok]
        wanted = [k for k in kinds for _ in range(per_kind)]
        if len(wanted) > len(slots):
            wanted = wanted[: len(slots)]
        chosen = np.sort(self.rng.choice(slots, len(wanted), replace=False))
        for kind, start in zip(self.rng.permutation(wanted), chosen, strict=True):
            start = float(start + self.rng.uniform(0, margin / 2))
            if kind == "fall":
                self.fall(start)
            elif kind == "prolonged_water":
                self.prolonged_water(start, water_seconds)
            else:
                self.impact_burst(start)
