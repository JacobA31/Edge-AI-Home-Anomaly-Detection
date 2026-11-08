"""Synthetic household frames for tests, benchmarks and smoke runs of the evaluation harness.

NOT a substitute for SINS. It turns a simple daily routine (sleep, morning wash, cooking, TV,
hand-washing, kitchen impacts, night bathroom trips) into YAMNet-shaped frames: embeddings from
one low-rank Gaussian per activity, matching class scores and levels. It exercises every code
path at realistic frame rates; numbers measured on it say nothing about real-world accuracy.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .categories import class_names
from .frames import NUM_CLASSES, FrameStream

QUIET, FOOTSTEPS, WATER, KITCHEN, MEDIA, SPEECH, IMPACT, FALL = range(8)
CONTEXT_NAMES = ("quiet", "footsteps", "water", "kitchen", "media", "speech", "impact", "fall")
_CLASSES = {
    QUIET: ("Silence",),
    FOOTSTEPS: ("Walk, footsteps",),
    WATER: ("Water tap, faucet", "Sink (filling or washing)"),
    KITCHEN: ("Dishes, pots, and pans", "Cutlery, silverware"),
    MEDIA: ("Television",),
    SPEECH: ("Speech",),
    IMPACT: ("Thump, thud", "Clatter"),
    FALL: ("Thump, thud", "Bang"),
}
_LEVEL = {QUIET: (-62.0, 1.5), IMPACT: (-24.0, 3.0), FALL: (-15.0, 2.0)}  # others: (-38, 4)
# Background activity by hour of day (UTC), before episodes are laid on top.
_BY_HOUR = [QUIET] * 7 + [FOOTSTEPS, KITCHEN] + [SPEECH, QUIET, QUIET] + [KITCHEN] + [QUIET] * 4 \
    + [MEDIA, KITCHEN, KITCHEN] + [MEDIA] * 3 + [QUIET]
assert len(_BY_HOUR) == 24


class HouseholdSimulator:
    def __init__(self, dim: int = 1024, hop: float = 0.48, seed: int = 0, rank: int = 8):
        self.dim, self.hop, self.seed = dim, hop, seed
        rng = np.random.default_rng([seed, 999])
        base = np.abs(rng.normal(0, 0.3, dim))
        self.means = np.stack([base + _sparse_direction(rng, dim, 3.0 if c == FALL else 1.0)
                               for c in range(len(CONTEXT_NAMES))]).astype(np.float32)
        self.loadings = rng.normal(0, 0.15, (len(CONTEXT_NAMES), dim, rank)).astype(np.float32)
        index = {n: i for i, n in enumerate(class_names())}
        self.class_idx = {c: np.array([index[n] for n in names]) for c, names in _CLASSES.items()}

    # ---- whole streams -------------------------------------------------------------------
    def stream(self, start: float, seconds: float) -> FrameStream:
        t = start + np.arange(int(seconds / self.hop)) * self.hop
        return self.frames(t, self.contexts(t))

    def chunks(self, start: float, seconds: float, chunk_seconds: float = 3600.0) -> Iterator[FrameStream]:
        """The same frames as stream(), in bounded-memory pieces."""
        n_total = int(seconds / self.hop)
        per_chunk = max(1, int(chunk_seconds / self.hop))
        for i in range(0, n_total, per_chunk):
            t = start + np.arange(i, min(i + per_chunk, n_total)) * self.hop
            yield self.frames(t, self.contexts(t))

    def contexts(self, t: np.ndarray) -> np.ndarray:
        ctx = np.array(_BY_HOUR, dtype=np.int8)[((t % 86400) // 3600).astype(int)]
        for day in np.unique(t // 86400):
            for begin, end, c in self._day_plan(int(day)):
                ctx[(t >= begin) & (t < end)] = c
        return ctx

    def frames(self, t: np.ndarray, ctx: np.ndarray) -> FrameStream:
        rng = np.random.default_rng([self.seed, int(t[0] * 1000) % (2**32), len(t)])
        n = len(t)
        emb = np.empty((n, self.dim), dtype=np.float32)
        scores = rng.uniform(0, 0.03, (n, NUM_CLASSES)).astype(np.float32)
        level = np.empty(n, dtype=np.float32)
        for c in np.unique(ctx):
            rows = np.flatnonzero(ctx == c)
            z = rng.normal(size=(len(rows), self.loadings.shape[2])).astype(np.float32)
            emb[rows] = self.means[c] + z @ self.loadings[c].T + rng.normal(0, 0.05, (len(rows), self.dim))
            cols = self.class_idx[c]
            scores[np.ix_(rows, cols)] = rng.uniform(0.35, 0.85, (len(rows), len(cols)))
            mean, sd = _LEVEL.get(int(c), (-38.0, 4.0))
            level[rows] = rng.normal(mean, sd, len(rows))
        return FrameStream(t, np.maximum(emb, 0.0), scores, None, level)

    # ---- scenarios -------------------------------------------------------------------------
    def put(self, stream: FrameStream, start: float, seconds: float, context: int) -> None:
        """Overwrites frames in [start, start + seconds) with the given activity, in place."""
        rows = np.flatnonzero((stream.timestamps >= start) & (stream.timestamps < start + seconds))
        if len(rows):
            fresh = self.frames(stream.timestamps[rows], np.full(len(rows), context, dtype=np.int8))
            stream.embeddings[rows] = fresh.embeddings
            stream.scores[rows] = fresh.scores
            stream.level_dbfs[rows] = fresh.level_dbfs

    # Lengths are a hair under whole hops so float rounding can't add a frame.
    def put_fall(self, stream: FrameStream, t: float, quiet_seconds: float) -> None:
        """A two-frame fall-like impact at t, then quiet (as if the person isn't moving)."""
        self.put(stream, t, 1.999 * self.hop, FALL)
        self.put(stream, t + 1.999 * self.hop, quiet_seconds, QUIET)

    def put_impacts(self, stream: FrameStream, t: float, count: int, spacing: float) -> None:
        for k in range(count):
            self.put(stream, t + k * spacing, 0.999 * self.hop, IMPACT)

    # ---- routine ---------------------------------------------------------------------------
    def _day_plan(self, day: int) -> list[tuple[float, float, int]]:
        rng = np.random.default_rng([self.seed, day])
        d0 = day * 86400.0
        plan: list[tuple[float, float, int]] = []

        def add(begin: float, length: float, c: int) -> None:
            plan.append((d0 + begin, d0 + begin + length, c))

        add(7.3 * 3600 + rng.uniform(0, 1800), rng.uniform(360, 720), WATER)  # shower
        add(18.6 * 3600 + rng.uniform(0, 1800), rng.uniform(180, 480), WATER)  # dishes
        for _ in range(rng.poisson(8)):  # hand washing
            add(rng.uniform(7, 22) * 3600, float(np.clip(rng.lognormal(3.6, 0.4), 10, 120)), WATER)
        for _ in range(rng.poisson(30)):
            add(rng.uniform(7, 23) * 3600, rng.uniform(10, 60), FOOTSTEPS)
        if rng.random() < 0.5:  # night bathroom trip
            begin = rng.uniform(2, 5) * 3600
            add(begin, 30, FOOTSTEPS)
            add(begin + 30, rng.uniform(20, 60), WATER)
            add(begin + 90, 30, FOOTSTEPS)
        for hour, rate in enumerate(self._impacts_per_hour()):
            for _ in range(rng.poisson(rate)):
                add(hour * 3600 + rng.uniform(0, 3600), self.hop, IMPACT)
        return plan

    @staticmethod
    def _impacts_per_hour() -> list[float]:
        return [0.1] * 7 + [3, 6] + [1] * 3 + [6] + [1] * 4 + [1, 8, 8] + [1] * 3 + [0.1]


def _sparse_direction(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    v = np.zeros(dim)
    idx = rng.choice(dim, max(4, dim // 16), replace=False)
    v[idx] = rng.normal(0, scale, len(idx))
    return v
