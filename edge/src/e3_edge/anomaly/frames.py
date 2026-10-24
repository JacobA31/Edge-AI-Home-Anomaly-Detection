"""What the anomaly module consumes: one YAMNet frame at a time.

Frames are produced upstream (capture -> preprocessing -> YAMNet); this module never sees raw
audio. For offline replay (e.g. SINS, proposal 1.6.1) the same frames are stored in a
"frame stream" .npz file with these arrays:

    timestamps     float64 [N]       window start, Unix seconds (UTC), strictly increasing
    embeddings     float32 [N, D]    YAMNet embeddings (D = 1024)
    scores         float32 [N, C]    optional YAMNet class scores
    score_classes  int32   [C]       optional YAMNet class index of each scores column
                                     (default: all 521 classes in class-map order)
    level_dbfs     float32 [N]       optional RMS level of each window, from frame_level_dbfs()

Arrays only (loaded with allow_pickle=False), so a stream file can never carry code.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_CLASSES = 521  # YAMNet / AudioSet classes


@dataclass(frozen=True, slots=True)
class Frame:
    t: float  # window start, Unix seconds (UTC)
    embedding: np.ndarray  # (D,) float32
    scores: np.ndarray | None = None  # (521,) YAMNet class scores
    level_dbfs: float | None = None


def frame_level_dbfs(waveform: np.ndarray) -> float:
    """RMS level of one window of float audio in [-1, 1], in dBFS (silence -> -120)."""
    x = np.asarray(waveform, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return max(-120.0, 20.0 * np.log10(rms)) if rms > 0 else -120.0


@dataclass
class FrameStream:
    timestamps: np.ndarray
    embeddings: np.ndarray
    scores: np.ndarray | None = None
    score_classes: np.ndarray | None = None
    level_dbfs: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        n = len(self.timestamps)
        if self.embeddings.ndim != 2 or len(self.embeddings) != n:
            raise ValueError("embeddings must be [N, D] with one row per timestamp")
        if n > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=np.float32)
            if self.scores.ndim != 2 or len(self.scores) != n:
                raise ValueError("scores must be [N, C] with one row per timestamp")
            if self.score_classes is None:
                if self.scores.shape[1] != NUM_CLASSES:
                    raise ValueError("scores without score_classes must have all 521 columns")
                self.score_classes = np.arange(NUM_CLASSES, dtype=np.int32)
            self.score_classes = np.asarray(self.score_classes, dtype=np.int32)
            if self.score_classes.shape != (self.scores.shape[1],):
                raise ValueError("score_classes must list one class index per scores column")
            if self.score_classes.min(initial=0) < 0 or self.score_classes.max(initial=0) >= NUM_CLASSES:
                raise ValueError("score_classes must be YAMNet class indices (0..520)")
        if self.level_dbfs is not None:
            self.level_dbfs = np.asarray(self.level_dbfs, dtype=np.float32)
            if self.level_dbfs.shape != (n,):
                raise ValueError("level_dbfs must have one value per timestamp")

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def frames(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.frame(i)

    def frame(self, i: int) -> Frame:
        scores = None
        if self.scores is not None:
            scores = np.zeros(NUM_CLASSES, dtype=np.float32)
            scores[self.score_classes] = self.scores[i]
        level = None if self.level_dbfs is None else float(self.level_dbfs[i])
        return Frame(float(self.timestamps[i]), self.embeddings[i], scores, level)

    def full_scores(self) -> np.ndarray | None:
        """All 521 class columns (zeros where the file kept only a subset)."""
        if self.scores is None:
            return None
        full = np.zeros((len(self), NUM_CLASSES), dtype=np.float32)
        full[:, self.score_classes] = self.scores
        return full

    def slice(self, start: int, stop: int) -> FrameStream:
        return FrameStream(
            self.timestamps[start:stop],
            self.embeddings[start:stop],
            None if self.scores is None else self.scores[start:stop],
            self.score_classes,
            None if self.level_dbfs is None else self.level_dbfs[start:stop],
        )

    def save(self, path: str | Path) -> None:
        arrays = {"timestamps": self.timestamps, "embeddings": self.embeddings}
        if self.scores is not None:
            arrays |= {"scores": self.scores, "score_classes": self.score_classes}
        if self.level_dbfs is not None:
            arrays["level_dbfs"] = self.level_dbfs
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> FrameStream:
        with np.load(path, allow_pickle=False) as data:
            missing = {"timestamps", "embeddings"} - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing array(s) {sorted(missing)}")
            arrays = {k: data[k] for k in data.files}
        return cls(arrays["timestamps"], arrays["embeddings"], arrays.get("scores"),
                   arrays.get("score_classes"), arrays.get("level_dbfs"))


def iter_streams(streams: Iterable[FrameStream]) -> Iterator[Frame]:
    """Frames from several streams (e.g. one file per day) in time order. Overlaps are an error."""
    ordered = sorted((s for s in streams if len(s)), key=lambda s: s.timestamps[0])
    last_t = -np.inf
    for stream in ordered:
        if stream.timestamps[0] <= last_t:
            raise ValueError("frame streams overlap in time")
        yield from stream.frames()
        last_t = stream.timestamps[-1]


def observed_seconds(timestamps: np.ndarray, frame_seconds: float, max_gap_seconds: float) -> float:
    """Seconds of audio the frames cover, counting gaps longer than max_gap as one frame."""
    if len(timestamps) == 0:
        return 0.0
    steps = np.diff(np.asarray(timestamps, dtype=np.float64))
    steps = np.where(steps > max_gap_seconds, frame_seconds, steps)
    return float(steps.sum() + frame_seconds)
