"""The real fall classifier in the harness, in place of the stand-ins (task 9).

The classifier scores one YAMNet embedding at a time (e3_edge.fall_classifier), so it scores the
same frames the anomaly module sees. It is turned into fall-candidate notifications the way the
main loop would have to: one per burst of frames at or above the threshold, sent once the
burst's first window has been heard. Notifying on every such frame would restart the post-fall
watch (PostImpactMonitor.trigger starts a new run for a trigger more than merge_seconds after
the last), and a 3 s SAFE fall has frames above the threshold up to 2.4 s apart.

SAFE clips in falls.npz / non_falls.npz have no names. export_frames.py wrote them in sorted
filename order, so the WAV folder gives each clip's fold (the first two digits of its name):
the fall classifier trained on folds 01-07, validated on 08 and was tested on 09-10.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from e3_edge.anomaly import FrameStream

from .inject import clips

NOTIFY_WINDOW_SECONDS = 10.0  # a notification this soon after a planted fall's start is for it


class FrameScores:
    """Scores frames once. A planted copy of the test days re-scores only the rows planting changed."""

    def __init__(self, score_batch: Callable[[np.ndarray], np.ndarray]):
        self._score_batch = score_batch
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []  # (embeddings, scores), by identity

    def _score(self, embeddings: np.ndarray) -> np.ndarray:
        scores = np.asarray(self._score_batch(embeddings), dtype=np.float64).reshape(-1)
        if len(scores) != len(embeddings) or not np.all((scores >= 0) & (scores <= 1)):
            raise ValueError("the fall classifier must give one score in [0, 1] per frame")
        return scores

    def of(self, stream: FrameStream) -> np.ndarray:
        for embeddings, scores in self._cache:
            if embeddings is stream.embeddings:
                return scores
        scores = self._score(stream.embeddings)
        self._cache.append((stream.embeddings, scores))
        return scores

    def planted(self, stream: FrameStream, original: FrameStream, touched: np.ndarray) -> np.ndarray:
        """Scores of `stream`, a copy of `original` with only the `touched` rows changed."""
        scores = self.of(original).copy()
        if len(touched):
            scores[touched] = self._score(stream.embeddings[touched])
        return scores


def notifications(timestamps: np.ndarray, scores: np.ndarray, threshold: float, frame_seconds: float,
                  rearm_seconds: float) -> list[tuple[float, float]]:
    """(time, confidence) per burst of frames scoring >= threshold. A frame within rearm_seconds of
    the previous such frame belongs to the same burst. The time is when the burst's first window
    has been heard (its start + frame_seconds), the confidence that frame's score."""
    above = np.flatnonzero(scores >= threshold)
    out: list[tuple[float, float]] = []
    last = -np.inf
    for i in above:
        t = float(timestamps[i])
        if t - last > rearm_seconds:
            out.append((t + frame_seconds, float(scores[i])))
        last = t
    return out


@dataclass(frozen=True)
class RealClassifier:
    scores: FrameScores
    threshold: float
    frame_seconds: float = 0.96
    rearm_seconds: float = 5.0

    def candidates(self, stream: FrameStream, original: FrameStream | None = None,
                   touched: np.ndarray | None = None) -> list[tuple[float, float]]:
        if original is None:
            scores = self.scores.of(stream)
        else:
            scores = self.scores.planted(stream, original, np.asarray(touched, dtype=int))
        return notifications(stream.timestamps, scores, self.threshold, self.frame_seconds,
                             self.rearm_seconds)


def split_candidates(candidates: list[tuple[float, float]], fall_starts: list[float]) -> list[float]:
    """Times of the candidates that aren't for a planted fall: false triggers."""
    starts = np.asarray(sorted(fall_starts))
    false = []
    for t, _ in candidates:
        k = np.searchsorted(starts, t, side="right") - 1
        if k < 0 or t - starts[k] > NOTIFY_WINDOW_SECONDS:
            false.append(t)
    return false


def load(model_path: Path) -> tuple[FrameScores, float, dict]:
    """The exported LiteRT model through the edge runner (what the Pi runs), its threshold and identity."""
    from e3_edge.fall_classifier import FallClassifierRunner

    runner = FallClassifierRunner(model_path)

    def score_batch(embeddings: np.ndarray) -> np.ndarray:
        rows = np.asarray(embeddings, dtype=np.float32)
        return np.array([runner.predict(row) for row in rows], dtype=np.float32)

    info = {"model": str(model_path), "model_sha256": runner.model_sha256,
            "yamnet_sha256": runner.yamnet_sha256, "threshold": runner.threshold}
    return FrameScores(score_batch), runner.threshold, info


def select_folds(stream: FrameStream, wav_dir: Path, suffix: str, folds: tuple[int, ...]) -> FrameStream:
    """The clips of a SAFE frame file whose WAV (sorted name order, ending in `suffix`) is in `folds`."""
    names = sorted(p.name for p in Path(wav_dir).rglob("*")
                   if p.is_file() and p.suffix.lower() == ".wav" and p.stem.endswith(suffix))
    parts = clips(stream)
    if len(names) != len(parts):
        raise ValueError(f"{wav_dir} has {len(names)} *{suffix}.wav files but the frame file has "
                         f"{len(parts)} clips; they must be the WAVs it was exported from")
    keep = [rows for name, rows in zip(names, parts, strict=True) if int(name[:2]) in folds]
    if not keep:
        raise ValueError(f"no clips in folds {folds}")
    rows = np.concatenate(keep)
    return FrameStream(stream.timestamps[rows], stream.embeddings[rows],
                       None if stream.scores is None else stream.scores[rows], stream.score_classes,
                       None if stream.level_dbfs is None else stream.level_dbfs[rows])
