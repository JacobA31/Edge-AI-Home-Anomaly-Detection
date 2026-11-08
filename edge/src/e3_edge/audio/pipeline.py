"""Turn one supplied audio window into the frame consumed by the detectors."""

from numbers import Real
from pathlib import Path

import numpy as np

from ..anomaly.frames import Frame, frame_level_dbfs
from .features import SAMPLE_RATE, prepare_audio
from .yamnet import WINDOW_SAMPLES, YAMNetRunner


class AudioPipeline:
    """Keep YAMNet loaded and combine audio preparation, inference, and levels.

    The caller supplies normalized floating point audio at 16 kHz, shaped
    (samples,) or (samples, channels), and the window's UTC Unix start time.
    Complete windows contain 15,600 samples per channel and normally start
    0.48 seconds apart. Capture, buffering, window timing, and microphone IDs
    belong to the caller; this component does not read files or microphones.

    Reuse an instance sequentially. It does not enforce timestamp ordering, so
    the caller can keep separate microphone streams while sharing a runner.
    """

    def __init__(self, model_path: str | Path, *, num_threads: int = 1) -> None:
        self._yamnet = YAMNetRunner(model_path, num_threads=num_threads)

    def process_window(
        self, samples: np.ndarray, *, start_time: float,
        sample_rate: int = SAMPLE_RATE, pad_end: bool = False,
    ) -> Frame:
        """Return the embedding, scores, start time, and level for one window.

        Set pad_end=True only for an incomplete final window at the end of a
        recording/stream. Silence then fills the remaining model input, but the
        level is measured from observed samples only. During live capture, wait
        for a complete window instead of padding each incoming audio chunk.
        Longer inputs are rejected rather than silently discarding samples.
        """
        if (
            isinstance(start_time, bool)
            or not isinstance(start_time, Real)
            or not np.isfinite(float(start_time))
        ):
            raise ValueError("start_time must be finite UTC Unix seconds")

        waveform = prepare_audio(samples, sample_rate)
        length = len(waveform)
        if length > WINDOW_SAMPLES or (length < WINDOW_SAMPLES and not pad_end):
            raise ValueError(
                f"Expected {WINDOW_SAMPLES} samples per window; got {length}. "
                "Use pad_end=True only for an incomplete final window.",
            )

        # Use the same prepared samples as YAMNet, excluding any artificial
        # end padding so a short final window does not look falsely quiet.
        level_dbfs = frame_level_dbfs(waveform)
        if length < WINDOW_SAMPLES:
            waveform = np.pad(waveform, (0, WINDOW_SAMPLES - length))
        embedding, scores = self._yamnet.extract_features(waveform)
        return Frame(
            t=float(start_time), embedding=embedding, scores=scores,
            level_dbfs=level_dbfs,
        )
