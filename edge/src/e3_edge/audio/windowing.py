"""Assemble captured audio into complete, overlapping AudioPipeline inputs."""

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from .capture import CapturedBlock
from .features import HOP_SECONDS, SAMPLE_RATE
from .yamnet import WINDOW_SAMPLES

HOP_SAMPLES = round(HOP_SECONDS * SAMPLE_RATE)
OVERLAP_SAMPLES = WINDOW_SAMPLES - HOP_SAMPLES
# Match capture's clock-discontinuity tolerance while allowing driver jitter.
TIMESTAMP_TOLERANCE_SECONDS = 0.01


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """Owned mono float32 audio and its first sample's UTC Unix timestamp.

    sample_index uses the source CapturedBlock's sample counter. discontinuity
    marks the first full window after initialization, reset, or a recording gap;
    downstream temporal detection should interrupt continuity before this window.
    samples is excluded from repr to avoid logging raw audio.
    """

    samples: np.ndarray = field(repr=False)
    start_time: float
    sample_index: int
    discontinuity: bool


class WindowAssembler:
    """Build 15,600-sample windows starting every 7,680 samples (0.48 s).

    Use one instance per microphone, on the consumer thread outside the capture
    callback. push() copies incoming samples into a fixed 62,400-byte buffer and
    returns any complete windows; it retains no output queue or input blocks.
    Returned windows own their arrays and must be processed or released by the
    caller. With capture's default 1,024-sample blocks, each push returns at most
    one window; larger inputs can return several.

    Within a continuous segment, timestamps derive from its first block and
    integer sample offsets, not repeated floating-point addition. Discontinuity
    flags, unexpected sample indices, or ADC/UTC drift over 10 ms start a new
    segment and discard old overlap. Timestamp ordering is only guaranteed within
    a segment: a clock correction on capture restart can move UTC backwards.

    There is no padding or partial-window flush. Call reset() on mute, shutdown,
    or capture failure to clear retained audio. No audio is saved or transmitted.
    """

    def __init__(self) -> None:
        self._samples = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self.reset()

    @property
    def pending_samples(self) -> int:
        """Retained samples, including overlap used by a previously emitted window."""
        return self._filled

    def reset(self) -> None:
        """Discard and clear partial audio; the next segment gets a fresh anchor."""
        self._samples.fill(0)
        self._filled = self._window_offset = 0
        self._expected_index: int | None = None
        self._segment_index = 0
        self._segment_start = self._segment_adc = 0.0
        self._discontinuity = True

    def push(self, block: CapturedBlock) -> list[AudioWindow]:
        """Consume one valid block and return complete windows in sample order.

        Invalid input raises before modifying retained audio. A valid block at a
        continuity boundary becomes the start of a new segment, even when it
        repeats or precedes a previous block's index. Samples are not normalized
        or DC-corrected here; that remains AudioPipeline's responsibility.
        """
        samples = self._validate(block)
        if self._starts_segment(block):
            self.reset()
            self._segment_index = int(block.sample_index)
            self._segment_start = float(block.start_time)
            self._segment_adc = float(block.adc_time)
        self._expected_index = int(block.sample_index) + len(samples)

        windows = []
        offset = 0
        while offset < len(samples):
            count = min(WINDOW_SAMPLES - self._filled, len(samples) - offset)
            self._samples[self._filled:self._filled + count] = samples[offset:offset + count]
            self._filled += count
            offset += count
            if self._filled == WINDOW_SAMPLES:
                windows.append(AudioWindow(
                    samples=self._samples.copy(),
                    start_time=self._segment_start + self._window_offset / SAMPLE_RATE,
                    sample_index=self._segment_index + self._window_offset,
                    discontinuity=self._discontinuity,
                ))
                self._discontinuity = False
                self._window_offset += HOP_SAMPLES
                self._samples[:OVERLAP_SAMPLES] = self._samples[HOP_SAMPLES:]
                self._samples[OVERLAP_SAMPLES:].fill(0)
                self._filled = OVERLAP_SAMPLES
        return windows

    def _starts_segment(self, block: CapturedBlock) -> bool:
        if block.discontinuity or block.sample_index != self._expected_index:
            return True
        seconds = (block.sample_index - self._segment_index) / SAMPLE_RATE
        return (
            abs(block.start_time - (self._segment_start + seconds)) > TIMESTAMP_TOLERANCE_SECONDS
            or abs(block.adc_time - (self._segment_adc + seconds)) > TIMESTAMP_TOLERANCE_SECONDS
        )

    @staticmethod
    def _validate(block: CapturedBlock) -> np.ndarray:
        samples = np.asarray(block.samples)
        if samples.ndim != 1 or samples.size == 0 or samples.dtype != np.float32:
            raise ValueError("Captured samples must be a nonempty mono float32 array")
        if not np.isfinite(samples).all():
            raise ValueError("Captured samples must be finite")
        if (
            isinstance(block.sample_index, bool)
            or not isinstance(block.sample_index, Integral)
            or block.sample_index < 0
        ):
            raise ValueError("sample_index must be a nonnegative integer")
        for name in ("start_time", "adc_time"):
            value = getattr(block, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite seconds")
        if not isinstance(block.discontinuity, (bool, np.bool_)):
            raise ValueError("discontinuity must be a boolean")
        return samples
