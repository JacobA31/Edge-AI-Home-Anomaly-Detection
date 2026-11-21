"""Bounded, callback-based microphone acquisition; raw samples stay in RAM.

This module supplies blocks, not model windows. The consumer must assemble
15,600-sample windows with a 7,680-sample hop before calling AudioPipeline.
"""

from dataclasses import dataclass, field
from importlib import import_module
from math import isfinite
from threading import Lock
from time import time

import numpy as np

from .features import SAMPLE_RATE


class CaptureError(RuntimeError):
    """Capture failed; stop and restart the stream before consuming more audio."""


@dataclass(frozen=True, slots=True)
class CapturedBlock:
    """Owned samples and the capture time of their first sample.

    start_time is UTC Unix seconds; adc_time is in the PortAudio stream clock.
    sample_index counts samples supplied by callbacks since start(), including
    locally dropped blocks. Driver-side losses have unknown length. A true
    discontinuity means the consumer must discard any partial model window.
    samples is excluded from repr so logging this object does not expose audio.
    """

    samples: np.ndarray = field(repr=False)
    start_time: float
    adc_time: float
    sample_index: int
    discontinuity: bool


class MicrophoneCapture:
    """Acquire 16 kHz mono float32 blocks for one consumer.

    start(), read(), and stop() belong to one application thread. read() polls
    without waiting for audio. The callback never waits for the consumer: on
    lock contention it drops the incoming block; on capacity exhaustion it
    replaces the oldest queued block. Both losses are visible in sample_index
    and discontinuity. dropped_samples excludes unknown driver-side losses;
    input_overflows counts driver overflow reports separately.

    The default ring holds at most 32 * 1,024 samples (128 KiB, 2.048 seconds).
    A fixed callback block size bounds copying and preallocated slot size; it
    is independent of the model's window and hop. Python callbacks provide
    best-effort real-time behavior, not a hard real-time guarantee.

    UTC is anchored to stream.time before capture starts. Later timestamps use
    inputBufferAdcTime, so callback delays and wall-clock steps do not change
    them. Absolute accuracy depends on the host clock and driver. An ADC jump
    larger than 10 ms breaks continuity; a backwards clock fails capture.
    Restart to establish a fresh UTC anchor after a system clock correction.

    No files or network connections are opened. stop() aborts capture and
    clears internal sample buffers; arrays already returned belong to the caller.
    """

    def __init__(
        self, device: int | str | None = None, *, block_size: int = 1024, buffer_blocks: int = 32,
    ) -> None:
        for name, value in (("block_size", block_size), ("buffer_blocks", buffer_blocks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if device is not None and (
            isinstance(device, bool)
            or not isinstance(device, (int, str))
            or (isinstance(device, int) and device < 0)
            or (isinstance(device, str) and not device.strip())
        ):
            raise ValueError("device must be an input device index, name, or None")
        self.device = device
        self.block_size = block_size
        self.buffer_blocks = buffer_blocks
        self._samples = np.zeros((buffer_blocks, block_size), dtype=np.float32)
        self._lengths = [0] * buffer_blocks
        self._adc_times = [0.0] * buffer_blocks
        self._indices = [0] * buffer_blocks
        self._gaps = [False] * buffer_blocks
        self._lock = Lock()
        self._head = self._count = self._next_index = 0
        self._last_end: int | None = None
        self._previous_adc: float | None = None
        self._expected_adc: float | None = None
        self._utc_offset = 0.0
        self._stream = None
        self._accepting = False
        self._error: Exception | None = None
        self._callback_abort = Exception
        self._dropped_samples = self._input_overflows = 0

    @property
    def dropped_samples(self) -> int:
        return self._dropped_samples

    @property
    def input_overflows(self) -> int:
        return self._input_overflows

    def start(self) -> "MicrophoneCapture":
        """Open the selected device; unsupported formats fail without resampling."""
        if self._stream is not None:
            raise RuntimeError("Capture is already started; stop it before restarting")
        try:
            sd = import_module("sounddevice")
        except ModuleNotFoundError as exc:
            if exc.name != "sounddevice":
                raise
            raise ImportError('Microphone capture requires: pip install -e ".[pi]"') from exc
        sd.check_input_settings(device=self.device, channels=1, dtype="float32", samplerate=SAMPLE_RATE)
        self._clear()
        self._next_index = self._dropped_samples = self._input_overflows = 0
        self._previous_adc = self._expected_adc = None
        self._error = None
        self._callback_abort = sd.CallbackAbort
        stream = sd.InputStream(
            device=self.device, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=self.block_size, callback=self._callback,
        )
        self._stream = stream
        try:
            # PortAudio time has an arbitrary origin. Bracket its measurement
            # to reduce the uncertainty of mapping it into UTC, outside callback.
            before = time()
            stream_time = float(stream.time)
            after = time()
            if not all(isfinite(value) for value in (before, stream_time, after)):
                raise CaptureError("Cannot establish a finite stream-to-UTC clock mapping")
            if after < before or after - before > 0.01:
                raise CaptureError("Clock mapping took over 10 ms or the wall clock moved backwards")
            self._utc_offset = (before + after) / 2 - stream_time
            self._accepting = True
            stream.start()
        except BaseException:
            self.stop()
            raise
        return self

    def read(self) -> CapturedBlock | None:
        """Return the next owned block, or None when no audio is ready.

        Stream/callback failures raise instead of silently appearing as silence.
        The first block after every start is a new continuity boundary.
        """
        stream = self._stream
        if stream is None:
            raise RuntimeError("Capture is not started")
        if self._error is not None:
            raise CaptureError("Microphone callback failed") from self._error
        if not stream.active:
            raise CaptureError("Microphone stream became inactive")
        with self._lock:
            if not self._count:
                return None
            slot = self._head
            length = self._lengths[slot]
            index = self._indices[slot]
            block = CapturedBlock(
                samples=self._samples[slot, :length].copy(),
                start_time=self._adc_times[slot] + self._utc_offset,
                adc_time=self._adc_times[slot], sample_index=index,
                discontinuity=self._gaps[slot] or index != self._last_end,
            )
            self._samples[slot].fill(0)
            self._head = (slot + 1) % self.buffer_blocks
            self._count -= 1
            self._last_end = index + length
            return block

    def stop(self) -> None:
        """Abort acquisition, close the device, and discard queued audio; idempotent."""
        self._accepting = False
        stream = self._stream
        try:
            if stream is not None:
                try:
                    stream.abort()
                finally:
                    stream.close()
        finally:
            self._stream = None
            self._clear()

    def _clear(self) -> None:
        with self._lock:
            self._samples.fill(0)
            self._head = self._count = 0
            self._last_end = None

    def _callback(self, indata, frames, timing, status) -> None:
        if not self._accepting:
            return
        try:
            if not 0 < frames <= self.block_size:
                raise ValueError("Unexpected microphone callback block size")
            if indata.shape != (frames, 1) or indata.dtype != np.float32:
                raise ValueError("Microphone must supply mono float32 samples")
            adc = float(timing.inputBufferAdcTime)
            if not isfinite(adc):
                raise ValueError("Microphone ADC timestamp must be finite")
            if self._previous_adc is not None and adc <= self._previous_adc:
                raise ValueError("Microphone ADC clock moved backwards or repeated")
            gap = bool(status) or (
                self._expected_adc is not None and abs(adc - self._expected_adc) > 0.01
            )
            self._previous_adc = adc
            self._expected_adc = adc + frames / SAMPLE_RATE
            index = self._next_index
            self._next_index += frames
            if status.input_overflow:
                self._input_overflows += 1
            if not self._lock.acquire(blocking=False):
                self._dropped_samples += frames
                return
            try:
                if self._count == self.buffer_blocks:
                    self._dropped_samples += self._lengths[self._head]
                    self._head = (self._head + 1) % self.buffer_blocks
                    self._count -= 1
                slot = (self._head + self._count) % self.buffer_blocks
                self._samples[slot, :frames] = indata[:, 0]
                self._lengths[slot] = frames
                self._adc_times[slot] = adc
                self._indices[slot] = index
                self._gaps[slot] = gap
                self._count += 1
            finally:
                self._lock.release()
        except Exception as exc:
            # sounddevice does not propagate callback exceptions to the caller.
            # Keep the cause for read(), then use PortAudio's quiet abort path.
            self._error = exc
            self._accepting = False
            raise self._callback_abort from exc

    def __enter__(self) -> "MicrophoneCapture":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
