"""Prepare audio samples for YAMNet without microphone or file access (preprocessor)."""

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.96  # Duration of a YAMNet spectrogram patch.
HOP_SECONDS = 0.48  # Time between consecutive patch starts.


def prepare_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return DC-corrected mono float32 samples in [-1, 1].

    Input must already be normalized floating point audio at 16 kHz, shaped
    (samples,) or (samples, channels). Remove each channel's mean over this
    buffer before averaging channels and clipping samples outside [-1, 1].
    This removes a constant bias without normalizing the remaining sound's volume.

    The mean is calculated over the supplied buffer. A live window and a whole
    recording may have different means, even when using the same calculation.

    The caller (audio acquisition) handles audio capture and any sample-rate conversion. This function
    does not split audio into windows or run YAMNet.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
    audio = np.asarray(samples)
    if audio.size == 0 or audio.ndim not in (1, 2):
        raise ValueError("Audio must contain samples shaped (samples,) or (samples, channels)")
    if not np.issubdtype(audio.dtype, np.floating) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite, normalized floating point samples")
    # Match the laptop's DC removal. Float64 avoids accumulation error hiding
    # quiet sounds behind a residual offset; the copy preserves the caller's buffer.
    channels = audio.reshape(len(audio), -1).astype(np.float64)
    channels -= channels.mean(axis=0)
    return np.clip(channels.mean(axis=1), -1.0, 1.0).astype(np.float32)
