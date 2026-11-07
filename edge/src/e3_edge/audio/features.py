"""Prepare audio samples for YAMNet without microphone or file access (preprocessor)."""

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.96  # Duration of a YAMNet spectrogram patch.
HOP_SECONDS = 0.48  # Time between consecutive patch starts.


def prepare_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return mono float32 samples in [-1, 1], preserving the recording's level.

    Input must already be normalized floating point audio at 16 kHz, shaped
    (samples,) or (samples, channels). Average channels to mono and clip samples
    outside [-1, 1]; do not adjust the recording's overall volume.

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
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
