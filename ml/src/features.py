"""Shared audio preparation. Keep in sync with edge inference."""

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.96
HOP_SECONDS = 0.48


def prepare_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare normalized float samples shaped (samples,) or (samples, channels).

    Input must already be 16 kHz. The development exporter handles WAV decoding
    and resampling. The output is mono float32 in [-1, 1]; YAMNet handles framing.
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
