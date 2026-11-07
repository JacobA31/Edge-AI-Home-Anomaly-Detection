"""Shared feature extraction. Keep in sync with what edge/ computes at inference time."""

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.96


def prepare_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare 16 kHz samples as a continuous mono float32 waveform for YAMNet.

    Pass floating point samples on the [-1.0, 1.0] scale, shaped as
    (samples,) for mono or (samples, channels) for multichannel audio.
    Values outside that range are clipped.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")

    audio = np.asarray(samples)
    if audio.size == 0:
        raise ValueError("Audio samples are empty")
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError("Audio samples must be floating point on the [-1.0, 1.0] scale")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError(
            f"Expected mono or multichannel audio, got shape {audio.shape}"
        )

    audio = np.asarray(audio, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0)
