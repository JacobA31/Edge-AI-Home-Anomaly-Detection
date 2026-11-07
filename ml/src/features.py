"""Shared audio preparation. Keep in sync with edge inference."""

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.96
HOP_SECONDS = 0.48


def prepare_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare normalized float samples shaped (samples,) or (samples, channels).

    Input must already be 16 kHz. The development exporter handles WAV decoding
    and resampling. The output is mono float32 in [-1, 1]; YAMNet handles framing.

    Each channel's DC offset (its mean over this buffer) is removed before the channels
    are averaged. SINS recordings sit at about half full scale with the room's sound a
    tiny ripple on top; left in, YAMNet hears "Hum" in a silent room and every frame's
    level reads about -6 dBFS.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
    audio = np.asarray(samples)
    if audio.size == 0 or audio.ndim not in (1, 2):
        raise ValueError("Audio must contain samples shaped (samples,) or (samples, channels)")
    if not np.issubdtype(audio.dtype, np.floating) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite, normalized floating point samples")
    # float64 throughout: a float32 mean over a minute of audio leaves tens of counts of
    # offset behind, as loud as a quiet room. astype also keeps the caller's buffer intact.
    channels = audio.reshape(len(audio), -1).astype(np.float64)
    channels -= channels.mean(axis=0)
    return np.clip(channels.mean(axis=1), -1.0, 1.0).astype(np.float32)
