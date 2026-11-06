"""Shared audio preparation for YAMNet and edge inference."""

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.96


def load_audio(path: str | Path) -> np.ndarray:
    """Load a recording as mono, 16 kHz, float32 audio for YAMNet.

    Prepares a WAV file for YAMNet model. Combines multiple audio channels into one,
    changes the sample rate to 16,000 samples per second, and puts the sound values in
    the range YAMNet expects. It returns the prepared audio.

    The returned samples stay in the range [-1.0, 1.0]. The recording's
    duration is preserved while resampling. YAMNet itself creates its
    overlapping 0.96-second frames, so this function returns one continuous
    waveform rather than splitting it into frames.
    """
    audio, original_sample_rate = librosa.load(
        path,
        sr=None,
        mono=False,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if audio.ndim == 2:
        # librosa returns multichannel audio as (channels, samples).
        audio = np.mean(audio, axis=0, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError(
            f"Expected mono or multichannel audio, got shape {audio.shape}"
        )

    if original_sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=original_sample_rate,
            target_sr=SAMPLE_RATE,
        )

    audio = np.asarray(audio, dtype=np.float32)
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio
