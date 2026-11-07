"""YAMNet inference helpers for prepared audio waveforms. For """

from functools import lru_cache
from importlib import import_module

import numpy as np

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_SIZE = 1_024


@lru_cache(maxsize=1)
def _load_yamnet_model():
    """Load YAMNet once and reuse it for later recordings in this process."""
    hub = import_module("tensorflow_hub")
    return hub.load(YAMNET_MODEL_URL)


def extract_embeddings(waveform: np.ndarray) -> np.ndarray:
    """Return YAMNet embeddings for one prepared audio waveform.

    ``waveform`` should be the mono, 16 kHz float32 array returned by
    ``features.prepare_audio``. YAMNet handles its own spectrogram generation and
    overlapping audio windows. The result has one 1,024-value row per window.
    """
    audio = np.asarray(waveform, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError(f"Expected a one-dimensional mono waveform, got {audio.shape}")
    if audio.size == 0:
        raise ValueError("Cannot extract embeddings from empty audio")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio waveform contains non-finite values")
    if np.any(audio < -1.0) or np.any(audio > 1.0):
        raise ValueError("Audio waveform values must be in the range [-1.0, 1.0]")

    _scores, embeddings, _spectrogram = _load_yamnet_model()(audio)
    result = np.asarray(embeddings.numpy(), dtype=np.float32)

    if result.ndim != 2 or result.shape[1] != EMBEDDING_SIZE:
        raise RuntimeError(
            f"Expected YAMNet embeddings with {EMBEDDING_SIZE} values per row, "
            f"got shape {result.shape}"
        )

    return result
