"""Laptop YAMNet inference, returning embeddings and all AudioSet class scores."""

import os
import ssl
from functools import lru_cache
from importlib import import_module

import certifi
import numpy as np

from .features import FRAME_SECONDS, HOP_SECONDS, SAMPLE_RATE

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_SIZE = 1024
CLASS_COUNT = 521
HOP_SAMPLES = round(HOP_SECONDS * SAMPLE_RATE)
# A 0.96 s log-mel patch needs another 15 ms to finish its final STFT window.
# Matches tensorflow/models/research/audioset/yamnet/features.py:pad_waveform.
WINDOW_SAMPLES = round((FRAME_SECONDS + 0.025 - 0.010) * SAMPLE_RATE)


def window_count(sample_count: int) -> int:
    """Count the windows produced by Hub YAMNet, including its padded final window."""
    if sample_count <= 0:
        raise ValueError("Audio is empty")
    remaining = max(0, sample_count - WINDOW_SAMPLES)
    return 1 + (remaining + HOP_SAMPLES - 1) // HOP_SAMPLES


@lru_cache(maxsize=1)
def _load_yamnet_model():
    # Hub builds its HTTPS context on import. Supply trusted roots when the
    # Python installation has none, while respecting explicit certificate settings.
    trust = ssl.get_default_verify_paths()
    if not trust.cafile and not trust.capath and "SSL_CERT_DIR" not in os.environ:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    hub = import_module("tensorflow_hub")
    return hub.load(MODEL_URL)


def extract_features(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings[N, 1024], scores[N, 521]) for prepared 16 kHz audio."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("YAMNet needs a nonempty mono waveform")
    if not np.isfinite(audio).all() or np.any(np.abs(audio) > 1.0):
        raise ValueError("YAMNet needs finite audio samples in [-1, 1]")
    scores, embeddings, _spectrogram = _load_yamnet_model()(audio)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    count = window_count(len(audio))
    if embeddings.shape != (count, EMBEDDING_SIZE) or scores.shape != (count, CLASS_COUNT):
        raise ValueError(f"Unexpected YAMNet output shapes: {embeddings.shape}, {scores.shape}")
    if not np.isfinite(embeddings).all() or not np.isfinite(scores).all():
        raise ValueError("YAMNet returned non-finite output")
    return embeddings, scores
