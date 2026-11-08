"""A small window scorer, trained with recording-level max pooling.

The standalone window model maps (batch, 1024) embeddings to (batch, 1) fall
scores. Its built-in scaling uses TRAINING windows only and stays inside the
saved model, so future Pi inference will not need a separate scaler.

The training wrapper applies that same scorer to each recording's windows and
compares the highest valid score against ONE recording label. It does not
claim that every window in a fall recording contains an impact, nor that
window scores are calibrated probabilities or verified impact annotations.
"""

import numpy as np
import tensorflow as tf
from e3_edge.audio.yamnet import EMBEDDING_SIZE


def build_window_model(training_embeddings: np.ndarray) -> tf.keras.Model:
    """Define the scorer; initialize scaling from training data, not validation."""
    samples = np.asarray(training_embeddings)
    if samples.ndim != 2 or samples.shape[1] != EMBEDDING_SIZE or len(samples) == 0:
        raise ValueError("Expected nonempty training windows shaped (N, 1024)")
    if not np.isfinite(samples).all():
        raise ValueError("Training embeddings must be finite")
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    # A floor avoids magnifying tiny numerical differences in near-constant features.
    variance = np.maximum(samples.var(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    inputs = tf.keras.Input(shape=(EMBEDDING_SIZE,), dtype="float32", name="embedding")
    normalized = tf.keras.layers.Normalization(
        mean=mean.tolist(), variance=variance.tolist(), name="training_scaling",
    )(inputs)
    hidden = tf.keras.layers.Dense(
        64, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(1e-4), name="hidden",
    )(normalized)
    hidden = tf.keras.layers.Dropout(0.2, name="dropout")(hidden)
    score = tf.keras.layers.Dense(1, activation="sigmoid", name="fall_score")(hidden)
    return tf.keras.Model(inputs, score, name="window_classifier")


def build_recording_model(window_model: tf.keras.Model) -> tf.keras.Model:
    """Share the window model's weights and take the maximum valid window score."""
    embeddings = tf.keras.Input(shape=(None, EMBEDDING_SIZE), dtype="float32", name="embeddings")
    mask = tf.keras.Input(shape=(None, 1), dtype="float32", name="valid_windows")
    scores = tf.keras.layers.TimeDistributed(window_model, name="window_scores")(embeddings)
    # Scores are nonnegative, so zeroed batch-padding entries cannot raise the max.
    scores = tf.keras.layers.Multiply(name="exclude_batch_padding")([scores, mask])
    highest = tf.keras.layers.GlobalMaxPooling1D(name="recording_score")(scores)
    return tf.keras.Model(
        {"embeddings": embeddings, "valid_windows": mask}, highest, name="recording_classifier",
    )
