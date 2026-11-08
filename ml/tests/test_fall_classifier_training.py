"""Exercise max-score supervision and save/reload on tiny synthetic embeddings.

Requires the laptop's training dependencies; skipped in the lightweight CI job.
No real SAFE/test recordings, YAMNet models, downloads, or MLflow stores are used.
"""

import csv
import json
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from src.fall_classifier.dataset import Recording, TrainingData

tf = pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")
model = import_module("src.fall_classifier.model")
training = import_module("src.fall_classifier.train")


def test_max_scoring_ignores_padding_and_uses_shared_window_model():
    inputs = tf.keras.Input(shape=(1024,))
    layer = tf.keras.layers.Dense(1, activation="sigmoid")
    scorer = tf.keras.Model(inputs, layer(inputs))
    weights = np.zeros((1024, 1), dtype=np.float32)
    weights[0] = 1
    layer.set_weights([weights, np.zeros(1, dtype=np.float32)])
    recording = model.build_recording_model(scorer)
    embeddings = np.zeros((1, 3, 1024), dtype=np.float32)
    embeddings[0, :, 0] = [-1, 1, 100]
    mask = np.array([[[1], [1], [0]]], dtype=np.float32)

    actual = recording({"embeddings": embeddings, "valid_windows": mask}, training=False).numpy()
    expected = scorer(embeddings[0, :2], training=False).numpy().max()

    np.testing.assert_allclose(actual, [[expected]], atol=1e-6)
    assert actual[0, 0] < 0.99  # The masked slot's near-1 score must not win.
    # A different number of windows needs no different classifier or weights.
    single = recording({"embeddings": embeddings[:, :1], "valid_windows": mask[:, :1]}, training=False)
    np.testing.assert_allclose(single.numpy(), scorer(embeddings[0, :1]).numpy(), atol=1e-6)


def test_positive_recording_does_not_supervise_every_window_as_a_fall():
    inputs = tf.keras.Input(shape=(1024,))
    layer = tf.keras.layers.Dense(1, activation="sigmoid")
    scorer = tf.keras.Model(inputs, layer(inputs))
    weights = np.zeros((1024, 1), dtype=np.float32)
    weights[0] = 1
    layer.set_weights([weights, np.zeros(1, dtype=np.float32)])
    recording = model.build_recording_model(scorer)
    embeddings = np.zeros((1, 3, 1024), dtype=np.float32)
    embeddings[0, :, 0] = [-1, 1, 100]
    windows = tf.Variable(embeddings)
    mask = tf.constant([[[1.0], [1.0], [0.0]]])
    with tf.GradientTape() as tape:
        score = recording({"embeddings": windows, "valid_windows": mask}, training=False)
        loss = tf.keras.losses.binary_crossentropy(tf.constant([[1.0]]), score)
    gradient = tape.gradient(loss, windows).numpy()

    np.testing.assert_array_equal(gradient[0, 0], np.zeros(1024))
    np.testing.assert_array_equal(gradient[0, 2], np.zeros(1024))
    assert gradient[0, 1, 0] < 0


def test_window_model_includes_scaling_and_reloads_without_custom_layers(tmp_path):
    samples = np.stack([np.full(1024, 1), np.full(1024, 3)]).astype(np.float32)
    scorer = model.build_window_model(samples)
    scaling = scorer.get_layer("training_scaling")
    np.testing.assert_allclose(scaling(samples).numpy(), np.stack([-np.ones(1024), np.ones(1024)]))
    expected = scorer(samples, training=False).numpy()
    path = tmp_path / "window.keras"
    scorer.save(path)

    loaded = tf.keras.models.load_model(path, compile=False)

    actual = loaded(samples, training=False).numpy()
    assert actual.shape == (2, 1)
    assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_training_saves_best_scorer_history_and_provenance_without_test_access(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,split,features_path,windows\n")
    rng = np.random.default_rng(42)

    def example(index, label, split, count):
        embeddings = rng.normal(0, 0.1, (count, 1024)).astype(np.float32)
        if label == "fall":
            embeddings[0, :10] += 1
        return Recording(f"example-{index}.wav", label, split, embeddings)

    data = TrainingData(
        train=(example(1, "fall", "train", 6), example(2, "non_fall", "train", 5)),
        validation=(example(3, "fall", "validation", 6), example(4, "non_fall", "validation", 5)),
        provenance={"manifest": str(manifest), "manifest_sha256": "a" * 64, "yamnet_sha256": "b" * 64},
    )
    monkeypatch.setattr(training, "load_training_data", lambda path: data)
    tracked = []
    monkeypatch.setattr(training, "_log_run", lambda *args: tracked.append(args) or "synthetic-run")
    config = training.TrainingConfig(epochs=3, batch_size=2, patience=1)

    directory = training.train(manifest, tmp_path / "models", config=config)

    assert tracked and tracked[0][0] == directory
    summary = json.loads((directory / "run.json").read_text())
    with (directory / "history.csv").open(newline="") as source:
        history = list(csv.DictReader(source))
    losses = [float(row["val_loss"]) for row in history]
    assert summary["best_epoch"] == int(np.argmin(losses)) + 1
    assert summary["test_evaluated"] is False and summary["deployment_threshold"] is None
    assert len(summary["training_sources"]) == len(summary["validation_sources"]) == 2
    loaded = tf.keras.models.load_model(directory / "window_classifier.keras", compile=False)
    scaled_mean = loaded.get_layer("training_scaling").mean.numpy().reshape(-1)
    expected_mean = np.concatenate([recording.embeddings for recording in data.train]).mean(axis=0)
    np.testing.assert_allclose(scaled_mean, expected_mean, atol=1e-7)
    wrapper = model.build_recording_model(loaded)
    wrapper.compile(loss=tf.keras.losses.BinaryCrossentropy())
    restored_loss = wrapper.evaluate(training._batches(data.validation, config, shuffle=False), verbose=0)
    np.testing.assert_allclose(restored_loss, min(losses), atol=1e-5)
    np.testing.assert_allclose(restored_loss, summary["validation_metrics"]["loss"], atol=1e-5)


@pytest.mark.parametrize(
    "settings",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"patience": -1},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"seed": -1},
    ],
)
def test_invalid_training_settings_fail_before_loading_data(monkeypatch, settings):
    def unexpected_load(path):
        pytest.fail("Invalid settings should not load recordings")

    monkeypatch.setattr(training, "load_training_data", unexpected_load)
    with pytest.raises(ValueError):
        training.train(config=training.TrainingConfig(**settings))


def test_mlflow_logging_uses_the_local_store_and_records_history_and_artifacts(tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *, tracking_uri):
            assert tracking_uri == training.TRACKING_URI
            assert tracking_uri.startswith("sqlite:///")

        def get_experiment_by_name(self, name):
            return None

        def create_experiment(self, name, *, artifact_location):
            assert artifact_location.startswith("file://")
            return "experiment"

        def create_run(self, experiment_id, *, tags):
            return SimpleNamespace(info=SimpleNamespace(run_id="run"))

        def log_param(self, *args):
            calls.append(("param", args))

        def log_metric(self, *args, **kwargs):
            calls.append(("metric", args, kwargs))

        def log_artifact(self, run_id, path):
            calls.append(("artifact", path))

        def set_terminated(self, run_id, *, status):
            calls.append(("status", status))

    monkeypatch.setattr(training, "MlflowClient", FakeClient)
    summary = {
        "settings": {"seed": 42},
        "data": {"manifest_sha256": "a" * 64, "yamnet_sha256": "b" * 64},
        "best_epoch": 2,
        "validation_metrics": {"loss": 0.3},
    }

    run_id = training._log_run(tmp_path, summary, {"loss": [0.8, 0.4], "val_loss": [0.7, 0.3]})

    assert run_id == "run" and calls[-1] == ("status", "FINISHED")
    artifacts = [call[1] for call in calls if call[0] == "artifact"]
    assert len(artifacts) == 4
    assert str(tmp_path / "window_classifier.keras") in artifacts
    assert ("metric", ("run", "val_loss", 0.3), {"step": 2}) in calls
