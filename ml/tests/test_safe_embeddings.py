"""Check persisted recording labels and edge windowing without loading a real model."""

import csv
import hashlib
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from e3_edge.audio import pipeline as edge_pipeline

from src.fall_classifier import export_safe_embeddings as exporter


def make_splits(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    for name, _, _ in rows:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    path = root / "splits.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["path", "label", "split"])
        writer.writerows(rows)
    return path


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    model = tmp_path / "yamnet.tflite"
    model.write_bytes(b"fake model for export tests")
    state = SimpleNamespace(loads=0, prepared=[])

    class FakeRunner:
        def __init__(self, model_path, *, num_threads=1):
            state.loads += 1

        def extract_features(self, waveform):
            state.prepared.append(waveform.copy())
            embedding = np.full(1024, np.max(np.abs(waveform)), dtype=np.float32)
            return embedding, np.zeros(521, dtype=np.float32)

    monkeypatch.setattr(edge_pipeline, "YAMNetRunner", FakeRunner)
    monkeypatch.setattr(
        exporter,
        "_load_recording",
        lambda path: np.linspace(0.1, 0.4, 48000, dtype=np.float32),
    )
    return model, state


def test_saved_examples_preserve_recording_labels_splits_and_model_identity(tmp_path, fake_model):
    model, state = fake_model
    rows = [
        ("nested/01-001-01-001-01.wav", "fall", "train"),
        ("08-002-00-001-02.wav", "non_fall", "validation"),
        ("09-003-02-002-01.wav", "fall", "test"),
    ]
    splits = make_splits(tmp_path / "safe", rows)
    original_csv = splits.read_bytes()
    output = tmp_path / "processed"

    manifest_path = exporter.export_safe_embeddings(splits, model, output)

    assert state.loads == 1
    assert splits.read_bytes() == original_csv
    with manifest_path.open(newline="") as source:
        manifest = list(csv.DictReader(source))
    assert len(manifest) == 3
    assert {row["split"] for row in manifest} == {"train", "validation", "test"}
    expected = {path: (label, split) for path, label, split in rows}
    for row in manifest:
        assert (row["label"], row["split"]) == expected[row["path"]]
        with np.load(output / row["features_path"], allow_pickle=False) as saved:
            assert saved["embeddings"].shape == (6, 1024)
            assert saved["embeddings"].dtype == np.float32
            # One label for the whole recording, not six labels claiming six falls.
            assert saved["label"].shape == saved["split"].shape == ()
            assert saved["label"].item() == row["label"]
            assert saved["split"].item() == row["split"]
            assert saved["source_path"].item() == row["path"]
            np.testing.assert_allclose(saved["window_start_seconds"], np.arange(6) * 0.48)
            np.testing.assert_array_equal(saved["valid_samples"], [15600] * 5 + [9600])
            assert saved["model_sha256"].item() == hashlib.sha256(model.read_bytes()).hexdigest()
            assert saved["sample_rate"].item() == 16000
            assert saved["window_samples"].item() == 15600
            assert saved["hop_samples"].item() == 7680
        assert row["windows"] == "6"
    # Exercise real edge preprocessing: remove each window's mean before end padding.
    assert all(abs(window.mean()) < 1e-7 for window in state.prepared)
    np.testing.assert_array_equal(state.prepared[5][9600:], np.zeros(6000, dtype=np.float32))


@pytest.mark.parametrize(
    ("length", "count", "last_valid"),
    [
        (1, 1, 1),
        (15600, 1, 15600),
        (15601, 2, 7921),
        (23280, 2, 15600),
        (48000, 6, 9600),
    ],
)
def test_short_and_exact_window_boundaries(tmp_path, fake_model, monkeypatch, length, count, last_valid):
    model, state = fake_model
    splits = make_splits(tmp_path / "safe", [("01-001-01-001-01.wav", "fall", "train")])
    monkeypatch.setattr(exporter, "_load_recording", lambda path: np.zeros(length, dtype=np.float32))

    manifest = exporter.export_safe_embeddings(splits, model, tmp_path / "processed")

    with manifest.open(newline="") as source:
        row = next(csv.DictReader(source))
    with np.load(manifest.parent / row["features_path"], allow_pickle=False) as saved:
        assert saved["embeddings"].shape == (count, 1024)
        assert saved["valid_samples"][-1] == last_valid
    assert len(state.prepared) == count


@pytest.mark.parametrize(
    "rows",
    [
        [("01-001-01-001-01.wav", "non_fall", "train")],
        [("01-001-01-001-01.wav", "fall", "unknown")],
        [("01-001-01-001-01.wav", "fall", "train")] * 2,
        [("01-001-01-001-01.wav", "fall", "train"), ("08-002-01-001-01.wav", "fall", "test")],
        [("01-001-01-001-01.wav", "fall", "train"), ("01-002-01-002-01.wav", "fall", "test")],
        [("../01-001-01-001-01.wav", "fall", "train")],
        [],
    ],
)
def test_invalid_assignments_fail_before_loading_or_writing(tmp_path, fake_model, rows):
    model, state = fake_model
    splits = make_splits(tmp_path / "safe", rows)
    output = tmp_path / "processed"

    with pytest.raises(ValueError):
        exporter.export_safe_embeddings(splits, model, output)

    assert state.loads == 0
    assert not output.exists()


def test_processing_failure_preserves_previous_export(tmp_path, fake_model, monkeypatch):
    model, _ = fake_model
    splits = make_splits(
        tmp_path / "safe",
        [
            ("01-001-01-001-01.wav", "fall", "train"),
            ("08-002-00-002-02.wav", "non_fall", "validation"),
        ],
    )
    output = tmp_path / "processed"
    exporter.export_safe_embeddings(splits, model, output)
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}

    def failed_load(path):
        if path.name.startswith("08"):
            raise ValueError("corrupt WAV")
        return np.zeros(48000, dtype=np.float32)

    monkeypatch.setattr(exporter, "_load_recording", failed_load)
    with pytest.raises(ValueError, match="08-002-00-002-02.wav.*corrupt WAV"):
        exporter.export_safe_embeddings(splits, model, output)

    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before == after


def test_decoder_preserves_channels_and_dc_until_edge_preprocessing(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = np.tile([8192, 16384], (8000, 1)).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())

    loaded = exporter._load_recording(path)

    assert loaded.shape == (16000, 2) and loaded.dtype == np.float32
    np.testing.assert_allclose(loaded[1000:-1000, 0], 0.25, atol=1e-4)
    np.testing.assert_allclose(loaded[1000:-1000, 1], 0.5, atol=1e-4)
