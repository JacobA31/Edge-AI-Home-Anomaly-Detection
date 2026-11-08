"""Validate recording-level examples and split isolation without TensorFlow."""

import csv
import hashlib

import numpy as np
import pytest

from src.fall_classifier import dataset


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    examples = [
        ("01-001-01-001-01.wav", "fall", "train", 6),
        ("02-002-00-001-02.wav", "non_fall", "train", 5),
        ("08-003-01-002-01.wav", "fall", "validation", 6),
        ("08-004-00-002-02.wav", "non_fall", "validation", 5),
        ("09-005-01-003-01.wav", "fall", "test", 6),
        ("10-006-00-003-02.wav", "non_fall", "test", 6),
    ]
    rows = []
    for name, label, split, count in examples:
        features = name.replace(".wav", ".npz")
        rows.append((name, label, split, features, count))
        # Deliberately do NOT create test archives: training must never need them.
        if split == "test":
            continue
        np.savez_compressed(
            root / features,
            embeddings=np.full((count, 1024), count, dtype=np.float32),
            window_start_seconds=np.arange(count) * 0.48,
            valid_samples=np.full(count, 15600),
            source_path=name,
            label=label,
            split=split,
            sample_rate=16000,
            window_samples=15600,
            hop_samples=7680,
            model_sha256="a" * 64,
        )
    path = root / "manifest.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["path", "label", "split", "features_path", "windows"])
        writer.writerows(rows)
    return path


def rewrite_archive(path, **changes):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_loads_only_train_and_validation_recordings(exported):
    loaded = dataset.load_training_data(exported)

    assert len(loaded.train) == len(loaded.validation) == 2
    assert {recording.label for recording in loaded.train} == {"fall", "non_fall"}
    assert all(recording.split == "train" for recording in loaded.train)
    assert all(recording.split == "validation" for recording in loaded.validation)
    assert loaded.provenance["counts"]["test"] == {"non_fall": 1, "fall": 1}
    assert loaded.provenance["yamnet_sha256"] == "a" * 64


def test_packing_keeps_one_label_per_recording_and_masks_only_batch_padding(exported):
    short = exported.parent / "02-002-00-001-02.npz"
    # A real final window made from incomplete audio must remain a valid embedding.
    rewrite_archive(short, valid_samples=np.array([15600, 15600, 15600, 15600, 10000]))
    loaded = dataset.load_training_data(exported)

    inputs, labels = dataset.pack_recordings(loaded.train)

    assert inputs["embeddings"].shape == (2, 6, 1024)
    assert labels.shape == (2, 1)
    np.testing.assert_array_equal(labels[:, 0], [1, 0])
    np.testing.assert_array_equal(inputs["valid_windows"][0, :, 0], np.ones(6))
    np.testing.assert_array_equal(inputs["valid_windows"][1, :, 0], [1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(inputs["embeddings"][1, 5], np.zeros(1024))


@pytest.mark.parametrize(
    "changes",
    [
        {"label": "non_fall"},
        {"split": "test"},
        {"source_path": "other.wav"},
        {"embeddings": np.zeros((5, 1024), dtype=np.float32)},
        {"embeddings": np.zeros((6, 1024), dtype=np.float64)},
        {"embeddings": np.full((6, 1024), np.nan, dtype=np.float32)},
        {"sample_rate": 48000},
        {"hop_samples": 15600},
        {"window_start_seconds": np.arange(6)},
        {"valid_samples": np.array([15600, 100, 15600, 15600, 15600, 15600])},
        {"valid_samples": np.array([15600, 15600, 15600, 15600, 15600, 1])},
        {"model_sha256": "b" * 64},
    ],
)
def test_rejects_mismatched_corrupt_or_mixed_feature_exports(exported, changes):
    rewrite_archive(exported.parent / "01-001-01-001-01.npz", **changes)

    with pytest.raises(ValueError):
        dataset.load_training_data(exported)


def test_duplicate_recordings_cannot_cross_splits(exported):
    with exported.open(newline="") as source:
        rows = list(csv.reader(source))
    duplicate = rows[1].copy()
    duplicate[2] = "test"
    rows.append(duplicate)
    with exported.open("w", newline="") as output:
        csv.writer(output).writerows(rows)

    with pytest.raises(ValueError, match="Duplicate"):
        dataset.load_training_data(exported)


def test_rejects_feature_paths_outside_processed_folder(exported):
    text = exported.read_text().replace("01-001-01-001-01.npz", "../outside.npz")
    exported.write_text(text)

    with pytest.raises(ValueError, match="relative path"):
        dataset.load_training_data(exported)


def test_requires_both_labels_in_training_and_validation(exported):
    with exported.open(newline="") as source:
        rows = list(csv.reader(source))
    rows = [row for row in rows if row[0] != "08-004-00-002-02.wav"]
    with exported.open("w", newline="") as output:
        csv.writer(output).writerows(rows)

    with pytest.raises(ValueError, match="validation must contain both"):
        dataset.load_training_data(exported)


def evaluation_settings(manifest):
    return {
        "expected_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expected_yamnet_sha256": "a" * 64,
    }


def test_validation_evaluation_does_not_need_training_or_test_archives(exported):
    (exported.parent / "01-001-01-001-01.npz").unlink()
    (exported.parent / "02-002-00-001-02.npz").unlink()

    recordings = dataset.load_evaluation_split(exported, "validation", **evaluation_settings(exported))

    assert len(recordings) == 2
    assert all(recording.split == "validation" for recording in recordings)


def test_test_evaluation_loads_only_explicit_test_archives(exported, monkeypatch):
    with np.load(exported.parent / "01-001-01-001-01.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    for name, label in (("09-005-01-003-01", "fall"), ("10-006-00-003-02", "non_fall")):
        np.savez_compressed(
            exported.parent / f"{name}.npz",
            **{**arrays, "source_path": f"{name}.wav", "label": label, "split": "test"},
        )
    loaded_splits = []
    original = dataset._load_recording

    def tracked_load(root, row):
        loaded_splits.append(row["split"])
        return original(root, row)

    monkeypatch.setattr(dataset, "_load_recording", tracked_load)

    recordings = dataset.load_evaluation_split(exported, "test", **evaluation_settings(exported))

    assert loaded_splits == ["test", "test"]
    assert len(recordings) == 2 and all(recording.split == "test" for recording in recordings)


def test_evaluation_rejects_changed_manifest_before_opening_any_archives(exported, monkeypatch):
    settings = evaluation_settings(exported)
    exported.write_text(exported.read_text() + "\n")

    def unexpected_load(*args):
        pytest.fail("Changed manifest must fail before loading embeddings")

    monkeypatch.setattr(dataset, "_load_recording", unexpected_load)
    with pytest.raises(ValueError, match="Manifest differs"):
        dataset.load_evaluation_split(exported, "test", **settings)


def test_evaluation_checks_yamnet_fingerprint_against_training_run(exported):
    settings = {**evaluation_settings(exported), "expected_yamnet_sha256": "b" * 64}

    with pytest.raises(ValueError, match="YAMNet fingerprint differs"):
        dataset.load_evaluation_split(exported, "validation", **settings)


def test_evaluation_rejects_training_split(exported):
    with pytest.raises(ValueError, match="validation or test"):
        dataset.load_evaluation_split(exported, "train", **evaluation_settings(exported))
