"""Load recording-level training/validation examples; never open test archives."""

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from e3_edge.audio.features import HOP_SECONDS, SAMPLE_RATE
from e3_edge.audio.yamnet import EMBEDDING_SIZE, WINDOW_SAMPLES

from .index_safe import LABELS
from .split_safe import FILENAME

LABEL_IDS = {"non_fall": 0, "fall": 1}
HOP_SAMPLES = round(HOP_SECONDS * SAMPLE_RATE)


@dataclass(frozen=True)
class Recording:
    source_path: str
    label: str
    split: str
    embeddings: np.ndarray


@dataclass(frozen=True)
class TrainingData:
    train: tuple[Recording, ...]
    validation: tuple[Recording, ...]
    provenance: dict


def _relative_path(value: str, root: Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not (root / relative).resolve().is_relative_to(root):
        raise ValueError(f"Expected a relative path inside the dataset folder: {value}")
    return relative


def _read_manifest(manifest: Path) -> list[dict]:
    rows = []
    sources, features = set(), set()
    folds, clips = {}, {}
    with manifest.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, strict=True)
        columns = {"path", "label", "split", "features_path", "windows"}
        if not reader.fieldnames or len(reader.fieldnames) != len(columns) or set(reader.fieldnames) != columns:
            raise ValueError("Expected the manifest.csv produced by export_safe_embeddings")
        for line, raw in enumerate(reader, start=2):
            if None in raw or any(not (raw.get(key) or "").strip() for key in columns):
                raise ValueError(f"Invalid manifest row {line}")
            row = {key: raw[key].strip() for key in columns}
            path = _relative_path(row["path"], manifest.parent)
            feature = _relative_path(row["features_path"], manifest.parent)
            metadata = FILENAME.fullmatch(path.stem)
            if path.suffix.lower() != ".wav" or metadata is None or feature.suffix != ".npz":
                raise ValueError(f"Invalid SAFE recording/feature path on row {line}")
            if (
                row["label"] != LABELS[metadata["class"]]
                or row["split"] not in {"train", "validation", "test"}
            ):
                raise ValueError(f"Invalid label or split on row {line}")
            feature_key = (manifest.parent / feature).resolve()
            source_key = path.as_posix().casefold()
            if source_key in sources or feature_key in features:
                raise ValueError(f"Duplicate source or feature archive: {path}")
            sources.add(source_key)
            features.add(feature_key)
            group = (metadata["class"], metadata["environment"], metadata["source"])
            if folds.setdefault(metadata["fold"], row["split"]) != row["split"]:
                raise ValueError(f"SAFE fold crosses splits: {path}")
            if clips.setdefault(group, row["split"]) != row["split"]:
                raise ValueError(f"Source clip crosses splits: {path}")
            row["windows"] = int(row["windows"])
            if row["windows"] < 1:
                raise ValueError(f"Recording has no windows: {path}")
            rows.append(row)
    if not rows:
        raise ValueError("The manifest is empty")
    return sorted(rows, key=lambda row: row["path"])


def _load_recording(root: Path, row: dict) -> tuple[Recording, str]:
    """Validate persisted labels, shape, timing, and edge feature settings."""
    path = root / row["features_path"]
    try:
        with np.load(path, allow_pickle=False) as saved:
            for key, expected in (
                ("source_path", row["path"]), ("label", row["label"]), ("split", row["split"]),
            ):
                if saved[key].shape != () or saved[key].item() != expected:
                    raise ValueError(f"Archive {key} disagrees with manifest")
            for key, expected in (
                ("sample_rate", SAMPLE_RATE), ("window_samples", WINDOW_SAMPLES), ("hop_samples", HOP_SAMPLES),
            ):
                if saved[key].shape != () or saved[key].item() != expected:
                    raise ValueError(f"Unexpected {key}; re-export with the edge pipeline")
            fingerprint = saved["model_sha256"].item()
            if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                raise ValueError("Invalid YAMNet model fingerprint")
            embeddings = saved["embeddings"]
            count = row["windows"]
            if embeddings.dtype != np.float32 or embeddings.shape != (count, EMBEDDING_SIZE):
                raise ValueError("Embedding shape/dtype disagrees with the manifest")
            if not np.isfinite(embeddings).all():
                raise ValueError("Non-finite embeddings")
            starts, valid = saved["window_start_seconds"], saved["valid_samples"]
            if starts.shape != (count,) or not np.isfinite(starts).all() or not np.allclose(
                starts, np.arange(count) * HOP_SECONDS, rtol=0, atol=1e-6,
            ):
                raise ValueError("Unexpected window start times")
            if (
                valid.shape != (count,) or not np.issubdtype(valid.dtype, np.integer)
                or np.any(valid < 1) or np.any(valid > WINDOW_SAMPLES)
                or np.any(valid[:-1] != WINDOW_SAMPLES)
            ):
                raise ValueError("Invalid observed sample counts")
            length = (count - 1) * HOP_SAMPLES + int(valid[-1])
            expected_count = 1 + (max(0, length - WINDOW_SAMPLES) + HOP_SAMPLES - 1) // HOP_SAMPLES
            if expected_count != count:
                raise ValueError("Extra padded windows do not match edge framing")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid recording archive {path}: {error}") from error
    return Recording(row["path"], row["label"], row["split"], embeddings), fingerprint


def load_training_data(manifest_path: Path) -> TrainingData:
    """Read test assignments for leakage checks, but never load test embeddings."""
    manifest = Path(manifest_path).expanduser().resolve()
    rows = _read_manifest(manifest)
    recordings = {"train": [], "validation": []}
    fingerprints = set()
    for row in rows:
        if row["split"] == "test":
            continue
        recording, fingerprint = _load_recording(manifest.parent, row)
        recordings[row["split"]].append(recording)
        fingerprints.add(fingerprint)
    for split, examples in recordings.items():
        if {example.label for example in examples} != set(LABEL_IDS):
            raise ValueError(f"{split} must contain both falls and non-falls")
    if len(fingerprints) != 1:
        raise ValueError("Examples were exported with different YAMNet models; re-export them together")
    return TrainingData(
        train=tuple(recordings["train"]), validation=tuple(recordings["validation"]),
        provenance={
            "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "yamnet_sha256": fingerprints.pop(), "sample_rate": SAMPLE_RATE,
            "window_samples": WINDOW_SAMPLES, "hop_samples": HOP_SAMPLES,
            "counts": {
                split: {
                    label: sum(row["split"] == split and row["label"] == label for row in rows)
                    for label in LABEL_IDS
                } for split in ("train", "validation", "test")
            },
        },
    )


def pack_recordings(recordings: tuple[Recording, ...]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pad groups for batching; a mask excludes extra batch entries from max scoring.

    Existing edge embeddings, including a WAV's padded final audio window, remain
    valid. Only the additional empty embedding slots introduced here are masked.
    There is one target per recording, never one target per window.
    """
    if not recordings:
        raise ValueError("Cannot pack an empty set of recordings")
    longest = max(len(recording.embeddings) for recording in recordings)
    embeddings = np.zeros((len(recordings), longest, EMBEDDING_SIZE), dtype=np.float32)
    mask = np.zeros((len(recordings), longest, 1), dtype=np.float32)
    labels = np.empty((len(recordings), 1), dtype=np.float32)
    for index, recording in enumerate(recordings):
        count = len(recording.embeddings)
        embeddings[index, :count] = recording.embeddings
        mask[index, :count] = 1
        labels[index, 0] = LABEL_IDS[recording.label]
    return {"embeddings": embeddings, "valid_windows": mask}, labels
