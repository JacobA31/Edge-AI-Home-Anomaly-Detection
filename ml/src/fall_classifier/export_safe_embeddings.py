"""Save recording-level SAFE examples using the actual edge audio pipeline.

Run from ml/:
    python -m src.fall_classifier.export_safe_embeddings data/safe/splits.csv

Requires the local model produced by python -m src.export_yamnet. No downloads
or training happen here. Defaults to models/yamnet.tflite and writes beneath
data/processed/safe_classifier/. Use --model and --output-dir to override them.

Each WAV becomes one compressed .npz with embeddings[N, 1024], window start
times relative to the recording, observed sample counts, and scalar source_path,
label and split. Labels describe entire recordings, NOT individual windows.
Do not shuffle these windows into different sets or assume each fall-labelled
recording contains an impact in every window. No pooling or classifier is chosen.

manifest.csv contains path,label,split,features_path,windows. Source paths are
relative to the input CSV's folder; feature paths are relative to the output
folder. Archives also store model_sha256 and the sample/window/hop settings.
Only the last incomplete window is padded, after per-window DC removal. The
decoder retains channels and DC offset until edge's AudioPipeline prepares them.
All examples are processed in a staging folder before existing outputs are
replaced; manifest.csv is published last. Reruns rebuild the listed recordings.
"""

import argparse
import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import numpy as np
from e3_edge.audio.features import HOP_SECONDS, SAMPLE_RATE
from e3_edge.audio.pipeline import AudioPipeline
from e3_edge.audio.yamnet import EMBEDDING_SIZE, WINDOW_SAMPLES

from .index_safe import LABELS
from .split_safe import FILENAME

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ML_ROOT / "models" / "yamnet.tflite"
DEFAULT_OUTPUT = ML_ROOT / "data" / "processed" / "safe_classifier"
HOP_SAMPLES = round(HOP_SECONDS * SAMPLE_RATE)
SPLITS = {"train", "validation", "test"}


def _read_splits(csv_path: Path) -> list[tuple[Path, str, str]]:
    """Validate all assignments before loading a model or writing output."""
    root = csv_path.parent
    rows = []
    seen = set()
    destinations = set()
    source_splits = {}
    fold_splits = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, strict=True)
        if (
            not reader.fieldnames
            or len(reader.fieldnames) != 3
            or set(reader.fieldnames)
            != {
                "path",
                "label",
                "split",
            }
        ):
            raise ValueError("Expected path,label,split columns from src.fall_classifier.split_safe")
        for line, row in enumerate(reader, start=2):
            if None in row or any(not (row.get(key) or "").strip() for key in ("path", "label", "split")):
                raise ValueError(f"Invalid split CSV row {line}")
            relative = Path(row["path"].strip())
            label, split = row["label"].strip(), row["split"].strip()
            path = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
                raise ValueError(f"Row {line}: recording path must stay inside the CSV folder")
            metadata = FILENAME.fullmatch(relative.stem)
            if relative.suffix.lower() != ".wav" or not path.is_file() or metadata is None:
                raise ValueError(f"Row {line}: expected an existing SAFE WAV: {relative}")
            if label != LABELS[metadata["class"]] or split not in SPLITS:
                raise ValueError(f"Row {line}: filename/label mismatch or unknown split")
            destination = relative.with_suffix(".npz").as_posix().casefold()
            if path in seen or destination in destinations:
                raise ValueError(f"Duplicate recording or feature output: {relative}")
            seen.add(path)
            destinations.add(destination)
            source_group = (metadata["class"], metadata["environment"], metadata["source"])
            if source_splits.setdefault(source_group, split) != split:
                raise ValueError(f"Source recording crosses splits: {relative}")
            if fold_splits.setdefault(metadata["fold"], split) != split:
                raise ValueError(f"SAFE fold {metadata['fold']} crosses splits")
            rows.append((relative, label, split))
    if not rows:
        raise ValueError("The split CSV contains no recordings")
    return sorted(rows, key=lambda row: row[0].as_posix())


def _load_recording(path: Path) -> np.ndarray:
    """Decode and resample only; edge handles channel mixing and DC removal."""
    samples, sample_rate = librosa.load(path, sr=None, mono=False, dtype=np.float32)
    if samples.size == 0 or samples.ndim not in (1, 2) or not np.isfinite(samples).all():
        raise ValueError(f"Expected nonempty, finite mono or multichannel audio: {path}")
    if sample_rate != SAMPLE_RATE:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=SAMPLE_RATE, axis=-1)
    return np.ascontiguousarray(samples.T, dtype=np.float32)


def _save_recording(
    samples: np.ndarray,
    audio: AudioPipeline,
    destination: Path,
    *,
    source_path: Path,
    label: str,
    split: str,
    model_sha256: str,
) -> int:
    """Keep every window from a recording in one archive, with one recording label."""
    count = 1 + (max(0, len(samples) - WINDOW_SAMPLES) + HOP_SAMPLES - 1) // HOP_SAMPLES
    starts = np.arange(count, dtype=np.int64) * HOP_SAMPLES
    valid_samples = np.minimum(WINDOW_SAMPLES, len(samples) - starts)
    embeddings = np.empty((count, EMBEDDING_SIZE), dtype=np.float32)
    for index, start in enumerate(starts):
        raw = samples[start : start + WINDOW_SAMPLES]
        frame = audio.process_window(
            raw,
            start_time=float(start / SAMPLE_RATE),
            sample_rate=SAMPLE_RATE,
            pad_end=len(raw) < WINDOW_SAMPLES,
        )
        embedding = np.asarray(frame.embedding)
        if embedding.shape != (EMBEDDING_SIZE,) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid embedding for {source_path}, window {index + 1}")
        embeddings[index] = embedding
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        embeddings=embeddings,
        window_start_seconds=starts / SAMPLE_RATE,
        valid_samples=valid_samples,
        source_path=source_path.as_posix(),
        label=label,
        split=split,
        sample_rate=SAMPLE_RATE,
        window_samples=WINDOW_SAMPLES,
        hop_samples=HOP_SAMPLES,
        model_sha256=model_sha256,
    )
    return count


def export_safe_embeddings(
    splits_csv: Path,
    model_path: Path = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    """Export every listed recording, reusing one loaded edge YAMNet model."""
    splits_csv = Path(splits_csv).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    rows = _read_splits(splits_csv)
    # Keep generated files away from the source dataset and the model.
    if splits_csv.is_relative_to(output_dir) or model_path.is_relative_to(output_dir):
        raise ValueError("Output must be a separate processed-data folder, outside the CSV/model folders")
    if output_dir.is_relative_to(splits_csv.parent):
        raise ValueError("Output must be outside the source SAFE folder")
    audio = AudioPipeline(model_path)
    with model_path.open("rb") as model:
        model_sha256 = hashlib.file_digest(model, "sha256").hexdigest()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    total_windows = 0
    counts = dict.fromkeys(("train", "validation", "test"), 0)
    print(f"Processing {len(rows)} recordings with edge YAMNet: {model_path}", flush=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=".safe-embeddings-") as temporary:
        staging = Path(temporary)
        for index, (relative, label, split) in enumerate(rows, start=1):
            features_path = Path("recordings") / relative.with_suffix(".npz")
            try:
                windows = _save_recording(
                    _load_recording(splits_csv.parent / relative),
                    audio,
                    staging / features_path,
                    source_path=relative,
                    label=label,
                    split=split,
                    model_sha256=model_sha256,
                )
            except (OSError, ValueError, RuntimeError) as error:
                raise ValueError(f"Failed to process {relative}: {error}") from error
            manifest.append((relative.as_posix(), label, split, features_path.as_posix(), windows))
            counts[split] += 1
            total_windows += windows
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} recordings ({total_windows} windows)", flush=True)
        with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["path", "label", "split", "features_path", "windows"])
            writer.writerows(manifest)
        # Publish only after every recording succeeds; replace each complete file
        # on the same filesystem, with the manifest last. This is not a directory transaction.
        for _, _, _, features_path, _ in manifest:
            destination = output_dir / features_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / features_path).replace(destination)
        (staging / "manifest.csv").replace(output_dir / "manifest.csv")
    print(f"Saved {len(rows)} recordings, {total_windows} windows")
    print(f"Recordings by split: {counts}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")
    return output_dir / "manifest.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "splits_csv",
        type=Path,
        help="SAFE splits.csv produced by src.fall_classifier.split_safe",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Exported YAMNet .tflite file")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Processed-data output folder"
    )
    args = parser.parse_args(argv)
    try:
        export_safe_embeddings(args.splits_csv, args.model, args.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError, csv.Error) as error:
        parser.exit(1, f"SAFE embedding export failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
