"""Development exporter for the anomaly evaluator's FrameStream .npz files.

Run from ml/:
  python -m src.export_frames safe data/safe
  python -m src.export_frames sins data/sins/living/recordings.csv

The SINS CSV describes ONE room/microphone and has columns path,start_time.
Paths are relative to the CSV. Times must include an offset, for example
Node1_20170701_080000_000_audio.wav,2017-07-01T08:00:00+02:00
Use verified recording times; the exporter does not infer a timezone from filenames.
"""

import argparse
import csv
from datetime import datetime
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import librosa
import numpy as np
from e3_edge.anomaly.frames import FrameStream, frame_level_dbfs

from .features import FRAME_SECONDS, HOP_SECONDS, SAMPLE_RATE, prepare_audio
from .yamnet import HOP_SAMPLES, WINDOW_SAMPLES, extract_features

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
SAFE_REPLAY_START = 946684800.0  # 2000-01-01 UTC; artificial replay time, not recording time.
SAFE_GAP_SECONDS = 6.0  # Evaluator uses gaps > 5 s to separate SAFE clips.
SAFE_LABELS = {"01": "falls", "02": "non_falls"}


def load_audio(path: Path) -> np.ndarray:
    """Read a development WAV and adapt it to shared preparation's input contract."""
    audio, sample_rate = librosa.load(path, sr=None, mono=False, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Empty WAV: {path}")
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return prepare_audio(audio.T, SAMPLE_RATE)


def make_stream(waveform: np.ndarray, start_time: float) -> FrameStream:
    """Align every YAMNet row with its start time and observed waveform level."""
    if not np.isfinite(start_time):
        raise ValueError("Start time must be finite UTC Unix seconds")
    embeddings, scores = extract_features(waveform)
    starts = np.arange(len(embeddings)) * HOP_SAMPLES
    # Match the waveform covered by each model patch. For the final padded
    # patch, measure actual recorded samples so padding cannot create fake quiet.
    levels = np.array([
        frame_level_dbfs(waveform[start:start + WINDOW_SAMPLES]) for start in starts
    ], dtype=np.float32)
    return FrameStream(
        timestamps=start_time + np.arange(len(embeddings), dtype=np.float64) * HOP_SECONDS,
        embeddings=embeddings,
        scores=scores,
        level_dbfs=levels,
    )


def _join_streams(streams: list[FrameStream]) -> FrameStream:
    return FrameStream(
        timestamps=np.concatenate([s.timestamps for s in streams]),
        embeddings=np.concatenate([s.embeddings for s in streams]),
        scores=np.concatenate([s.scores for s in streams]),
        level_dbfs=np.concatenate([s.level_dbfs for s in streams]),
    )


def _save_stream(stream: FrameStream, path: Path) -> None:
    """Replace an output only once its complete FrameStream has been saved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        stream.save(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved {len(stream)} windows to {path}")


def export_safe(input_path: Path, output_dir: Path) -> list[Path]:
    """Save falls.npz and/or non_falls.npz using the confirmed SAFE suffix labels."""
    if input_path.is_dir():
        paths = sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    else:
        paths = [input_path]
    if not paths:
        raise ValueError(f"No WAV files found in {input_path}")
    groups: dict[str, list[Path]] = {label: [] for label in SAFE_LABELS.values()}
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a WAV file: {path}")
        code = path.stem.rsplit("-", 1)[-1]
        if code not in SAFE_LABELS:
            raise ValueError(f"SAFE filename must end in -01.wav or -02.wav: {path.name}")
        groups[SAFE_LABELS[code]].append(path)

    outputs = []
    for label, clips in groups.items():
        if not clips:
            continue
        cursor = SAFE_REPLAY_START
        streams = []
        for path in clips:
            stream = make_stream(load_audio(path), cursor)
            streams.append(stream)
            cursor = float(stream.timestamps[-1]) + FRAME_SECONDS + SAFE_GAP_SECONDS
        output = output_dir / f"{label}.npz"
        _save_stream(_join_streams(streams), output)
        outputs.append(output)
    return outputs


def _read_recordings(manifest: Path) -> list[tuple[float, Path]]:
    recordings = []
    seen = set()
    with manifest.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"path", "start_time"}.issubset(reader.fieldnames):
            raise ValueError("SINS CSV needs path,start_time columns for one room/microphone")
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("path") or "").strip()
            time_text = (row.get("start_time") or "").strip()
            if not filename or not time_text:
                raise ValueError(f"Missing path or start_time on CSV row {row_number}")
            start = datetime.fromisoformat(time_text.replace("Z", "+00:00"))
            if start.utcoffset() is None:
                raise ValueError(f"Include a timezone offset in start_time on CSV row {row_number}")
            path = (manifest.parent / filename).resolve()
            if not path.is_file() or path.suffix.lower() != ".wav":
                raise ValueError(f"Expected a WAV file on CSV row {row_number}: {path}")
            if path in seen:
                raise ValueError(f"Duplicate WAV in SINS CSV: {path}")
            seen.add(path)
            recordings.append((start.timestamp(), path))
    if not recordings:
        raise ValueError("SINS CSV has no recordings")
    return sorted(recordings)


def _daily_parts(recordings: list[tuple[float, Path]], timezone: ZoneInfo):
    previous_end = -np.inf
    for start_time, path in recordings:
        waveform = load_audio(path)
        if start_time < previous_end - 1 / SAMPLE_RATE:
            raise ValueError(f"Overlapping SINS recordings at {path}; check the recording start times")
        previous_end = start_time + len(waveform) / SAMPLE_RATE
        stream = make_stream(waveform, start_time)
        days = [datetime.fromtimestamp(float(t), timezone).date() for t in stream.timestamps]
        begin = 0
        for day, rows in groupby(days):
            stop = begin + sum(1 for _ in rows)
            yield day, stream.slice(begin, stop)
            begin = stop


def export_sins(manifest: Path, output_dir: Path, timezone: str = "Europe/Brussels") -> list[Path]:
    """Save one stream per local date, preserving the CSV's actual UTC start times."""
    recordings = _read_recordings(manifest)
    daily_parts = _daily_parts(recordings, ZoneInfo(timezone))
    outputs = []
    # Work one day at a time, rather than retaining a week of embeddings in RAM.
    for day, parts in groupby(daily_parts, key=lambda item: item[0]):
        output = output_dir / f"{day.isoformat()}.npz"
        _save_stream(_join_streams([stream for _, stream in parts]), output)
        outputs.append(output)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="dataset", required=True)
    safe = commands.add_parser("safe", help="Export SAFE WAVs grouped by -01/-02 labels")
    safe.add_argument("input", type=Path, help="One SAFE WAV or a folder of SAFE WAVs")
    safe.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "safe")
    sins = commands.add_parser("sins", help="Export one room/microphone using a path,start_time CSV")
    sins.add_argument(
        "manifest", type=Path, help="CSV paths are relative to this file; times include offsets",
    )
    sins.add_argument("--timezone", default="Europe/Brussels", help="Timezone used to group windows by date")
    sins.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "sins")
    args = parser.parse_args(argv)
    try:
        if args.dataset == "safe":
            export_safe(args.input, args.output_dir)
        else:
            export_sins(args.manifest, args.output_dir, args.timezone)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
