"""Compare the edge audio pipeline with original Hub YAMNet using one WAV.

Run from ml/ with its requirements installed:
    python -m src.check_edge_audio data/safe/10-914-00-466-02.wav
    python -m src.check_edge_audio recording.wav --model models/yamnet.tflite

Both paths prepare each window separately, including DC removal before padding.
The existing dataset exporter prepares whole recordings; this tool verifies
agreement with a window-matched reference, not equivalence to whole-recording exports.
Timestamps are replay times starting at zero unless --start-time is supplied.
No audio, frames, or model files are written. The first Hub call may download YAMNet.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from e3_edge.anomaly.frames import frame_level_dbfs
from e3_edge.audio import features as edge_features
from e3_edge.audio.pipeline import AudioPipeline
from e3_edge.audio.yamnet import WINDOW_SAMPLES

from . import features, yamnet
from .export_yamnet import VERIFICATION_ATOL, VERIFICATION_RTOL

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "yamnet.tflite"


@dataclass(frozen=True)
class ComparisonSummary:
    windows: int
    padded_windows: int
    duration_seconds: float
    max_audio_error: float
    max_embedding_error: float
    max_score_error: float
    max_level_error_db: float
    max_timestamp_error_seconds: float


def load_recording(path: Path) -> np.ndarray:
    """Decode/resample a WAV to 16 kHz, retaining its channels and DC offset."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a WAV file: {path}")
    samples, sample_rate = librosa.load(path, sr=None, mono=False, dtype=np.float32)
    if samples.size == 0 or samples.ndim not in (1, 2):
        raise ValueError(f"Expected nonempty mono or multichannel audio: {path}")
    if sample_rate != features.SAMPLE_RATE:
        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=features.SAMPLE_RATE, axis=-1,
        )
    # Librosa returns (channels, samples). Do not use export_frames.load_audio:
    # it removes DC and mixes channels over the whole recording before windowing.
    return np.ascontiguousarray(samples.T, dtype=np.float32)


def _check_settings() -> None:
    for name in ("SAMPLE_RATE", "FRAME_SECONDS", "HOP_SECONDS"):
        if getattr(features, name) != getattr(edge_features, name):
            raise ValueError(f"Laptop and edge {name} settings differ")
    if yamnet.WINDOW_SAMPLES != WINDOW_SAMPLES:
        raise ValueError("Laptop and edge waveform window lengths differ")


def _array_error(actual, expected, *, name: str, window: int, rtol: float, atol: float) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != np.float32 or expected.dtype != np.float32 or actual.shape != expected.shape:
        raise ValueError(f"Window {window}: {name} must have matching float32 shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Window {window}: non-finite {name}")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"Window {window}: {name} differ (max absolute error {error:g})")
    return error


def _scalar_error(actual, expected: float, *, name: str, window: int, atol: float) -> float:
    if actual is None or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"Window {window}: {name} differ (edge {actual}, reference {expected})")
    return abs(float(actual) - expected)


def compare_recording(
    path: Path, model_path: Path = DEFAULT_MODEL, *, start_time: float = 0.0,
) -> ComparisonSummary:
    """Run a WAV through the edge component and a reference with identical window boundaries."""
    _check_settings()
    start_time = float(start_time)
    if not np.isfinite(start_time):
        raise ValueError("start_time must be finite replay Unix seconds")
    samples = load_recording(path)
    count = yamnet.window_count(len(samples))
    hop_samples = round(edge_features.HOP_SECONDS * edge_features.SAMPLE_RATE)
    duration = len(samples) / features.SAMPLE_RATE
    print(f"Loaded {path}: {duration:.3f} s at {features.SAMPLE_RATE} Hz; {count} windows", flush=True)
    print("Reference: original Hub YAMNet with per-window preprocessing", flush=True)
    audio = AudioPipeline(model_path)
    max_audio = max_embedding = max_score = max_level = max_timestamp = 0.0
    padded_windows = 0

    for index in range(count):
        window = index + 1
        start = index * hop_samples
        raw = samples[start:start + WINDOW_SAMPLES]
        timestamp = start_time + start / features.SAMPLE_RATE
        pad_end = len(raw) < WINDOW_SAMPLES
        padded_windows += int(pad_end)

        # Keep the laptop implementation independent, but give both paths the
        # same raw window so their channel means cover the same observed samples.
        reference_audio = features.prepare_audio(raw, features.SAMPLE_RATE)
        prepared_edge = edge_features.prepare_audio(raw, edge_features.SAMPLE_RATE)
        max_audio = max(max_audio, _array_error(
            prepared_edge, reference_audio, name="prepared audio", window=window, rtol=1e-6, atol=1e-7,
        ))
        reference_level = frame_level_dbfs(reference_audio)
        if pad_end:
            reference_audio = np.pad(reference_audio, (0, WINDOW_SAMPLES - len(reference_audio)))
        reference_embeddings, reference_scores = yamnet.extract_features(reference_audio)
        frame = audio.process_window(
            raw, start_time=timestamp, sample_rate=features.SAMPLE_RATE, pad_end=pad_end,
        )

        max_embedding = max(max_embedding, _array_error(
            frame.embedding, reference_embeddings[0], name="embeddings", window=window,
            rtol=VERIFICATION_RTOL, atol=VERIFICATION_ATOL,
        ))
        max_score = max(max_score, _array_error(
            frame.scores, reference_scores[0], name="class scores", window=window,
            rtol=VERIFICATION_RTOL, atol=VERIFICATION_ATOL,
        ))
        max_level = max(max_level, _scalar_error(
            frame.level_dbfs, reference_level, name="audio levels", window=window, atol=1e-5,
        ))
        max_timestamp = max(max_timestamp, _scalar_error(
            frame.t, timestamp, name="timestamps", window=window, atol=1e-6,
        ))
        if window == 1 or window % 25 == 0 or window == count:
            print(f"Checked window {window}/{count}", flush=True)

    return ComparisonSummary(
        windows=count, padded_windows=padded_windows, duration_seconds=duration,
        max_audio_error=max_audio, max_embedding_error=max_embedding, max_score_error=max_score,
        max_level_error_db=max_level, max_timestamp_error_seconds=max_timestamp,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="One WAV recording to replay")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Exported YAMNet .tflite file")
    parser.add_argument("--start-time", type=float, default=0.0, help="Replay start in Unix seconds (default: 0)")
    args = parser.parse_args(argv)
    try:
        result = compare_recording(args.input, args.model, start_time=args.start_time)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Edge audio check failed: {error}\n")
    print(f"PASS: {result.windows} windows matched; padded final windows: {result.padded_windows}")
    print(f"Max prepared-audio error: {result.max_audio_error:g}")
    print(f"Max embedding error: {result.max_embedding_error:g}")
    print(f"Max class-score error: {result.max_score_error:g}")
    print(f"Max audio-level error: {result.max_level_error_db:g} dB")
    print(f"Max timestamp error: {result.max_timestamp_error_seconds:g} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
