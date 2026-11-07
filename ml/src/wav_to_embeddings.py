"""Convert one local WAV file to YAMNet embeddings for development."""

import argparse
from pathlib import Path

import librosa
import numpy as np

from .features import SAMPLE_RATE, prepare_audio
from .yamnet import extract_embeddings

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_audio(path: str | Path) -> np.ndarray:
    """Read a WAV, resample it if needed, and run shared audio preparation."""
    audio, sample_rate = librosa.load(path, sr=None, mono=False, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        sample_rate = SAMPLE_RATE

    # librosa puts channels first. Shared preparation expects samples first.
    return prepare_audio(audio.T, sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save YAMNet embeddings for one WAV file.")
    parser.add_argument("wav_path", type=Path, help="Path to the WAV file")
    wav_path = parser.parse_args().wav_path
    if wav_path.suffix.lower() != ".wav" or not wav_path.is_file():
        parser.error(f"Expected a WAV file at {wav_path}")

    embeddings = extract_embeddings(load_audio(wav_path))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{wav_path.stem}.npy"
    np.save(output_path, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {output_path}")


if __name__ == "__main__":
    main()
