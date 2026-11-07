"""Export pretrained YAMNet for the edge LiteRT runner (run from ml/).

    python -m src.export_yamnet
    python -m src.export_yamnet --output models/yamnet.tflite

This is a development tool, not training or Pi startup code. It downloads the Hub
model if needed, exports one fixed waveform window, and checks LiteRT's results
against the original model before saving. Requires ml/requirements.txt.
"""

import argparse
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .features import SAMPLE_RATE
from .yamnet import CLASS_COUNT, EMBEDDING_SIZE, WINDOW_SAMPLES, _load_yamnet_model

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "yamnet.tflite"
VERIFICATION_RTOL = 1e-3
VERIFICATION_ATOL = 1e-4
OUTPUT_SHAPES = {"embeddings": (1, EMBEDDING_SIZE), "scores": (1, CLASS_COUNT)}


def _convert_model(model) -> bytes:
    """Freeze a fixed-window Hub call, exposing named outputs and builtin ops only."""
    # Load the large export dependencies only when exporting, so --help and the
    # offline unit tests work without TensorFlow.
    tf = import_module("tensorflow")
    module = tf.Module()
    module.model = model

    def infer(waveform):
        scores, embeddings, _spectrogram = module.model(waveform)
        return {"scores": scores, "embeddings": embeddings}

    module.infer = tf.function(
        infer, input_signature=[tf.TensorSpec([WINDOW_SAMPLES], tf.float32, name="waveform")],
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.infer.get_concrete_function()], module,
    )
    # No SELECT_TF_OPS fallback: the Pi must not need full TensorFlow/Flex.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def _verification_audio() -> dict[str, np.ndarray]:
    """Exercise silence, a tone, and repeatable noise without dataset recordings."""
    time = np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    return {
        "silence": np.zeros(WINDOW_SAMPLES, dtype=np.float32),
        "tone": (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32),
        "noise": np.random.default_rng(0).uniform(-0.25, 0.25, WINDOW_SAMPLES).astype(np.float32),
    }


def _checked_output(value, name: str) -> np.ndarray:
    output = np.asarray(value)
    if output.dtype != np.float32 or output.shape != OUTPUT_SHAPES[name]:
        raise ValueError(f"Expected {name} float32 {OUTPUT_SHAPES[name]}, got {output.dtype} {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite values in {name}")
    if name == "scores" and np.any((output < 0) | (output > 1)):
        raise ValueError("Class scores must be between 0 and 1")
    return output


def verify_model(model_bytes: bytes, reference_model) -> None:
    """Run the export in LiteRT and compare both outputs with Hub YAMNet."""
    litert = import_module("ai_edge_litert.interpreter")
    interpreter = litert.Interpreter(
        model_content=model_bytes, num_threads=1,
        experimental_op_resolver_type=litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    if (len(inputs) != 1 or inputs[0]["dtype"] != np.float32
            or tuple(inputs[0]["shape"]) != (WINDOW_SAMPLES,)):
        raise ValueError(f"Expected one float32 waveform input with {WINDOW_SAMPLES} samples")
    signature = interpreter.get_signature_list().get("serving_default", {})
    if signature.get("inputs") != ["waveform"] or set(signature.get("outputs", [])) != set(OUTPUT_SHAPES):
        raise ValueError("Export needs waveform input and named embeddings/scores outputs")
    runner = interpreter.get_signature_runner("serving_default")

    for label, waveform in _verification_audio().items():
        scores, embeddings, _spectrogram = reference_model(waveform)
        expected = {"embeddings": embeddings, "scores": scores}
        actual = runner(waveform=waveform)
        for name in OUTPUT_SHAPES:
            reference = _checked_output(expected[name], name)
            exported = _checked_output(actual[name], name)
            if not np.allclose(exported, reference, rtol=VERIFICATION_RTOL, atol=VERIFICATION_ATOL):
                error = float(np.max(np.abs(exported - reference)))
                raise ValueError(
                    f"LiteRT {name} differ from YAMNet on {label} (max absolute error {error:g})",
                )
        print(f"Verified {label}: embeddings (1, {EMBEDDING_SIZE}), scores (1, {CLASS_COUNT})", flush=True)


def export_yamnet(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Export and verify before atomically replacing the destination model file."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".tflite":
        raise ValueError("Output must have a .tflite extension")
    print("Loading pretrained YAMNet (downloads on first use)...", flush=True)
    model = _load_yamnet_model()
    print(f"Converting one {WINDOW_SAMPLES}-sample waveform window to LiteRT...", flush=True)
    model_bytes = _convert_model(model)
    print("Checking LiteRT outputs against the original YAMNet...", flush=True)
    verify_model(model_bytes, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".tflite", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_bytes(model_bytes)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved verified model to {output_path.resolve()} ({len(model_bytes):,} bytes)", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination .tflite model file")
    args = parser.parse_args(argv)
    try:
        export_yamnet(args.output)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"YAMNet export failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
