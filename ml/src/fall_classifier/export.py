"""Export the selected fall classifier for the Pi, without retraining.

Run from ml/:
    python -m src.fall_classifier.export

Reads fall_classifier_selection.json; paths inside it are relative to ml/.
Writes models/fall_classifier.tflite and models/fall_classifier.json. Rerunning
replaces these exports after conversion and structural checks succeed. Original
training artifacts, selection, and evaluation reports are unchanged.

The float32 model takes one embedding[1, 1024] and returns fall_score[1, 1].
The threshold remains in the companion JSON; it is not baked into the model.
Three synthetic inputs check basic execution. Full comparison on saved SAFE
embeddings is a separate verification step.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, OpResolverType

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = ML_ROOT / "fall_classifier_selection.json"
DEFAULT_OUTPUT = ML_ROOT / "models" / "fall_classifier.tflite"
SIGNATURE_NAME = "serving_default"
INPUT_NAME = "embedding"
OUTPUT_NAME = "fall_score"
INPUT_SHAPE = (1, 1024)
OUTPUT_SHAPE = (1, 1)


def _load_selection(path: Path) -> tuple[dict, Path]:
    """Identify the selected source model and verify its recorded fingerprint."""
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != 1 or selection.get("paths_relative_to") != "ml/":
        raise ValueError("Expected a version-1 selection with paths relative to ml/")
    relative = Path(selection["model_path"])
    model_path = (ML_ROOT / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not model_path.is_relative_to(ML_ROOT):
        raise ValueError("Selected model_path must be relative to ml/ and stay inside it")
    threshold = selection["threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Selected threshold must be a finite number between 0 and 1")
    if (
        selection.get("input_shape") != [None, 1024]
        or selection.get("output_shape") != [None, 1]
        or selection.get("label_ids") != {"non_fall": 0, "fall": 1}
        or selection.get("decision_rule") != "score >= threshold"
        or selection.get("recording_score") != "maximum valid window score"
    ):
        raise ValueError("Selection must describe the recording-max fall window scorer")
    with model_path.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    if fingerprint != selection["model_sha256"]:
        raise ValueError("Selected model fingerprint does not match; refusing to export a different model")
    return selection, model_path


def _convert_model(model: tf.keras.Model) -> bytes:
    """Expose one fixed-size window score using only built-in LiteRT operations."""
    module = tf.Module()
    module.model = model
    # Freeze Keras weights as constants, so the Pi needs no TensorFlow variable
    # initialization. A resource-variable graph can convert but fail at inference.
    trainable = [tf.constant(variable.numpy()) for variable in model.trainable_variables]
    non_trainable = [tf.constant(variable.numpy()) for variable in model.non_trainable_variables]

    def infer(embedding):
        scores, _state = module.model.stateless_call(
            trainable,
            non_trainable,
            embedding,
            training=False,
        )
        return {OUTPUT_NAME: scores}

    module.infer = tf.function(
        infer,
        input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32, name=INPUT_NAME)],
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.infer.get_concrete_function()],
        module,
    )
    # The Pi must not need full TensorFlow or the Flex runtime.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def _check_export(model_bytes: bytes, reference: tf.keras.Model) -> None:
    """Check structure and a few synthetic inputs before publishing the export."""
    interpreter = Interpreter(
        model_content=model_bytes,
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    signature = interpreter.get_signature_list().get(SIGNATURE_NAME, {})
    if signature.get("inputs") != [INPUT_NAME] or signature.get("outputs") != [OUTPUT_NAME]:
        raise ValueError("Export must expose embedding input and fall_score output")
    runner = interpreter.get_signature_runner(SIGNATURE_NAME)
    for details, name, shape in (
        (runner.get_input_details(), INPUT_NAME, INPUT_SHAPE),
        (runner.get_output_details(), OUTPUT_NAME, OUTPUT_SHAPE),
    ):
        tensor = details[name]
        if tensor["dtype"] != np.float32 or tuple(tensor["shape"]) != shape:
            raise ValueError(f"Exported {name} must be float32 with shape {shape}")
    probes = (
        np.zeros(INPUT_SHAPE, dtype=np.float32),
        np.ones(INPUT_SHAPE, dtype=np.float32),
        np.random.default_rng(42).uniform(0, 2, INPUT_SHAPE).astype(np.float32),
    )
    for embedding in probes:
        actual = runner(embedding=embedding)[OUTPUT_NAME]
        expected = np.asarray(reference(embedding, training=False))
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
            raise ValueError("Export failed its synthetic execution check against the selected model")


def export_classifier(selection_path: Path = DEFAULT_SELECTION, output_path: Path = DEFAULT_OUTPUT) -> Path:
    selection_path = Path(selection_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix != ".tflite":
        raise ValueError("Output must have a .tflite extension")
    metadata_path = output_path.with_suffix(".json")
    selection, model_path = _load_selection(selection_path)
    if metadata_path == selection_path or output_path.is_relative_to(model_path.parent):
        raise ValueError("Export destination must be separate from selection and original training artifacts")
    print(f"Verified selected model: {model_path}", flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.input_shape != (None, 1024) or model.output_shape != (None, 1):
        raise ValueError("Expected the selected 1024-input, one-score window classifier")
    print("Converting one YAMNet embedding to a float32 LiteRT model...", flush=True)
    model_bytes = _convert_model(model)
    _check_export(model_bytes, model)
    metadata = {
        "schema_version": 1,
        "model_file": output_path.name,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "source_model_sha256": selection["model_sha256"],
        "training_run": selection["training_run"],
        "best_epoch": selection["best_epoch"],
        "embedding_scaling": selection["embedding_scaling"],
        "threshold": selection["threshold"],
        "decision_rule": selection["decision_rule"],
        "recording_score": selection["recording_score"],
        "label_ids": selection["label_ids"],
        "signature_name": SIGNATURE_NAME,
        "input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "dtype": "float32"},
        "output": {"name": OUTPUT_NAME, "shape": list(OUTPUT_SHAPE), "dtype": "float32"},
        "yamnet_sha256": selection["yamnet_sha256"],
        "manifest_sha256": selection["manifest_sha256"],
        "selection_method": selection["selection_method"],
        "test_used_for_threshold_selection": selection["test_used_for_threshold_selection"],
        "exported_utc": datetime.now(UTC).isoformat(),
        "tensorflow_version": tf.__version__,
        "synthetic_checks_passed": True,
        "dataset_parity_verified": False,
    }
    # Prepare both files before replacing any existing export. Publish metadata last;
    # its checksum identifies the exact model file it belongs to.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent, prefix=".export-") as temporary:
        staged_model = Path(temporary) / output_path.name
        staged_metadata = Path(temporary) / metadata_path.name
        staged_model.write_bytes(model_bytes)
        staged_metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged_model.replace(output_path)
        staged_metadata.replace(metadata_path)
    print(f"Saved model: {output_path} ({len(model_bytes):,} bytes)", flush=True)
    print(f"Saved settings: {metadata_path} (threshold {selection['threshold']})", flush=True)
    print("Structure and synthetic checks passed. SAFE embedding comparison is the next step.", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        export_classifier(args.selection, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Fall classifier export failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
