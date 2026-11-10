"""Compare the selected Keras classifier and real edge runner on saved SAFE embeddings.

Run from ml/:
    python -m src.fall_classifier.verify_export

Checks every window in train, validation and test, then compares recording-max
decisions at the already-selected threshold. Validation/test metrics are reported
separately. No training, threshold search, audio processing or downloads occur.

Writes reports/fall_classifier/export-verification/{report.md,results.json,predictions.csv}
and updates the export's companion JSON with the verification result. Original
training artifacts, selection and evaluation reports are unchanged.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from e3_edge.fall_classifier import FallClassifierRunner

from .dataset import LABEL_IDS, Recording, load_evaluation_split, load_training_data
from .evaluate import _save_json, _write_csv, classification_metrics, proposal_checks
from .export import (
    DEFAULT_OUTPUT,
    DEFAULT_SELECTION,
    ML_ROOT,
    _load_selection,
)

DEFAULT_REPORT = ML_ROOT / "reports" / "fall_classifier" / "export-verification"
RTOL = 1e-4
ATOL = 1e-5
PREDICTION_COLUMNS = [
    "path",
    "split",
    "label",
    "windows",
    "keras_score",
    "tflite_score",
    "max_window_error",
    "window_score_failures",
    "window_decision_mismatches",
    "keras_decision",
    "tflite_decision",
    "recording_decision_match",
]


def _load_export(path: Path, selection: dict):
    """Bind the saved selection to the model/settings loaded by the edge runner."""
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Export metadata must be a JSON object")
    if metadata.get("source_model_sha256") != selection["model_sha256"]:
        raise ValueError("Export came from a different Keras model")
    for key in (
        "threshold",
        "decision_rule",
        "recording_score",
        "label_ids",
        "embedding_scaling",
        "yamnet_sha256",
        "manifest_sha256",
    ):
        if metadata.get(key) != selection[key]:
            raise ValueError(f"Exported {key} differs from the saved selection; re-export first")
    runner = FallClassifierRunner(path)
    if (
        runner.model_sha256 != metadata.get("model_sha256")
        or runner.threshold != selection["threshold"]
        or runner.yamnet_sha256 != selection["yamnet_sha256"]
    ):
        raise ValueError("Loaded edge model/settings differ from the selected export")
    return metadata_path, metadata, runner


def _scores(value, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(value)
    if (
        values.dtype != np.float32
        or values.shape != shape
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError(f"Expected finite float32 scores in [0, 1] with shape {shape}")
    return values[:, 0].astype(np.float64)


def _compare(recording: Recording, model, runner: FallClassifierRunner, threshold: float) -> dict:
    count = len(recording.embeddings)
    reference = _scores(model(recording.embeddings, training=False), (count, 1))
    actual = np.array([runner.predict(embedding) for embedding in recording.embeddings])
    keras_score, tflite_score = float(reference.max()), float(actual.max())
    keras_decision = "fall" if keras_score >= threshold else "non_fall"
    tflite_decision = "fall" if tflite_score >= threshold else "non_fall"
    return {
        "path": recording.source_path,
        "split": recording.split,
        "label": recording.label,
        "windows": count,
        "keras_score": keras_score,
        "tflite_score": tflite_score,
        "max_window_error": float(np.max(np.abs(reference - actual))),
        "window_score_failures": int(np.count_nonzero(~np.isclose(actual, reference, rtol=RTOL, atol=ATOL))),
        "window_decision_mismatches": int(
            np.count_nonzero((reference >= threshold) != (actual >= threshold))
        ),
        "keras_decision": keras_decision,
        "tflite_decision": tflite_decision,
        "recording_decision_match": keras_decision == tflite_decision,
    }


def _report(result: dict) -> str:
    lines = [
        "# Fall classifier export verification",
        "",
        f"**{'PASS' if result['passed'] else 'FAIL'}** at fixed threshold **{result['threshold']}**.",
        f"Checked {result['recordings']} recordings and {result['windows']} audio windows.",
        f"Keras SHA-256: `{result['source_model_sha256']}`.",
        f"LiteRT SHA-256: `{result['model_sha256']}`.",
        f"LiteRT inference uses `{result['inference_runner']}`.",
        "",
        f"Maximum absolute window-score difference: **{result['max_window_error']:.8g}**.",
        f"Scores outside tolerance: **{result['window_score_failures']}**.",
        f"Window decisions differing: **{result['window_decision_mismatches']}**.",
        f"Recording decisions differing: **{result['recording_decision_mismatches']}**.",
        f"Tolerance: absolute {ATOL:g} + relative {RTOL:g} times the original score.",
        "",
        "## Validation and test results",
        "",
        "| Group | Version | Accuracy | Falls missed | False alarms | NF3 targets |",
        "|---|---|---:|---:|---:|---|",
    ]
    for split, versions in result["metrics"].items():
        for version, metrics in versions.items():
            lines.append(
                f"| {split} | {version} | {metrics['accuracy']:.2%} | "
                f"{metrics['false_negatives']}/{metrics['falls']} ({metrics['missed_fall_rate']:.2%}) | "
                f"{metrics['false_positives']}/{metrics['non_falls']} "
                f"({metrics['false_positive_rate']:.2%}) | "
                f"{'PASS' if result['target_checks'][split][version]['all_passed'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "Each recording uses its maximum valid window score; no threshold was selected here.",
            "Training recordings were checked for conversion equivalence, not independent performance.",
            "The saved threshold was previously chosen after reviewing validation and test results.",
            "These are repeat benchmark results; Pi performance and new audio remain separate checks.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_export(
    selection_path: Path = DEFAULT_SELECTION,
    model_path: Path = DEFAULT_OUTPUT,
    report_dir: Path = DEFAULT_REPORT,
) -> bool:
    selection_path = Path(selection_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    selection, source_path = _load_selection(selection_path)
    manifest = (ML_ROOT / selection["manifest_path"]).resolve()
    if not manifest.is_relative_to(ML_ROOT):
        raise ValueError("Selected manifest must stay inside ml/")
    for input_folder in (model_path.parent, source_path.parent, manifest.parent):
        if report_dir.is_relative_to(input_folder) or input_folder.is_relative_to(report_dir):
            raise ValueError("Reports must be separate from models and input embeddings")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != selection["manifest_sha256"]:
        raise ValueError("Manifest changed since model selection")
    metadata_path, metadata, runner = _load_export(model_path, selection)
    model = tf.keras.models.load_model(source_path, compile=False)
    if model.input_shape != (None, 1024) or model.output_shape != (None, 1):
        raise ValueError("Expected the selected 1024-input, one-score Keras model")
    data = load_training_data(manifest)
    if data.provenance["yamnet_sha256"] != selection["yamnet_sha256"]:
        raise ValueError("Training/validation embeddings use a different YAMNet model")
    groups = {
        "train": data.train,
        "validation": data.validation,
        "test": load_evaluation_split(
            manifest,
            "test",
            expected_manifest_sha256=selection["manifest_sha256"],
            expected_yamnet_sha256=selection["yamnet_sha256"],
        ),
    }
    threshold = float(selection["threshold"])
    print(
        f"Comparing original Keras and edge FallClassifierRunner at fixed threshold {threshold}...",
        flush=True,
    )
    rows = []
    for split, recordings in groups.items():
        for recording in recordings:
            rows.append(_compare(recording, model, runner, threshold))
        print(f"Checked {split}: {len(recordings)} recordings", flush=True)
    metrics, target_checks = {}, {}
    for split in ("validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        labels = np.array([LABEL_IDS[row["label"]] for row in subset])
        metrics[split] = {
            version: classification_metrics(labels, [row[f"{version}_score"] for row in subset], threshold)
            for version in ("keras", "tflite")
        }
        target_checks[split] = {name: proposal_checks(value) for name, value in metrics[split].items()}
    score_failures = sum(row["window_score_failures"] for row in rows)
    window_mismatches = sum(row["window_decision_mismatches"] for row in rows)
    recording_mismatches = sum(not row["recording_decision_match"] for row in rows)
    parity_passed = score_failures == window_mismatches == recording_mismatches == 0
    targets_passed = all(
        checks["all_passed"] for group in target_checks.values() for checks in group.values()
    )
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": parity_passed and targets_passed,
        "parity_passed": parity_passed,
        "inference_runner": "e3_edge.fall_classifier.FallClassifierRunner",
        "proposal_targets_passed": targets_passed,
        "source_model_sha256": selection["model_sha256"],
        "model_sha256": metadata["model_sha256"],
        "manifest_sha256": selection["manifest_sha256"],
        "yamnet_sha256": selection["yamnet_sha256"],
        "threshold": threshold,
        "recording_score": selection["recording_score"],
        "recordings": len(rows),
        "windows": sum(row["windows"] for row in rows),
        "recordings_by_split": {name: len(records) for name, records in groups.items()},
        "max_window_error": max(row["max_window_error"] for row in rows),
        "window_score_failures": score_failures,
        "window_decision_mismatches": window_mismatches,
        "recording_decision_mismatches": recording_mismatches,
        "rtol": RTOL,
        "atol": ATOL,
        "metrics": metrics,
        "target_checks": target_checks,
        "test_used_for_threshold_selection": selection["test_used_for_threshold_selection"],
        "threshold_changed_during_verification": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _save_json(report_dir / "results.json", result)
    (report_dir / "report.md").write_text(_report(result), encoding="utf-8")
    _write_csv(report_dir / "predictions.csv", PREDICTION_COLUMNS, rows)
    metadata["dataset_parity_verified"] = parity_passed
    metadata["dataset_verification"] = {
        key: result[key]
        for key in (
            "created_utc",
            "model_sha256",
            "inference_runner",
            "manifest_sha256",
            "threshold",
            "recordings",
            "windows",
            "max_window_error",
            "window_score_failures",
            "window_decision_mismatches",
            "recording_decision_mismatches",
            "rtol",
            "atol",
            "proposal_targets_passed",
        )
    }
    _save_json(metadata_path, metadata)
    print(
        f"{'PASS' if result['passed'] else 'FAIL'}: {len(rows)} recordings; {result['windows']} windows",
        flush=True,
    )
    print(f"Max window-score difference: {result['max_window_error']:.8g}", flush=True)
    print(f"Window decision differences: {window_mismatches}; recording differences: {recording_mismatches}")
    for split in ("validation", "test"):
        m = metrics[split]["tflite"]
        print(
            f"{split}: accuracy {m['accuracy']:.2%}; missed falls {m['missed_fall_rate']:.2%}; "
            f"false alarms {m['false_positive_rate']:.2%}; "
            f"targets {'PASS' if target_checks[split]['tflite']['all_passed'] else 'FAIL'}"
        )
    print(f"Report: {report_dir / 'report.md'}")
    return result["passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        return 0 if verify_export(args.selection, args.model, args.out) else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Fall classifier verification failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
