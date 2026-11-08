"""Choose a validation-only threshold, then assess the saved SAFE classifier.

Run from ml/ (pass a training RUN folder, not just the .keras file):
    python -m src.fall_classifier.evaluate models/fall_classifier/run-...

Defaults to the manifest recorded in run.json. --manifest may relocate that
unchanged dataset; its hash, split assignments and YAMNet fingerprint must match
training. No raw audio, training archives, YAMNet execution, or retraining is needed.

Score a recording by its maximum window score. A score >= threshold means fall.
Choose the threshold with greatest validation F1; ties prefer greater recall,
then the highest threshold. Candidates are unique validation scores plus 0 and 1.
Undefined precision/F1 is reported as zero. Test scores never select a threshold.

threshold.json is written BEFORE test archives are loaded. Reruns in the same
report folder reuse it after checking model/data hashes. Report files default to
reports/fall_classifier/<run name>/; --out selects another report folder.
Training artifacts are unchanged. This assesses labelled SAFE recordings, not
continuous-home false alerts/day, exact impact timing, or calibrated probabilities.
"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import tensorflow as tf

from .dataset import LABEL_IDS, Recording, load_evaluation_split

ML_ROOT = Path(__file__).resolve().parents[2]
SELECTION_RULE = "validation_f1_then_recall_then_highest_threshold_v1"
PREDICTION_COLUMNS = ["path", "split", "label", "windows", "score", "predicted_label", "correct", "error"]


def _checked_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("Expected equally sized, nonempty label and score vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Labels must be 0/1 and scores must be finite values in [0, 1]")
    return labels.astype(np.int64), scores


def classification_metrics(labels, scores, threshold: float) -> dict:
    """Recording-level confusion counts and rates, treating fall as positive."""
    labels, scores = _checked_scores(labels, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1")
    predicted = scores >= threshold
    positive = labels == 1
    tp, fn = int(np.sum(predicted & positive)), int(np.sum(~predicted & positive))
    fp, tn = int(np.sum(predicted & ~positive)), int(np.sum(~predicted & ~positive))
    return {
        "recordings": len(labels), "falls": tp + fn, "non_falls": tn + fp,
        "true_positives": tp, "false_negatives": fn, "false_positives": fp, "true_negatives": tn,
        "accuracy": (tp + tn) / len(labels), "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (tn + fp) if tn + fp else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_threshold(labels, scores) -> tuple[float, list[dict]]:
    """Search validation recording scores, never window labels or test results."""
    labels, scores = _checked_scores(labels, scores)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both validation labels")
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    sweep = []
    for threshold in candidates:
        metrics = classification_metrics(labels, scores, float(threshold))
        sweep.append({"threshold": float(threshold), **{
            key: metrics[key] for key in ("f1", "precision", "recall", "false_positive_rate", "accuracy")
        }})
    best = max(sweep, key=lambda row: (row["f1"], row["recall"], row["threshold"]))
    return best["threshold"], sweep


def _score_recordings(model, recordings: tuple[Recording, ...]) -> list[dict]:
    rows = []
    for recording in recordings:
        scores = np.asarray(model(recording.embeddings, training=False))
        if scores.shape != (len(recording.embeddings), 1):
            raise ValueError(f"Unexpected window scores for {recording.source_path}: {scores.shape}")
        _checked_scores(np.zeros(len(scores)), scores[:, 0])
        rows.append({
            "path": recording.source_path, "split": recording.split, "label": recording.label,
            "windows": len(scores), "score": float(scores.max()),
        })
    return rows


def _vectors(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([LABEL_IDS[row["label"]] for row in rows]), np.array([row["score"] for row in rows])


def _predictions(rows: list[dict], threshold: float) -> list[dict]:
    results = []
    for row in rows:
        label = "fall" if row["score"] >= threshold else "non_fall"
        correct = label == row["label"]
        error = "" if correct else ("missed_fall" if row["label"] == "fall" else "false_alarm")
        results.append({**row, "predicted_label": label, "correct": correct, "error": error})
    return results


def _save_json(path: Path, payload: dict) -> None:
    """Write a complete JSON file before replacing an earlier version."""
    with NamedTemporaryFile(dir=path.parent, suffix=".json", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        staged.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _report(results: dict) -> str:
    lines = [
        "# SAFE fall-classifier evaluation", "",
        f"Training run: `{results['run']}`", f"Window model SHA-256: `{results['model_sha256']}`", "",
        f"Fall if maximum window score **>= {results['threshold']!r}**.",
        "Threshold chosen on validation F1, then recall, then highest threshold; frozen before test access.",
        "", "| Set | Recordings | Accuracy | Precision | Recall | F1 | Non-falls flagged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        m = results[split]
        lines.append(
            f"| {split} | {m['recordings']} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} | {m['false_positives']}/{m['non_falls']} |",
        )
    for split in ("validation", "test"):
        m = results[split]
        lines.extend([
            "", f"## {split.capitalize()} confusion matrix", "",
            "| Actual / predicted | Non-fall | Fall |", "|---|---:|---:|",
            f"| Non-fall | {m['true_negatives']} | {m['false_positives']} |",
            f"| Fall | {m['false_negatives']} | {m['true_positives']} |",
        ])
    lines.extend([
        "", "Metrics describe whole SAFE recordings, not individual labelled impact windows.",
        "Precision is zero when nothing is predicted as fall. Scores are not calibrated probabilities.",
        "The non-fall error rate is per recording; it is not household alerts per day.",
        "Continuous-audio false alerts, timing, and Pi behaviour still need separate evaluation.",
        "Use the held-out test results for assessment, not to choose a new threshold or model.", "",
    ])
    return "\n".join(lines)


def evaluate(run_dir: Path, *, manifest_path: Path | None = None, output_dir: Path | None = None) -> Path:
    """Freeze a validation-only threshold before loading the held-out test split."""
    run_dir = Path(run_dir).expanduser().resolve()
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if summary.get("label_ids") != LABEL_IDS:
        raise ValueError("Training run must use non_fall=0 and fall=1")
    data = summary.get("data", {})
    if not all(data.get(key) for key in ("manifest", "manifest_sha256", "yamnet_sha256")):
        raise ValueError("Training run is missing dataset provenance")
    manifest = Path(manifest_path or data["manifest"]).expanduser().resolve()
    model_path = run_dir / "window_classifier.keras"
    with model_path.open("rb") as source:
        model_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    output = (
        Path(output_dir) if output_dir is not None
        else ML_ROOT / "reports" / "fall_classifier" / run_dir.name
    )
    output = output.expanduser().resolve()
    if (
        output.is_relative_to(run_dir) or run_dir.is_relative_to(output)
        or output.is_relative_to(manifest.parent) or manifest.is_relative_to(output)
    ):
        raise ValueError("Report folder must be separate from training artifacts and source features")
    provenance = {
        "model_sha256": model_sha256, "manifest_sha256": data["manifest_sha256"],
        "yamnet_sha256": data["yamnet_sha256"], "selection_rule": SELECTION_RULE,
        "comparison": "score >= threshold",
    }
    threshold_file = output / "threshold.json"
    frozen = None
    if threshold_file.exists():
        frozen = json.loads(threshold_file.read_text(encoding="utf-8"))
        if any(frozen.get(key) != value for key, value in provenance.items()):
            raise ValueError("Saved threshold belongs to a different model/dataset/selection rule")
        if not isinstance(frozen.get("threshold"), (int, float)) or not 0 <= frozen["threshold"] <= 1:
            raise ValueError("Saved threshold is invalid")
        if not isinstance(frozen.get("validation_sweep"), list) or not frozen["validation_sweep"]:
            raise ValueError("Saved threshold is missing its validation selection results")
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.input_shape != (None, 1024) or model.output_shape != (None, 1):
        raise ValueError("Expected the saved window classifier: 1024 inputs and one score")
    loader_settings = {
        "expected_manifest_sha256": data["manifest_sha256"], "expected_yamnet_sha256": data["yamnet_sha256"],
    }
    validation_rows = _score_recordings(
        model, load_evaluation_split(manifest, "validation", **loader_settings),
    )
    labels, scores = _vectors(validation_rows)
    output.mkdir(parents=True, exist_ok=True)
    if frozen is None:
        threshold, sweep = select_threshold(labels, scores)
        frozen = {
            **provenance, "threshold": threshold, "created_utc": datetime.now(UTC).isoformat(),
            "validation_metrics": classification_metrics(labels, scores, threshold), "validation_sweep": sweep,
            "scope": "SAFE recording-level baseline; continuous-audio alert threshold remains unvalidated",
        }
        _save_json(threshold_file, frozen)
        print(f"Selected and saved validation-only threshold: {threshold!r}", flush=True)
    else:
        print(f"Reusing saved validation-only threshold: {frozen['threshold']!r}", flush=True)
    threshold = float(frozen["threshold"])
    # No test archive is opened until the validation threshold is fixed on disk.
    test_rows = _score_recordings(model, load_evaluation_split(manifest, "test", **loader_settings))
    results = {
        **provenance, "created_utc": datetime.now(UTC).isoformat(), "run": str(run_dir),
        "manifest": str(manifest), "threshold": threshold,
        "validation": classification_metrics(*_vectors(validation_rows), threshold),
        "test": classification_metrics(*_vectors(test_rows), threshold),
        "test_used_for_threshold_selection": False,
        "confusion_matrix_order": {"rows": ["non_fall", "fall"], "columns": ["non_fall", "fall"]},
    }
    predictions = _predictions(validation_rows + test_rows, threshold)
    with TemporaryDirectory(dir=output, prefix=".evaluation-") as temporary:
        staging = Path(temporary)
        _save_json(staging / "results.json", results)
        (staging / "report.md").write_text(_report(results), encoding="utf-8")
        _write_csv(staging / "predictions.csv", PREDICTION_COLUMNS, predictions)
        _write_csv(
            staging / "misclassified.csv", PREDICTION_COLUMNS,
            [row for row in predictions if not row["correct"]],
        )
        sweep = frozen["validation_sweep"]
        _write_csv(staging / "validation_thresholds.csv", list(sweep[0]), sweep)
        for path in sorted(staging.iterdir()):
            path.replace(output / path.name)
    for split in ("validation", "test"):
        m = results[split]
        print(
            f"{split}: {m['recordings']} recordings; accuracy {m['accuracy']:.3f}; "
            f"precision {m['precision']:.3f}; recall {m['recall']:.3f}; F1 {m['f1']:.3f}; "
            f"missed falls {m['false_negatives']}; false alarms {m['false_positives']}",
        )
    print(f"Report: {output / 'report.md'}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run", type=Path, help="Training run folder containing window_classifier.keras and run.json",
    )
    parser.add_argument(
        "--manifest", type=Path, help="Relocated, unchanged feature manifest (defaults to run.json)",
    )
    parser.add_argument("--out", type=Path, help="Report folder (defaults to reports/fall_classifier/<run name>)")
    args = parser.parse_args(argv)
    try:
        evaluate(args.run, manifest_path=args.manifest, output_dir=args.out)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, csv.Error) as error:
        parser.exit(1, f"Fall classifier evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
