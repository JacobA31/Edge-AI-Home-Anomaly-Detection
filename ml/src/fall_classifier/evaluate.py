"""Choose a validation threshold and optionally assess the saved SAFE classifier on test.

Run from ml/ (pass a training RUN folder, not just the .keras file):
    python -m src.fall_classifier.evaluate models/fall_classifier/run-...
    python -m src.fall_classifier.evaluate models/fall_classifier/run-... --validation-only

Defaults to the manifest recorded in run.json. --manifest may relocate that
unchanged dataset; its hash, split assignments and YAMNet fingerprint must match
training. No raw audio, training archives, YAMNet execution, or retraining is needed.

Score a recording by its maximum window score. A score >= threshold means fall.
New thresholds first require validation accuracy >= 90%, missed falls <= 10%,
and non-falls flagged <= 5% (proposal NF3). Among qualifying thresholds, choose
greatest F1, then recall, then highest threshold. If none qualify, report FAIL
and show the best-F1 threshold only as a diagnostic fallback. Candidates are
unique validation scores plus 0 and 1. Undefined precision/F1 is zero.
Test scores never select a threshold. Saved original F1-only thresholds remain
unchanged and are identified as legacy selections, with target checks added.

threshold.json is written BEFORE any test archives are loaded. Reruns in the
same report folder reuse it after checking model/data hashes. --validation-only
does not open test archives: it writes only validation results in a validation-only/
subfolder, preserving full-evaluation reports. Both modes share threshold.json
in the parent report folder. Report files default to reports/fall_classifier/<run name>/;
--out selects another parent report folder.
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
LEGACY_SELECTION_RULE = "validation_f1_then_recall_then_highest_threshold_v1"
SELECTION_RULE = "validation_nf3_then_f1_then_recall_then_highest_threshold_v2"
PROPOSAL_TARGETS = {"min_accuracy": 0.90, "max_missed_fall_rate": 0.10, "max_false_positive_rate": 0.05}
PREDICTION_COLUMNS = ["path", "split", "label", "windows", "score", "predicted_label", "correct", "error"]


def _checked_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("Expected equally sized, nonempty label and score vectors")
    if (
        not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
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
        "recordings": len(labels),
        "falls": tp + fn,
        "non_falls": tn + fp,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "missed_fall_rate": fn / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (tn + fp) if tn + fp else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def proposal_checks(metrics: dict) -> dict[str, bool]:
    """Apply NF3 percentages; validation checks are development checks, not final test claims."""
    tolerance = 1e-12
    checks = {
        "accuracy": metrics["accuracy"] >= PROPOSAL_TARGETS["min_accuracy"] - tolerance,
        "missed_falls": 1 - metrics["recall"] <= PROPOSAL_TARGETS["max_missed_fall_rate"] + tolerance,
        "false_alarms": (
            metrics["false_positive_rate"] <= PROPOSAL_TARGETS["max_false_positive_rate"] + tolerance
        ),
    }
    return {**{key: bool(value) for key, value in checks.items()}, "all_passed": bool(all(checks.values()))}


def select_threshold(labels, scores) -> tuple[float, list[dict]]:
    """Prefer NF3-compliant validation thresholds; otherwise return an explicitly failing baseline."""
    labels, scores = _checked_scores(labels, scores)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both validation labels")
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    sweep = []
    for threshold in candidates:
        metrics = classification_metrics(labels, scores, float(threshold))
        sweep.append(
            {
                "threshold": float(threshold),
                **{
                    key: metrics[key]
                    for key in ("f1", "precision", "recall", "false_positive_rate", "accuracy")
                },
                "meets_targets": proposal_checks(metrics)["all_passed"],
            }
        )
    qualifying = [row for row in sweep if row["meets_targets"]]
    best = max(qualifying or sweep, key=lambda row: (row["f1"], row["recall"], row["threshold"]))
    return best["threshold"], sweep


def _score_recordings(model, recordings: tuple[Recording, ...]) -> list[dict]:
    rows = []
    for recording in recordings:
        scores = np.asarray(model(recording.embeddings, training=False))
        if scores.shape != (len(recording.embeddings), 1):
            raise ValueError(f"Unexpected window scores for {recording.source_path}: {scores.shape}")
        _checked_scores(np.zeros(len(scores)), scores[:, 0])
        rows.append(
            {
                "path": recording.source_path,
                "split": recording.split,
                "label": recording.label,
                "windows": len(scores),
                "score": float(scores.max()),
            }
        )
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
    splits = tuple(split for split in ("validation", "test") if split in results)
    lines = [
        "# SAFE fall-classifier evaluation",
        "",
        f"Training run: `{results['run']}`",
        f"Window model SHA-256: `{results['model_sha256']}`",
        "",
        f"Fall if maximum window score **>= {results['threshold']!r}**.",
        results["selection_note"],
        "Validation thresholds meeting all proposal targets: "
        f"**{results['qualifying_validation_thresholds']}**.",
    ]
    if results.get("validation_only", False):
        lines.extend(["", "**Validation-only development check. Test embeddings were not opened.**"])
    else:
        lines.extend(["", "Threshold frozen before test access."])
    lines.extend(
        [
            "",
            "| Set | Recordings | Accuracy | Precision | Recall | F1 | Non-falls flagged |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in splits:
        m = results[split]
        lines.append(
            f"| {split} | {m['recordings']} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} | {m['false_positives']}/{m['non_falls']} |",
        )
    lines.extend(
        [
            "",
            "## Proposal NF3 target checks",
            "",
            "Validation checks guide development; only test results assess the held-out SAFE targets.",
            "",
            "| Set | Accuracy >= 90% | Falls missed <= 10% | Non-falls flagged <= 5% | All targets |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for split in splits:
        m, checks = results[split], results["target_checks"][split]
        accuracy_status = "PASS" if checks["accuracy"] else "FAIL"
        missed_status = "PASS" if checks["missed_falls"] else "FAIL"
        false_status = "PASS" if checks["false_alarms"] else "FAIL"
        overall = "PASS" if checks["all_passed"] else "FAIL"
        lines.append(
            f"| {split} | {m['accuracy']:.1%} {accuracy_status} | "
            f"{m['false_negatives']}/{m['falls']} ({m['missed_fall_rate']:.1%}) {missed_status} | "
            f"{m['false_positives']}/{m['non_falls']} ({m['false_positive_rate']:.1%}) "
            f"{false_status} | {overall} |",
        )
    for split in splits:
        m = results[split]
        lines.extend(
            [
                "",
                f"## {split.capitalize()} confusion matrix",
                "",
                "| Actual / predicted | Non-fall | Fall |",
                "|---|---:|---:|",
                f"| Non-fall | {m['true_negatives']} | {m['false_positives']} |",
                f"| Fall | {m['false_negatives']} | {m['true_positives']} |",
            ]
        )
    lines.extend(
        [
            "",
            "Metrics describe whole SAFE recordings, not individual labelled impact windows.",
            "Precision is zero when nothing is predicted as fall. Scores are not calibrated probabilities.",
            "The non-fall error rate is per recording; it is not household alerts per day.",
            "Continuous-audio false alerts, timing, and Pi behaviour still need separate evaluation.",
            "Use validation results to choose changes; reserve the test assessment for the selected version.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(
    run_dir: Path,
    *,
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
    validation_only: bool = False,
) -> Path:
    """Freeze a validation-only threshold; open test embeddings only in full mode."""
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
    report_root = (
        Path(output_dir) if output_dir is not None else ML_ROOT / "reports" / "fall_classifier" / run_dir.name
    )
    report_root = report_root.expanduser().resolve()
    if (
        report_root.is_relative_to(run_dir)
        or run_dir.is_relative_to(report_root)
        or report_root.is_relative_to(manifest.parent)
        or manifest.is_relative_to(report_root)
    ):
        raise ValueError("Report folder must be separate from training artifacts and source features")
    provenance = {
        "model_sha256": model_sha256,
        "manifest_sha256": data["manifest_sha256"],
        "yamnet_sha256": data["yamnet_sha256"],
        "selection_rule": SELECTION_RULE,
        "comparison": "score >= threshold",
    }
    # Both modes use one frozen threshold. Validation-only report files live
    # separately so development checks cannot replace an existing test report.
    threshold_file = report_root / "threshold.json"
    output = report_root / "validation-only" if validation_only else report_root
    frozen = None
    if threshold_file.exists():
        frozen = json.loads(threshold_file.read_text(encoding="utf-8"))
        if any(frozen.get(key) != value for key, value in provenance.items() if key != "selection_rule"):
            raise ValueError("Saved threshold belongs to a different model/dataset")
        rule = frozen.get("selection_rule")
        if rule not in {LEGACY_SELECTION_RULE, SELECTION_RULE}:
            raise ValueError("Saved threshold uses an unknown selection rule")
        if rule == SELECTION_RULE and frozen.get("proposal_targets") != PROPOSAL_TARGETS:
            raise ValueError("Saved threshold uses different proposal targets")
        # Preserve the baseline cutoff and identify its original policy honestly.
        provenance["selection_rule"] = rule
        if not isinstance(frozen.get("threshold"), (int, float)) or not 0 <= frozen["threshold"] <= 1:
            raise ValueError("Saved threshold is invalid")
        if not isinstance(frozen.get("validation_sweep"), list) or not frozen["validation_sweep"]:
            raise ValueError("Saved threshold is missing its validation selection results")
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.input_shape != (None, 1024) or model.output_shape != (None, 1):
        raise ValueError("Expected the saved window classifier: 1024 inputs and one score")
    loader_settings = {
        "expected_manifest_sha256": data["manifest_sha256"],
        "expected_yamnet_sha256": data["yamnet_sha256"],
    }
    validation_rows = _score_recordings(
        model,
        load_evaluation_split(manifest, "validation", **loader_settings),
    )
    labels, scores = _vectors(validation_rows)
    report_root.mkdir(parents=True, exist_ok=True)
    if frozen is None:
        threshold, sweep = select_threshold(labels, scores)
        frozen = {
            **provenance,
            "threshold": threshold,
            "created_utc": datetime.now(UTC).isoformat(),
            "validation_metrics": classification_metrics(labels, scores, threshold),
            "validation_sweep": sweep,
            "proposal_targets": dict(PROPOSAL_TARGETS),
            "validation_target_checks": proposal_checks(classification_metrics(labels, scores, threshold)),
            "scope": "SAFE recording-level baseline; continuous-audio alert threshold remains unvalidated",
        }
        _save_json(threshold_file, frozen)
        print(f"Selected and saved validation-only threshold: {threshold!r}", flush=True)
    else:
        print(f"Reusing saved validation-only threshold: {frozen['threshold']!r}", flush=True)
    threshold = float(frozen["threshold"])
    # Enrich legacy sweeps in the new report, without changing the saved cutoff file.
    validation_sweep = [
        {**row, "meets_targets": proposal_checks(row)["all_passed"]} for row in frozen["validation_sweep"]
    ]
    qualifying = sum(row["meets_targets"] for row in validation_sweep)
    if provenance["selection_rule"] == LEGACY_SELECTION_RULE:
        selection_note = (
            "Original F1-only baseline threshold reused unchanged; proposal checks are reported below."
        )
    elif qualifying:
        selection_note = (
            "Best validation F1 among thresholds meeting all proposal targets; ties use recall, then cutoff."
        )
    else:
        selection_note = (
            "FAIL: no validation threshold meets all proposal targets. Best-F1 fallback is diagnostic only."
        )
    print(selection_note, flush=True)
    if not qualifying:
        print("Validation target search: FAIL (no qualifying threshold).", flush=True)
    rows_by_split = {"validation": validation_rows}
    if not validation_only:
        # No test archive is opened until the validation threshold is fixed on disk.
        rows_by_split["test"] = _score_recordings(
            model,
            load_evaluation_split(manifest, "test", **loader_settings),
        )
    results = {
        **provenance,
        "created_utc": datetime.now(UTC).isoformat(),
        "run": str(run_dir),
        "manifest": str(manifest),
        "threshold": threshold,
        "threshold_file": str(threshold_file),
        **{
            split: classification_metrics(*_vectors(rows), threshold) for split, rows in rows_by_split.items()
        },
        "proposal_targets": dict(PROPOSAL_TARGETS),
        "qualifying_validation_thresholds": qualifying,
        "selection_note": selection_note,
        "validation_only": validation_only,
        "test_evaluated": not validation_only,
        "test_used_for_threshold_selection": False,
        "confusion_matrix_order": {"rows": ["non_fall", "fall"], "columns": ["non_fall", "fall"]},
    }
    results["target_checks"] = {split: proposal_checks(results[split]) for split in rows_by_split}
    predictions = _predictions([row for rows in rows_by_split.values() for row in rows], threshold)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output, prefix=".evaluation-") as temporary:
        staging = Path(temporary)
        _save_json(staging / "results.json", results)
        (staging / "report.md").write_text(_report(results), encoding="utf-8")
        _write_csv(staging / "predictions.csv", PREDICTION_COLUMNS, predictions)
        _write_csv(
            staging / "misclassified.csv",
            PREDICTION_COLUMNS,
            [row for row in predictions if not row["correct"]],
        )
        _write_csv(staging / "validation_thresholds.csv", list(validation_sweep[0]), validation_sweep)
        for path in sorted(staging.iterdir()):
            path.replace(output / path.name)
    for split in rows_by_split:
        m = results[split]
        print(
            f"{split}: {m['recordings']} recordings; accuracy {m['accuracy']:.3f}; "
            f"precision {m['precision']:.3f}; recall {m['recall']:.3f}; F1 {m['f1']:.3f}; "
            f"missed falls {m['false_negatives']}; false alarms {m['false_positives']}",
        )
        checks = results["target_checks"][split]
        print(
            f"{split} proposal targets: {'PASS' if checks['all_passed'] else 'FAIL'} "
            f"(accuracy {'PASS' if checks['accuracy'] else 'FAIL'}, "
            f"missed falls {'PASS' if checks['missed_falls'] else 'FAIL'}, "
            f"false alarms {'PASS' if checks['false_alarms'] else 'FAIL'})",
        )
    if validation_only:
        print("Validation-only: test embeddings stayed unopened.")
    print(f"Report: {output / 'report.md'}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run",
        type=Path,
        help="Training run folder containing window_classifier.keras and run.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Relocated, unchanged feature manifest (defaults to run.json)",
    )
    parser.add_argument(
        "--out", type=Path, help="Parent report folder; validation-only reports use a subfolder"
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Evaluate validation only, without loading test embeddings",
    )
    args = parser.parse_args(argv)
    try:
        evaluate(
            args.run, manifest_path=args.manifest, output_dir=args.out, validation_only=args.validation_only
        )
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, csv.Error) as error:
        parser.exit(1, f"Fall classifier evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
