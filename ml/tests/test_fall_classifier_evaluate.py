"""Check validation-only threshold selection and reports with simulated scores.

Requires TensorFlow to import the laptop evaluator. No training, SAFE/YAMNet
downloads, real model weights, or local MLflow stores are used.
"""

import csv
import hashlib
import json
from importlib import import_module

import numpy as np
import pytest

from src.fall_classifier.dataset import LABEL_IDS, Recording

pytest.importorskip("tensorflow")
evaluation = import_module("src.fall_classifier.evaluate")


def test_metrics_count_missed_falls_false_alarms_and_equal_threshold():
    result = evaluation.classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.3, 0.5], 0.5)

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["false_positive_rate"] == 0.5


def test_no_fall_predictions_report_zero_precision_and_f1():
    result = evaluation.classification_metrics([0, 1], [0.2, 0.3], 1.0)

    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert result["false_negatives"] == 1 and result["false_positives"] == 0


def test_threshold_uses_best_validation_f1_with_deterministic_ties():
    # F1 ties between detecting one fall without false alarms and both with two false alarms.
    threshold, sweep = evaluation.select_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6])

    assert threshold == 0.6  # Prefer higher recall, then the highest equivalent threshold.
    assert evaluation.classification_metrics([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6], threshold)["recall"] == 1
    assert {row["threshold"] for row in sweep} == {0, 0.6, 0.7, 0.8, 0.9, 1}


def test_threshold_ties_with_same_recall_prefer_the_highest_candidate():
    threshold, _ = evaluation.select_threshold([1, 0], [0.2, 0.9])

    assert threshold == 0.2


@pytest.mark.parametrize(("labels", "scores"), [
    ([], []), ([0, 1], [0.1]), ([0, 2], [0.1, 0.9]),
    ([0, 1], [float("nan"), 0.9]), ([0, 1], [-0.1, 0.9]), ([0, 1], [0.1, 1.1]),
])
def test_rejects_invalid_labels_and_scores(labels, scores):
    with pytest.raises(ValueError):
        evaluation.select_threshold(labels, scores)


def test_threshold_selection_requires_both_validation_labels():
    with pytest.raises(ValueError, match="both validation labels"):
        evaluation.select_threshold([1, 1], [0.1, 0.9])


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    run = tmp_path / "models" / "run-example"
    run.mkdir(parents=True)
    (run / "window_classifier.keras").write_bytes(b"pretend saved window classifier")
    manifest = tmp_path / "features" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("path,label,split,features_path,windows\n")
    metadata = {
        "label_ids": LABEL_IDS, "data": {
            "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "yamnet_sha256": "a" * 64,
        },
    }
    (run / "run.json").write_text(json.dumps(metadata))
    output = tmp_path / "reports"
    loaded = []

    def recording(name, label, split, scores):
        embeddings = np.zeros((len(scores), 1024), dtype=np.float32)
        embeddings[:, 0] = scores
        return Recording(name, label, split, embeddings)

    examples = {
        "validation": (
            recording("val-fall-a.wav", "fall", "validation", [0.9, 0.1]),
            recording("val-nonfall-a.wav", "non_fall", "validation", [0.8, 0.1]),
            recording("val-fall-b.wav", "fall", "validation", [0.7, 0.2]),
            recording("val-nonfall-b.wav", "non_fall", "validation", [0.1, 0.2]),
        ),
        "test": (
            recording("test-fall.wav", "fall", "test", [0.1, 0.05]),
            recording("test-nonfall.wav", "non_fall", "test", [0.9, 0.1]),
        ),
    }

    class FakeModel:
        input_shape = (None, 1024)
        output_shape = (None, 1)

        def __call__(self, embeddings, *, training):
            assert training is False
            return embeddings[:, :1]

    def loader(path, split, **settings):
        assert path == manifest
        assert settings["expected_manifest_sha256"] == metadata["data"]["manifest_sha256"]
        assert settings["expected_yamnet_sha256"] == "a" * 64
        if split == "test":
            assert (output / "threshold.json").is_file()
            frozen = json.loads((output / "threshold.json").read_text())
            np.testing.assert_allclose(frozen["threshold"], 0.7)
        loaded.append(split)
        return examples[split]

    monkeypatch.setattr(evaluation.tf.keras.models, "load_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(evaluation, "load_evaluation_split", loader)
    return run, output, loaded


def test_evaluation_freezes_validation_threshold_before_test_access_and_writes_reports(saved_run):
    run, output, loaded = saved_run
    original_metadata = (run / "run.json").read_bytes()

    result_path = evaluation.evaluate(run, output_dir=output)

    assert result_path == output and loaded == ["validation", "test"]
    results = json.loads((output / "results.json").read_text())
    np.testing.assert_allclose(results["threshold"], 0.7)
    # Both test examples fail at this threshold; test-optimal tuning would change it.
    assert results["test"]["false_negatives"] == results["test"]["false_positives"] == 1
    assert results["test"]["f1"] == 0 and results["test_used_for_threshold_selection"] is False
    assert results["validation"]["f1"] == 0.8
    assert (run / "run.json").read_bytes() == original_metadata
    with (output / "predictions.csv").open(newline="") as source:
        predictions = list(csv.DictReader(source))
    with (output / "misclassified.csv").open(newline="") as source:
        mistakes = list(csv.DictReader(source))
    assert len(predictions) == 6 and len(mistakes) == 3
    assert {row["error"] for row in mistakes} == {"missed_fall", "false_alarm"}
    assert (output / "validation_thresholds.csv").is_file()
    assert "not household alerts per day" in (output / "report.md").read_text()


def test_rerun_reuses_frozen_threshold_without_selecting_again(saved_run, monkeypatch):
    run, output, loaded = saved_run
    evaluation.evaluate(run, output_dir=output)
    threshold_bytes = (output / "threshold.json").read_bytes()

    def unexpected_selection(*args):
        pytest.fail("A rerun must reuse the frozen threshold")

    monkeypatch.setattr(evaluation, "select_threshold", unexpected_selection)
    evaluation.evaluate(run, output_dir=output)

    assert loaded == ["validation", "test", "validation", "test"]
    assert (output / "threshold.json").read_bytes() == threshold_bytes


def test_changed_model_rejects_saved_threshold_before_loading_test(saved_run):
    run, output, loaded = saved_run
    evaluation.evaluate(run, output_dir=output)
    loaded.clear()
    (run / "window_classifier.keras").write_bytes(b"different model")

    with pytest.raises(ValueError, match="different model"):
        evaluation.evaluate(run, output_dir=output)

    assert loaded == []


def test_failed_test_loading_preserves_the_selected_threshold(saved_run, monkeypatch):
    run, output, loaded = saved_run
    original = evaluation.load_evaluation_split

    def failed_test(path, split, **settings):
        if split == "test":
            raise ValueError("missing test archive")
        return original(path, split, **settings)

    monkeypatch.setattr(evaluation, "load_evaluation_split", failed_test)
    with pytest.raises(ValueError, match="missing test archive"):
        evaluation.evaluate(run, output_dir=output)

    assert loaded == ["validation"]
    assert (output / "threshold.json").is_file() and not (output / "results.json").exists()
