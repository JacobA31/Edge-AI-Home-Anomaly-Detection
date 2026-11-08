"""Train the first fall classifier on saved edge YAMNet embeddings.

Run from ml/, with requirements installed:
    python -m src.fall_classifier.train

Uses 665 training / 95 validation recordings for the default SAFE split. Test
archives are not opened. The maximum window score is compared with a recording
label. Validation loss selects the best epoch; early stopping restores those
weights before saving the standalone window scorer as window_classifier.keras.

Each run gets its own folder beneath models/fall_classifier/ with the model,
history.csv, run.json and a manifest snapshot. Local MLflow tracking uses
ml/mlflow.db and ml/mlruns/ (both Git-ignored); no server or account is required.
No YAMNet execution/download, final test assessment, threshold tuning, TFLite
conversion, or Pi deployment happens here. Accuracy uses 0.5 only as a training
diagnostic, not as a recommended live alert threshold.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import tensorflow as tf
from mlflow.tracking import MlflowClient

from .dataset import LABEL_IDS, load_training_data, pack_recordings
from .model import build_recording_model, build_window_model

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ML_ROOT / "data" / "processed" / "safe_classifier" / "manifest.csv"
DEFAULT_OUTPUT = ML_ROOT / "models" / "fall_classifier"
TRACKING_URI = f"sqlite:///{ML_ROOT / 'mlflow.db'}"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    learning_rate: float = 0.001
    seed: int = 42

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")


def _batches(recordings, config: TrainingConfig, *, shuffle: bool) -> tf.data.Dataset:
    inputs, labels = pack_recordings(recordings)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        # Shuffle whole recordings; their windows and single target stay together.
        dataset = dataset.shuffle(len(recordings), seed=config.seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(config.batch_size)
    options = tf.data.Options()
    # This tiny in-memory dataset needs no large background worker pool.
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).prefetch(1)


def _log_run(directory: Path, summary: dict, history: dict) -> str:
    """Record this experiment locally, using already-declared MLflow dependencies."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    name = "safe-fall-classifier"
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(ML_ROOT / "mlruns" / name).as_uri(),
    )
    run = client.create_run(experiment_id, tags={"mlflow.runName": directory.name, "output": str(directory)})
    run_id = run.info.run_id
    try:
        for key, value in summary["settings"].items():
            client.log_param(run_id, key, value)
        for key in ("manifest_sha256", "yamnet_sha256"):
            client.log_param(run_id, key, summary["data"][key])
        client.log_param(run_id, "pooling", "max_valid_window_score")
        for key, values in history.items():
            for epoch, value in enumerate(values, start=1):
                client.log_metric(run_id, key, float(value), step=epoch)
        client.log_metric(run_id, "best_epoch", summary["best_epoch"])
        for key, value in summary["validation_metrics"].items():
            client.log_metric(run_id, f"restored_validation_{key}", value)
        for name in ("window_classifier.keras", "history.csv", "run.json", "manifest.csv"):
            client.log_artifact(run_id, str(directory / name))
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id, status="FINISHED")
    return run_id


def train(
    manifest_path: Path = DEFAULT_MANIFEST, output_dir: Path = DEFAULT_OUTPUT,
    *, config: TrainingConfig | None = None,
) -> Path:
    """Train, select by validation loss, and save a new run without overwriting old runs."""
    if config is None:
        config = TrainingConfig()
    config.validate()
    data = load_training_data(manifest_path)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    # No validation/test data enters the model's feature-scaling statistics.
    training_windows = np.concatenate([recording.embeddings for recording in data.train])
    window_model = build_window_model(training_windows)
    recording_model = build_recording_model(window_model)
    recording_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)],
        jit_compile=False,
    )
    training = _batches(data.train, config, shuffle=True)
    validation = _batches(data.validation, config, shuffle=False)
    print(f"Training: {len(data.train)} recordings; validation: {len(data.validation)} recordings", flush=True)
    print("Test embeddings stay unopened. One target per recording; maximum valid window score.", flush=True)
    history = recording_model.fit(
        training, validation_data=validation, epochs=config.epochs, verbose=2,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", min_delta=0, patience=config.patience,
                restore_best_weights=True, verbose=1,
            ),
        ],
    ).history
    if not history.get("val_loss") or any(not np.isfinite(values).all() for values in history.values()):
        raise ValueError("Training produced non-finite losses/metrics; no model was saved")
    if any(not np.isfinite(weight.numpy()).all() for weight in window_model.weights):
        raise ValueError("Training produced non-finite model weights; no model was saved")
    metrics = recording_model.evaluate(validation, return_dict=True, verbose=0)
    if any(not np.isfinite(value) for value in metrics.values()):
        raise ValueError("Restored validation metrics are non-finite; no model was saved")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    summary = {
        "created_utc": datetime.now(UTC).isoformat(), "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__, "settings": asdict(config), "data": data.provenance,
        "label_ids": LABEL_IDS, "input_shape": [None, 1024], "output_shape": [None, 1],
        "architecture": {"hidden_units": 64, "dropout": 0.2, "l2": 1e-4, "variance_floor": 1e-6},
        "training_objective": "recording-level binary crossentropy of maximum valid window score",
        "selection": "minimum validation loss (including regularization); restored best weights",
        "best_epoch": best_epoch, "epochs_completed": len(history["val_loss"]),
        "validation_metrics": {key: float(value) for key, value in metrics.items()},
        "diagnostic_accuracy_threshold": 0.5, "deployment_threshold": None,
        "test_evaluated": False, "tracking_uri": TRACKING_URI,
        "training_sources": [recording.source_path for recording in data.train],
        "validation_sources": [recording.source_path for recording in data.validation],
        "limitations": (
            "Weak recording labels do not establish exact impact times or calibrated window scores."
        ),
    }
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    destination = output_dir / run_name
    with TemporaryDirectory(dir=output_dir, prefix=".training-") as temporary:
        staging = Path(temporary) / "completed"
        staging.mkdir()
        # The shared scorer now has the best recording model's restored weights.
        # It contains only standard Keras layers and includes its own scaling.
        window_model.save(staging / "window_classifier.keras")
        with (staging / "history.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["epoch", *history])
            for index in range(len(history["val_loss"])):
                writer.writerow([index + 1, *(history[key][index] for key in history)])
        (staging / "run.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        (staging / "manifest.csv").write_bytes(Path(data.provenance["manifest"]).read_bytes())
        staging.replace(destination)
    print(f"Saved best window classifier and training results to {destination}", flush=True)
    print(f"Best epoch: {best_epoch}; validation accuracy at 0.5: {metrics['accuracy']:.3f}", flush=True)
    try:
        run_id = _log_run(destination, summary, history)
    except Exception as error:
        raise RuntimeError(f"Model saved to {destination}, but local MLflow logging failed: {error}") from error
    print(f"Local MLflow run: {run_id} ({TRACKING_URI})", flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Parent folder for new training runs",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = TrainingConfig(args.epochs, args.batch_size, args.patience, args.learning_rate, args.seed)
    try:
        train(args.manifest, args.output_dir, config=config)
    except (OSError, ValueError, RuntimeError, ImportError, csv.Error) as error:
        parser.exit(1, f"Fall classifier training failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
