"""Score YAMNet embeddings with the exported fall model, without TensorFlow."""

import hashlib
import json
import re
from importlib import import_module
from pathlib import Path

import numpy as np

from ..audio.yamnet import EMBEDDING_SIZE

SIGNATURE_NAME = "serving_default"
INPUT_NAME = "embedding"
OUTPUT_NAME = "fall_score"
INPUT_SHAPE = (1, EMBEDDING_SIZE)
OUTPUT_SHAPE = (1, 1)


class FallClassifierRunner:
    """Load one exported model/settings pair and score successive audio frames.

    The companion JSON defaults to the model path with its suffix changed to
    .json. Both files are produced by ml/src/fall_classifier/export.py. Paths
    are supplied by the caller; this runner never downloads files or reads audio.

    predict(frame.embedding) accepts a finite float32 vector shaped (1024,)
    and returns a Python float in [0, 1]. It applies no additional scaling.
    The read-only threshold comes from the settings, rather than a hardcoded
    cutoff. Timing, combining windows, risk decisions and alerts belong to the
    caller/event manager. A score is not necessarily a calibrated probability.

    Reuse an instance sequentially; concurrent calls on it are not supported.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        num_threads: int = 1,
    ) -> None:
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        self.model_path = Path(model_path).expanduser()
        self.metadata_path = (
            Path(metadata_path).expanduser()
            if metadata_path is not None
            else self.model_path.with_suffix(".json")
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Fall classifier model not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Fall classifier settings not found: {self.metadata_path}")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int:
            raise ValueError("Fall classifier settings must be a version-1 JSON object")
        if metadata["schema_version"] != 1 or metadata.get("model_file") != self.model_path.name:
            raise ValueError("Fall classifier settings do not describe this model file")
        threshold = metadata.get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Fall threshold must be finite and between 0 and 1")
        if (
            metadata.get("decision_rule") != "score >= threshold"
            or metadata.get("recording_score") != "maximum valid window score"
            or metadata.get("label_ids") != {"non_fall": 0, "fall": 1}
            or metadata.get("signature_name") != SIGNATURE_NAME
        ):
            raise ValueError("Fall classifier settings use an unsupported score/decision contract")
        for key, name, shape in (
            ("input", INPUT_NAME, INPUT_SHAPE),
            ("output", OUTPUT_NAME, OUTPUT_SHAPE),
        ):
            if metadata.get(key) != {"name": name, "shape": list(shape), "dtype": "float32"}:
                raise ValueError(f"Fall classifier {key} must be float32 with shape {shape}")
        yamnet_sha256 = metadata.get("yamnet_sha256")
        if not isinstance(yamnet_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", yamnet_sha256) is None:
            raise ValueError("Fall classifier settings need a valid YAMNet model fingerprint")
        model_bytes = self.model_path.read_bytes()
        fingerprint = hashlib.sha256(model_bytes).hexdigest()
        if fingerprint != metadata.get("model_sha256"):
            raise ValueError("Fall classifier model fingerprint does not match its settings")
        self._threshold = float(threshold)
        self._model_sha256 = fingerprint
        self._yamnet_sha256 = yamnet_sha256

        # Match YAMNetRunner: the optional runtime is needed only when constructing
        # a runner. The edge package does not import TensorFlow or laptop code.
        try:
            litert = import_module("ai_edge_litert.interpreter")
        except ModuleNotFoundError as error:
            if error.name not in {"ai_edge_litert", "ai_edge_litert.interpreter"}:
                raise
            raise ImportError("Fall classification requires the ai-edge-litert dependency") from error
        self._interpreter = litert.Interpreter(
            model_content=model_bytes,
            num_threads=num_threads,
            experimental_op_resolver_type=litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self._interpreter.allocate_tensors()
        signature = self._interpreter.get_signature_list().get(SIGNATURE_NAME, {})
        if signature.get("inputs") != [INPUT_NAME] or signature.get("outputs") != [OUTPUT_NAME]:
            raise ValueError("Fall model needs a named embedding input and fall_score output")
        self._runner = self._interpreter.get_signature_runner(SIGNATURE_NAME)
        for details, name, shape in (
            (self._runner.get_input_details(), INPUT_NAME, INPUT_SHAPE),
            (self._runner.get_output_details(), OUTPUT_NAME, OUTPUT_SHAPE),
        ):
            if details[name]["dtype"] != np.float32 or tuple(details[name]["shape"]) != shape:
                raise ValueError(f"Fall model {name} must be float32 with shape {shape}")

    @property
    def threshold(self) -> float:
        """The selected cutoff, available for the caller's decision-making code."""
        return self._threshold

    @property
    def model_sha256(self) -> str:
        """Fingerprint of the actual model bytes loaded by this runner."""
        return self._model_sha256

    @property
    def yamnet_sha256(self) -> str:
        """Expected upstream YAMNet identity, for checking during later integration."""
        return self._yamnet_sha256

    def predict(self, embedding: np.ndarray) -> float:
        """Return one fall score without modifying the embedding or making an alert."""
        values = np.asarray(embedding)
        if values.dtype != np.float32 or values.shape != (EMBEDDING_SIZE,):
            raise ValueError(f"Fall classifier needs a float32 embedding shaped ({EMBEDDING_SIZE},)")
        if not np.isfinite(values).all():
            raise ValueError("Fall classifier embedding must contain only finite values")
        batch = np.ascontiguousarray(values.reshape(INPUT_SHAPE))
        outputs = self._runner(embedding=batch)
        if set(outputs) != {OUTPUT_NAME}:
            raise ValueError("Fall classifier must return exactly one fall_score output")
        score = np.asarray(outputs[OUTPUT_NAME])
        if (
            score.dtype != np.float32
            or score.shape != OUTPUT_SHAPE
            or not np.isfinite(score).all()
            or np.any((score < 0) | (score > 1))
        ):
            raise ValueError("Fall classifier must return one finite float32 score between 0 and 1")
        return float(score[0, 0])
