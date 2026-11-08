"""Run the exported YAMNet model locally with LiteRT, without TensorFlow or downloads."""

from importlib import import_module
from pathlib import Path

import numpy as np

from .features import FRAME_SECONDS, SAMPLE_RATE

EMBEDDING_SIZE = 1024
CLASS_COUNT = 521
# A 0.96 s spectrogram patch needs another 15 ms of waveform for its last STFT window.
# This matches the fixed input exported by ml/src/export_yamnet.py.
WINDOW_SAMPLES = round((FRAME_SECONDS + 0.025 - 0.010) * SAMPLE_RATE)
SIGNATURE_NAME = "serving_default"
OUTPUT_SHAPES = {"embeddings": (1, EMBEDDING_SIZE), "scores": (1, CLASS_COUNT)}


def _check_tensor(details: dict, name: str, shape: tuple[int, ...]) -> None:
    if details["dtype"] != np.float32 or tuple(details["shape"]) != shape:
        raise ValueError(f"YAMNet {name} must be float32 with shape {shape}")


def _output_vector(value, name: str) -> np.ndarray:
    output = np.asarray(value)
    _check_tensor({"dtype": output.dtype, "shape": output.shape}, name, OUTPUT_SHAPES[name])
    if not np.isfinite(output).all():
        raise ValueError(f"YAMNet {name} contain non-finite values")
    if name == "scores" and np.any((output < 0) | (output > 1)):
        raise ValueError("YAMNet class scores must be between 0 and 1")
    # Each call represents one frame. Give the caller independent vectors that
    # remain valid when the next audio window is processed.
    return output[0].copy()


class YAMNetRunner:
    """Load once, then extract features from successive prepared audio windows.

    The local .tflite must be produced by ml/src/export_yamnet.py. Each call takes
    exactly 15,600 mono float32 samples (0.975 s at 16 kHz) in [-1, 1] and returns
    (embedding[1024], scores[521]). The scores retain AudioSet's class order.

    The caller handles capture, preprocessing, overlapping windows, padding at
    the end of a recording, and timestamps. Reuse this runner sequentially;
    concurrent calls from multiple threads are not supported.
    """

    def __init__(self, model_path: str | Path, *, num_threads: int = 1) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YAMNet model file not found: {self.model_path}")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")

        # Load LiteRT only when a runner is constructed. Edge's ordinary tests
        # and audio preparation do not require the optional model runtime.
        try:
            litert = import_module("ai_edge_litert.interpreter")
        except ModuleNotFoundError as error:
            if error.name not in {"ai_edge_litert", "ai_edge_litert.interpreter"}:
                raise
            raise ImportError(
                "YAMNet requires LiteRT; install it with: python -m pip install ai-edge-litert",
            ) from error

        self._interpreter = litert.Interpreter(
            model_path=str(self.model_path), num_threads=num_threads,
            experimental_op_resolver_type=litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self._interpreter.allocate_tensors()
        signature = self._interpreter.get_signature_list().get(SIGNATURE_NAME, {})
        if signature.get("inputs") != ["waveform"] or set(signature.get("outputs", [])) != set(OUTPUT_SHAPES):
            raise ValueError("YAMNet needs a waveform input and named embeddings/scores outputs")

        self._runner = self._interpreter.get_signature_runner(SIGNATURE_NAME)
        _check_tensor(self._runner.get_input_details()["waveform"], "waveform", (WINDOW_SAMPLES,))
        outputs = self._runner.get_output_details()
        for name, shape in OUTPUT_SHAPES.items():
            _check_tensor(outputs[name], name, shape)

    def extract_features(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one embedding and its class scores from one prepared window."""
        audio = np.asarray(waveform)
        if audio.dtype != np.float32 or audio.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"YAMNet needs a mono float32 window of exactly {WINDOW_SAMPLES} samples")
        if not np.isfinite(audio).all() or np.any(np.abs(audio) > 1):
            raise ValueError("YAMNet needs finite audio samples in [-1, 1]")

        output = self._runner(waveform=np.ascontiguousarray(audio))
        if set(output) != set(OUTPUT_SHAPES):
            raise ValueError("YAMNet must return embeddings and scores")
        return (
            _output_vector(output["embeddings"], "embeddings"),
            _output_vector(output["scores"], "scores"),
        )
