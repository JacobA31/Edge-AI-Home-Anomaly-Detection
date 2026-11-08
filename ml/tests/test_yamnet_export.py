"""Offline export checks; the real conversion test skips if its runtimes are absent."""

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

from src import export_yamnet


@pytest.fixture
def fake_runtime(monkeypatch):
    def reference(waveform):
        embeddings = np.full((1, 1024), 0.5 + waveform.mean(), dtype=np.float32)
        scores = np.full((1, 521), 0.25, dtype=np.float32)
        return scores, embeddings, None

    state = SimpleNamespace(
        inputs=[{"dtype": np.float32, "shape": np.array([15600])}],
        signature={"inputs": ["waveform"], "outputs": ["embeddings", "scores"]},
        overrides={}, calls=0,
    )

    def run(*, waveform):
        state.calls += 1
        scores, embeddings, _ = reference(waveform)
        return {"scores": scores, "embeddings": embeddings, **state.overrides}

    interpreter = SimpleNamespace(
        allocate_tensors=lambda: None,
        get_input_details=lambda: state.inputs,
        get_signature_list=lambda: {"serving_default": state.signature},
        get_signature_runner=lambda name: run,
    )
    runtime = SimpleNamespace(
        Interpreter=lambda **kwargs: interpreter,
        OpResolverType=SimpleNamespace(BUILTIN_WITHOUT_DEFAULT_DELEGATES="builtins"),
    )

    def load_runtime(name):
        assert name == "ai_edge_litert.interpreter"
        return runtime

    monkeypatch.setattr(export_yamnet, "import_module", load_runtime)
    return reference, state


def test_verification_checks_three_signals_and_both_outputs(fake_runtime):
    reference, state = fake_runtime

    export_yamnet.verify_model(b"model", reference)

    assert state.calls == 3


@pytest.mark.parametrize("inputs", [
    [],
    [{"dtype": np.int16, "shape": np.array([15600])}],
    [{"dtype": np.float32, "shape": np.array([1, 15600])}],
])
def test_verification_rejects_wrong_input_contract(fake_runtime, inputs):
    reference, state = fake_runtime
    state.inputs = inputs

    with pytest.raises(ValueError, match="waveform input"):
        export_yamnet.verify_model(b"model", reference)


def test_verification_rejects_a_scores_only_model(fake_runtime):
    reference, state = fake_runtime
    state.signature["outputs"] = ["scores"]

    with pytest.raises(ValueError, match="named embeddings/scores"):
        export_yamnet.verify_model(b"model", reference)


@pytest.mark.parametrize(("name", "value", "message"), [
    ("embeddings", np.zeros((1, 128), dtype=np.float32), "Expected embeddings"),
    ("scores", np.zeros((1, 521), dtype=np.float64), "Expected scores"),
    ("embeddings", np.full((1, 1024), np.nan, dtype=np.float32), "Non-finite"),
    ("scores", np.full((1, 521), 1.1, dtype=np.float32), "between 0 and 1"),
    ("embeddings", np.zeros((1, 1024), dtype=np.float32), "differ from YAMNet"),
    ("scores", np.full((1, 521), 0.75, dtype=np.float32), "differ from YAMNet"),
])
def test_verification_rejects_invalid_or_changed_outputs(fake_runtime, name, value, message):
    reference, state = fake_runtime
    state.overrides[name] = value

    with pytest.raises(ValueError, match=message):
        export_yamnet.verify_model(b"model", reference)


def test_cli_writes_verified_model(tmp_path, monkeypatch, fake_runtime):
    reference, state = fake_runtime
    monkeypatch.setattr(export_yamnet, "_load_yamnet_model", lambda: reference)
    monkeypatch.setattr(export_yamnet, "_convert_model", lambda model: b"converted-model")
    destination = tmp_path / "models" / "yamnet.tflite"

    assert export_yamnet.main(["--output", str(destination)]) == 0

    assert destination.read_bytes() == b"converted-model"
    assert list(destination.parent.iterdir()) == [destination]
    assert state.calls == 3


def test_failed_verification_preserves_existing_model(tmp_path, monkeypatch, fake_runtime):
    reference, state = fake_runtime
    monkeypatch.setattr(export_yamnet, "_load_yamnet_model", lambda: reference)
    monkeypatch.setattr(export_yamnet, "_convert_model", lambda model: b"bad-model")
    state.overrides["scores"] = np.full((1, 521), 0.75, dtype=np.float32)
    destination = tmp_path / "yamnet.tflite"
    destination.write_bytes(b"previous-model")

    with pytest.raises(ValueError, match="differ from YAMNet"):
        export_yamnet.export_yamnet(destination)

    assert destination.read_bytes() == b"previous-model"
    assert list(tmp_path.iterdir()) == [destination]


def test_rejects_wrong_output_extension_before_loading_model(tmp_path, monkeypatch):
    def unexpected_load():
        pytest.fail("A rejected destination should not trigger a model download")

    monkeypatch.setattr(export_yamnet, "_load_yamnet_model", unexpected_load)

    with pytest.raises(ValueError, match=".tflite extension"):
        export_yamnet.export_yamnet(tmp_path / "yamnet.npy")


def test_cli_reports_load_failure_without_writing_a_file(tmp_path, monkeypatch, capsys):
    def unavailable_model():
        raise OSError("model download unavailable")

    monkeypatch.setattr(export_yamnet, "_load_yamnet_model", unavailable_model)
    destination = tmp_path / "yamnet.tflite"

    with pytest.raises(SystemExit) as error:
        export_yamnet.main(["--output", str(destination)])

    assert error.value.code == 1
    assert "model download unavailable" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.skipif(
    find_spec("tensorflow") is None or find_spec("ai_edge_litert") is None,
    reason="Real conversion needs the optional laptop TensorFlow and LiteRT dependencies",
)
def test_real_conversion_freezes_weights_and_keeps_spectral_ops_in_litert():
    # A tiny local model exercises conversion and the spectral operations used by
    # YAMNet without fetching pretrained weights. The actual Hub export still
    # performs its own numerical checks when the user runs the export command.
    tf = import_module("tensorflow")

    class SpectralModel(tf.Module):
        def __init__(self):
            super().__init__()
            self.weights = tf.Variable(tf.linspace(0.0, 1.0, 1024), trainable=False)

        @tf.function(input_signature=[tf.TensorSpec([15600], tf.float32)])
        def __call__(self, waveform):
            spectrogram = tf.abs(tf.signal.stft(waveform, 400, 160, 512))
            level = 0.01 * tf.reduce_mean(tf.math.log(0.001 + spectrogram))
            embeddings = tf.expand_dims(self.weights + level, axis=0)
            scores = tf.fill([1, 521], tf.math.sigmoid(level))
            return scores, embeddings, spectrogram

    model = SpectralModel()
    converted = export_yamnet._convert_model(model)

    export_yamnet.verify_model(converted, model)
