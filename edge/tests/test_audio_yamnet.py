"""Runner checks without LiteRT; opt in to a real-model check with E3_YAMNET_MODEL."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from e3_edge.audio import yamnet


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    model_path = tmp_path / "yamnet.tflite"
    model_path.write_bytes(b"test-model")
    state = SimpleNamespace(
        model_path=model_path, loads=[], allocations=0, runner_loads=0, calls=[],
        signature={"inputs": ["waveform"], "outputs": ["embeddings", "scores"]},
        inputs={"waveform": {"dtype": np.float32, "shape": np.array([15600])}},
        outputs={
            "embeddings": {"dtype": np.float32, "shape": np.array([1, 1024])},
            "scores": {"dtype": np.float32, "shape": np.array([1, 521])},
        },
        result={
            "embeddings": np.ones((1, 1024), dtype=np.float32),
            "scores": np.full((1, 521), 0.25, dtype=np.float32),
        },
    )

    class SignatureRunner:
        def get_input_details(self):
            return state.inputs

        def get_output_details(self):
            return state.outputs

        def __call__(self, *, waveform):
            state.calls.append(waveform.copy())
            return state.result

    class Interpreter:
        def __init__(self, **kwargs):
            state.loads.append(kwargs)

        def allocate_tensors(self):
            state.allocations += 1

        def get_signature_list(self):
            return {"serving_default": state.signature}

        def get_signature_runner(self, name):
            assert name == "serving_default"
            state.runner_loads += 1
            return SignatureRunner()

    runtime = SimpleNamespace(
        Interpreter=Interpreter,
        OpResolverType=SimpleNamespace(BUILTIN_WITHOUT_DEFAULT_DELEGATES="builtins"),
    )

    def load_runtime(name):
        assert name == "ai_edge_litert.interpreter"
        return runtime

    monkeypatch.setattr(yamnet, "import_module", load_runtime)
    return state


def test_loads_local_model_once_for_repeated_windows(fake_runtime):
    runner = yamnet.YAMNetRunner(fake_runtime.model_path, num_threads=2)
    first = np.zeros(15600, dtype=np.float32)
    second = np.full(15600, 0.5, dtype=np.float32)

    embeddings, scores = runner.extract_features(first)
    runner.extract_features(second)

    assert len(fake_runtime.loads) == 1
    assert fake_runtime.loads[0]["model_path"] == str(fake_runtime.model_path)
    assert fake_runtime.loads[0]["num_threads"] == 2
    assert fake_runtime.allocations == fake_runtime.runner_loads == 1
    assert len(fake_runtime.calls) == 2
    np.testing.assert_array_equal(fake_runtime.calls[0], first)
    np.testing.assert_array_equal(fake_runtime.calls[1], second)
    assert embeddings.shape == (1024,) and scores.shape == (521,)
    assert embeddings.dtype == scores.dtype == np.float32
    np.testing.assert_array_equal(embeddings, np.ones(1024, dtype=np.float32))
    np.testing.assert_array_equal(scores, np.full(521, 0.25, dtype=np.float32))


def test_outputs_survive_later_calls_and_belong_to_the_caller(fake_runtime):
    runner = yamnet.YAMNetRunner(fake_runtime.model_path)
    waveform = np.zeros(15600, dtype=np.float32)
    first_embedding, first_scores = runner.extract_features(waveform)
    fake_runtime.result["embeddings"][:] = 2
    fake_runtime.result["scores"][:] = 0.75
    second_embedding, second_scores = runner.extract_features(waveform)

    assert np.all(first_embedding == 1) and np.all(first_scores == 0.25)
    assert np.all(second_embedding == 2) and np.all(second_scores == 0.75)
    second_embedding[:] = 3
    second_scores[:] = 0
    assert np.all(fake_runtime.result["embeddings"] == 2)
    assert np.all(fake_runtime.result["scores"] == 0.75)


def test_missing_model_is_reported_without_loading_a_runtime(tmp_path, monkeypatch):
    def unexpected_import(name):
        pytest.fail("A missing file should not load LiteRT or attempt a download")

    monkeypatch.setattr(yamnet, "import_module", unexpected_import)

    with pytest.raises(FileNotFoundError, match="model file not found"):
        yamnet.YAMNetRunner(tmp_path / "missing.tflite")


def test_missing_litert_has_installation_instructions(tmp_path, monkeypatch):
    model_path = tmp_path / "yamnet.tflite"
    model_path.touch()

    def missing_runtime(name):
        raise ModuleNotFoundError("No LiteRT installed", name="ai_edge_litert")

    monkeypatch.setattr(yamnet, "import_module", missing_runtime)

    with pytest.raises(ImportError, match="pip install ai-edge-litert"):
        yamnet.YAMNetRunner(model_path)


@pytest.mark.parametrize("num_threads", [0, -1, 1.5, True])
def test_rejects_invalid_thread_counts(fake_runtime, num_threads):
    with pytest.raises(ValueError, match="positive integer"):
        yamnet.YAMNetRunner(fake_runtime.model_path, num_threads=num_threads)
    assert not fake_runtime.loads


def test_rejects_a_classification_only_model(fake_runtime):
    fake_runtime.signature["outputs"] = ["scores"]

    with pytest.raises(ValueError, match="named embeddings/scores"):
        yamnet.YAMNetRunner(fake_runtime.model_path)


@pytest.mark.parametrize(("group", "name", "field", "value"), [
    ("inputs", "waveform", "shape", np.array([1, 15600])),
    ("inputs", "waveform", "dtype", np.int16),
    ("outputs", "embeddings", "shape", np.array([1, 128])),
    ("outputs", "scores", "dtype", np.int8),
])
def test_rejects_incompatible_model_tensors(fake_runtime, group, name, field, value):
    getattr(fake_runtime, group)[name][field] = value

    with pytest.raises(ValueError, match=f"{name} must be float32"):
        yamnet.YAMNetRunner(fake_runtime.model_path)


@pytest.mark.parametrize("waveform", [
    np.zeros(15600, dtype=np.int16),
    np.zeros(15600, dtype=np.float64),
    np.zeros((15600, 1), dtype=np.float32),
    np.zeros(15599, dtype=np.float32),
    np.zeros(15601, dtype=np.float32),
    np.full(15600, np.nan, dtype=np.float32),
    np.full(15600, np.inf, dtype=np.float32),
    np.full(15600, 1.1, dtype=np.float32),
    np.full(15600, -1.1, dtype=np.float32),
])
def test_rejects_invalid_audio_before_inference(fake_runtime, waveform):
    runner = yamnet.YAMNetRunner(fake_runtime.model_path)

    with pytest.raises(ValueError, match="YAMNet needs"):
        runner.extract_features(waveform)
    assert not fake_runtime.calls


@pytest.mark.parametrize(("name", "value", "message"), [
    ("embeddings", np.zeros((1, 128), dtype=np.float32), "embeddings must be float32"),
    ("scores", np.zeros((1, 521), dtype=np.float64), "scores must be float32"),
    ("embeddings", np.full((1, 1024), np.nan, dtype=np.float32), "non-finite"),
    ("scores", np.full((1, 521), -0.1, dtype=np.float32), "between 0 and 1"),
    ("scores", np.full((1, 521), 1.1, dtype=np.float32), "between 0 and 1"),
])
def test_rejects_invalid_model_results(fake_runtime, name, value, message):
    runner = yamnet.YAMNetRunner(fake_runtime.model_path)
    fake_runtime.result[name] = value

    with pytest.raises(ValueError, match=message):
        runner.extract_features(np.zeros(15600, dtype=np.float32))


@pytest.mark.skipif(
    not os.environ.get("E3_YAMNET_MODEL"),
    reason="Set E3_YAMNET_MODEL to an exported .tflite to test the real LiteRT model",
)
def test_real_exported_model_processes_repeated_audio_windows():
    runner = yamnet.YAMNetRunner(os.environ["E3_YAMNET_MODEL"])
    silence = np.zeros(15600, dtype=np.float32)
    time = np.arange(15600, dtype=np.float64) / 16000
    tone = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    silence_embedding, silence_scores = runner.extract_features(silence)
    tone_embedding, tone_scores = runner.extract_features(tone)
    repeated_embedding, repeated_scores = runner.extract_features(silence)

    for embedding, scores in ((silence_embedding, silence_scores), (tone_embedding, tone_scores)):
        assert embedding.shape == (1024,) and scores.shape == (521,)
        assert embedding.dtype == scores.dtype == np.float32
        assert np.isfinite(embedding).all() and np.isfinite(scores).all()
        assert np.all((scores >= 0) & (scores <= 1))
    assert not np.allclose(silence_embedding, tone_embedding)
    np.testing.assert_allclose(repeated_embedding, silence_embedding)
    np.testing.assert_allclose(repeated_scores, silence_scores)
