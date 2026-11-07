"""Replay/comparison checks without model files, TensorFlow, LiteRT, or downloads."""

import wave
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from e3_edge.audio import pipeline as edge_pipeline

from src import check_edge_audio


@pytest.fixture
def fake_models(monkeypatch):
    ripple = np.tile([-0.125, 0.125], 24000)
    offsets = np.linspace(0.25, 0.45, 48000)
    state = SimpleNamespace(
        samples=np.column_stack([ripple + offsets, ripple + 0.5]).astype(np.float32),
        loads=[], frames=[], edge_audio=[], reference_audio=[],
        embedding_error=0.0, score_error=0.0, frame_changes={},
    )

    def outputs(waveform):
        embedding = np.full(1024, np.max(np.abs(waveform)), dtype=np.float32)
        embedding[:3] = [waveform[0], waveform[-1], waveform.mean()]
        scores = np.linspace(0.0, 1.0, 521, dtype=np.float32)
        return embedding, scores

    class FakeRunner:
        def __init__(self, model_path, *, num_threads=1):
            state.loads.append(model_path)

        def extract_features(self, waveform):
            state.edge_audio.append(waveform.copy())
            embedding, scores = outputs(waveform)
            return embedding + state.embedding_error, scores + state.score_error

    def reference(waveform):
        state.reference_audio.append(waveform.copy())
        embedding, scores = outputs(waveform)
        return embedding[None, :], scores[None, :]

    def make_pipeline(model_path):
        audio = edge_pipeline.AudioPipeline(model_path)
        original_process = audio.process_window

        def process(*args, **kwargs):
            frame = replace(original_process(*args, **kwargs), **state.frame_changes)
            state.frames.append(frame)
            return frame

        monkeypatch.setattr(audio, "process_window", process)
        return audio

    monkeypatch.setattr(edge_pipeline, "YAMNetRunner", FakeRunner)
    monkeypatch.setattr(check_edge_audio, "AudioPipeline", make_pipeline)
    monkeypatch.setattr(check_edge_audio, "load_recording", lambda path: state.samples)
    monkeypatch.setattr(check_edge_audio.yamnet, "extract_features", reference)
    return state


def test_loader_resamples_without_mixing_channels_or_removing_dc(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = np.tile([8192, 16384], (8000, 1)).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())

    loaded = check_edge_audio.load_recording(path)

    assert loaded.dtype == np.float32 and loaded.shape == (16000, 2)
    assert loaded.flags.c_contiguous
    np.testing.assert_allclose(loaded[1000:-1000, 0], 0.25, atol=1e-4)
    np.testing.assert_allclose(loaded[1000:-1000, 1], 0.5, atol=1e-4)


@pytest.mark.parametrize("filename", ["missing.wav", "not-a-wav.txt"])
def test_loader_rejects_invalid_paths_before_decoding(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    if filename.endswith(".txt"):
        path.touch()

    def unexpected_load(*args, **kwargs):
        pytest.fail("An invalid path should not reach the decoder")

    monkeypatch.setattr(check_edge_audio.librosa, "load", unexpected_load)

    with pytest.raises(ValueError, match="Expected a WAV file"):
        check_edge_audio.load_recording(path)


def test_loader_rejects_empty_audio(tmp_path, monkeypatch):
    path = tmp_path / "empty.wav"
    path.touch()
    monkeypatch.setattr(
        check_edge_audio.librosa, "load", lambda *args, **kwargs: (np.zeros(0, dtype=np.float32), 16000),
    )

    with pytest.raises(ValueError, match="nonempty"):
        check_edge_audio.load_recording(path)


@pytest.mark.parametrize(("length", "windows", "padded"), [
    (1, 1, 1), (15600, 1, 0), (15601, 2, 1), (23280, 2, 0), (48000, 6, 1),
])
def test_window_boundaries_include_only_the_needed_final_window(fake_models, length, windows, padded):
    fake_models.samples = np.linspace(-0.2, 0.2, length, dtype=np.float32)

    result = check_edge_audio.compare_recording("recording.wav")

    assert result.windows == windows and result.padded_windows == padded
    assert len(fake_models.loads) == 1
    assert len(fake_models.edge_audio) == len(fake_models.reference_audio) == windows
    assert result.duration_seconds == length / 16000
    for edge, reference in zip(fake_models.edge_audio, fake_models.reference_audio, strict=True):
        assert edge.shape == (15600,) and edge.dtype == np.float32
        np.testing.assert_array_equal(edge, reference)


def test_windowed_reference_matches_audio_levels_and_replay_times(fake_models):
    original = fake_models.samples.copy()

    result = check_edge_audio.compare_recording("recording.wav", start_time=1700000000.0)

    assert result.windows == 6
    assert result.max_audio_error == result.max_embedding_error == result.max_score_error == 0.0
    assert result.max_level_error_db == result.max_timestamp_error_seconds == 0.0
    for index, frame in enumerate(fake_models.frames):
        assert frame.t == 1700000000.0 + index * 0.48
    # Removing the mean over each actual window keeps this changing DC offset
    # out of the model inputs, rather than subtracting one mean over the file.
    assert abs(float(fake_models.reference_audio[0].mean())) < 1e-7
    np.testing.assert_array_equal(fake_models.samples, original)


def test_final_window_level_excludes_padding_after_dc_removal(fake_models):
    fake_models.samples = np.tile([-0.5, 0.5], 80).astype(np.float32) + 0.25

    result = check_edge_audio.compare_recording("short.wav")

    assert result.windows == result.padded_windows == 1
    assert fake_models.frames[0].level_dbfs == pytest.approx(20 * np.log10(0.5))
    np.testing.assert_array_equal(fake_models.edge_audio[0][:160], np.tile([-0.5, 0.5], 80))
    np.testing.assert_array_equal(fake_models.edge_audio[0][160:], np.zeros(15440, dtype=np.float32))
    np.testing.assert_array_equal(fake_models.edge_audio[0], fake_models.reference_audio[0])


@pytest.mark.parametrize(("field", "message"), [
    ("embedding_error", "embeddings differ"), ("score_error", "class scores differ"),
])
def test_detects_model_output_differences(fake_models, field, message):
    setattr(fake_models, field, 0.1)

    with pytest.raises(ValueError, match=message):
        check_edge_audio.compare_recording("recording.wav")


def test_detects_preprocessing_differences_before_inference(fake_models, monkeypatch):
    original_prepare = check_edge_audio.edge_features.prepare_audio
    monkeypatch.setattr(
        check_edge_audio.edge_features, "prepare_audio",
        lambda samples, rate: original_prepare(samples, rate) + np.float32(0.1),
    )

    with pytest.raises(ValueError, match="prepared audio differ"):
        check_edge_audio.compare_recording("recording.wav")
    assert not fake_models.edge_audio


@pytest.mark.parametrize(("changes", "message"), [
    ({"t": 1.0}, "timestamps differ"),
    ({"t": np.nan}, "timestamps differ"),
    ({"level_dbfs": -1.0}, "audio levels differ"),
    ({"level_dbfs": None}, "audio levels differ"),
])
def test_detects_incorrect_frame_metadata(fake_models, changes, message):
    fake_models.frame_changes = changes

    with pytest.raises(ValueError, match=message):
        check_edge_audio.compare_recording("recording.wav")


def test_rejects_settings_that_would_change_window_timing(fake_models, monkeypatch):
    monkeypatch.setattr(check_edge_audio.edge_features, "HOP_SECONDS", 0.96)

    with pytest.raises(ValueError, match="HOP_SECONDS settings differ"):
        check_edge_audio.compare_recording("recording.wav")
    assert not fake_models.loads


def test_cli_prints_summary_without_writing_output_files(fake_models, tmp_path, capsys):
    model_path = tmp_path / "yamnet.tflite"

    assert check_edge_audio.main([
        "recording.wav", "--model", str(model_path), "--start-time", "1000",
    ]) == 0

    output = capsys.readouterr().out
    assert "PASS: 6 windows matched; padded final windows: 1" in output
    assert "Max embedding error: 0" in output
    assert "Max class-score error: 0" in output
    assert "Max timestamp error: 0 s" in output
    assert fake_models.loads == [model_path]
    assert not list(tmp_path.iterdir())


def test_cli_exits_unsuccessfully_if_comparison_fails(fake_models, capsys):
    fake_models.embedding_error = 0.1

    with pytest.raises(SystemExit) as error:
        check_edge_audio.main(["recording.wav"])

    output = capsys.readouterr()
    assert error.value.code == 1
    assert "Edge audio check failed" in output.err
    assert "PASS:" not in output.out


def test_rejects_non_finite_replay_start_before_model_loading(fake_models):
    with pytest.raises(ValueError, match="finite replay Unix seconds"):
        check_edge_audio.compare_recording("recording.wav", start_time=np.nan)
    assert not fake_models.loads


def test_help_does_not_load_models(fake_models):
    with pytest.raises(SystemExit) as error:
        check_edge_audio.main(["--help"])

    assert error.value.code == 0
    assert not fake_models.loads
