"""Pipeline wiring checks; set E3_YAMNET_MODEL to include the real-model check."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from e3_edge.anomaly.frames import Frame
from e3_edge.audio import pipeline


@pytest.fixture
def fake_yamnet(monkeypatch):
    state = SimpleNamespace(loads=[], calls=[])

    class FakeRunner:
        def __init__(self, model_path, *, num_threads=1):
            state.loads.append((model_path, num_threads))

        def extract_features(self, waveform):
            state.calls.append(waveform.copy())
            return (
                np.full(1024, np.max(np.abs(waveform)), dtype=np.float32),
                np.linspace(0.0, 1.0, 521, dtype=np.float32),
            )

    monkeypatch.setattr(pipeline, "YAMNetRunner", FakeRunner)
    return state


def test_complete_window_becomes_a_frame_with_aligned_outputs(fake_yamnet):
    audio = pipeline.AudioPipeline("model.tflite")
    clean = np.tile([-0.125, 0.125], 7800)
    samples = clean + 0.5
    original = samples.copy()

    frame = audio.process_window(samples, start_time=1700000000.25)

    assert isinstance(frame, Frame)
    assert frame.t == 1700000000.25
    assert frame.embedding.shape == (1024,) and frame.scores.shape == (521,)
    assert frame.embedding.dtype == frame.scores.dtype == np.float32
    np.testing.assert_array_equal(frame.embedding, np.full(1024, 0.125, dtype=np.float32))
    np.testing.assert_array_equal(frame.scores, np.linspace(0.0, 1.0, 521, dtype=np.float32))
    assert frame.level_dbfs == pytest.approx(20 * np.log10(0.125))
    assert fake_yamnet.calls[0].dtype == np.float32
    np.testing.assert_array_equal(fake_yamnet.calls[0], clean)
    np.testing.assert_array_equal(samples, original)


def test_stereo_and_clipping_use_the_same_samples_for_yamnet_and_level(fake_yamnet):
    audio = pipeline.AudioPipeline("model.tflite")
    samples = np.tile([[0.25, 0.75], [-0.25, -0.75]], (7800, 1))
    samples[0] = [1.5, 1.5]
    samples[1] = [-1.5, -1.5]
    original = samples.copy()
    expected = np.tile(np.array([0.5, -0.5], dtype=np.float32), 7800)
    expected[0] = 1.0
    expected[1] = -1.0

    frame = audio.process_window(samples, start_time=1000.0)

    np.testing.assert_array_equal(fake_yamnet.calls[0], expected)
    expected_level = 20 * np.log10(np.sqrt((2.0 + 15598 * 0.5 ** 2) / 15600))
    assert frame.level_dbfs == pytest.approx(expected_level)
    np.testing.assert_array_equal(samples, original)


def test_reuses_runner_and_keeps_each_supplied_timestamp(fake_yamnet):
    audio = pipeline.AudioPipeline("model.tflite", num_threads=2)
    first = audio.process_window(np.tile([-0.25, 0.25], 7800), start_time=1000.0)
    second = audio.process_window(np.tile([-0.5, 0.5], 7800), start_time=1000.48)

    assert fake_yamnet.loads == [("model.tflite", 2)]
    assert len(fake_yamnet.calls) == 2
    assert first.t == 1000.0 and second.t == 1000.48
    assert np.all(first.embedding == 0.25) and np.all(second.embedding == 0.5)
    assert first.level_dbfs == pytest.approx(20 * np.log10(0.25))
    assert second.level_dbfs == pytest.approx(20 * np.log10(0.5))


def test_silence_has_a_finite_floor_level(fake_yamnet):
    audio = pipeline.AudioPipeline("model.tflite")

    frame = audio.process_window(np.zeros(15600, dtype=np.float32), start_time=1000.0)

    assert frame.level_dbfs == -120.0


def test_dc_offset_alone_is_silent_for_both_yamnet_and_level(fake_yamnet):
    audio = pipeline.AudioPipeline("model.tflite")
    samples = np.tile([0.25, 0.5], (15600, 1))

    frame = audio.process_window(samples, start_time=1000.0)

    np.testing.assert_array_equal(fake_yamnet.calls[0], np.zeros(15600, dtype=np.float32))
    assert np.all(frame.embedding == 0)
    assert frame.level_dbfs == -120.0


@pytest.mark.parametrize("length", [1, 100, 15599])
def test_short_windows_need_explicit_end_padding(fake_yamnet, length):
    audio = pipeline.AudioPipeline("model.tflite")

    with pytest.raises(ValueError, match="incomplete final window"):
        audio.process_window(np.zeros(length, dtype=np.float32), start_time=1000.0)
    assert not fake_yamnet.calls


@pytest.mark.parametrize("pad_end", [False, True])
def test_long_windows_are_not_silently_truncated(fake_yamnet, pad_end):
    audio = pipeline.AudioPipeline("model.tflite")

    with pytest.raises(ValueError, match="Expected 15600 samples"):
        audio.process_window(np.zeros(15601, dtype=np.float32), start_time=1000.0, pad_end=pad_end)
    assert not fake_yamnet.calls


def test_final_window_is_padded_but_level_uses_only_recorded_samples(fake_yamnet):
    audio = pipeline.AudioPipeline("model.tflite")
    samples = np.tile(np.array([-0.5, 0.5], dtype=np.float32), 80)
    original = samples.copy()

    frame = audio.process_window(samples, start_time=1000.0, pad_end=True)

    sent = fake_yamnet.calls[0]
    assert sent.shape == (15600,) and sent.dtype == np.float32
    np.testing.assert_array_equal(sent[:160], samples)
    np.testing.assert_array_equal(sent[160:], np.zeros(15440, dtype=np.float32))
    assert frame.t == 1000.0
    assert frame.level_dbfs == pytest.approx(20 * np.log10(0.5))
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("start_time", [np.nan, np.inf, -np.inf, "not-a-timestamp", None, True])
def test_rejects_invalid_timestamps_before_inference(fake_yamnet, start_time):
    audio = pipeline.AudioPipeline("model.tflite")

    with pytest.raises(ValueError, match="finite UTC Unix seconds"):
        audio.process_window(np.zeros(15600, dtype=np.float32), start_time=start_time)
    assert not fake_yamnet.calls


@pytest.mark.parametrize("samples", [
    np.array([], dtype=np.float32),
    np.zeros(15600, dtype=np.int16),
    np.full(15600, np.nan, dtype=np.float32),
    np.zeros((15600, 1, 1), dtype=np.float32),
])
def test_rejects_invalid_audio_before_inference(fake_yamnet, samples):
    audio = pipeline.AudioPipeline("model.tflite")

    with pytest.raises(ValueError):
        audio.process_window(samples, start_time=1000.0)
    assert not fake_yamnet.calls


@pytest.mark.parametrize("sample_rate", [8000, 48000])
def test_rejects_audio_that_needs_resampling(fake_yamnet, sample_rate):
    audio = pipeline.AudioPipeline("model.tflite")

    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        audio.process_window(
            np.zeros(15600, dtype=np.float32), start_time=1000.0, sample_rate=sample_rate,
        )
    assert not fake_yamnet.calls


@pytest.mark.skipif(
    not os.environ.get("E3_YAMNET_MODEL"),
    reason="Set E3_YAMNET_MODEL to an exported .tflite to test the real audio pipeline",
)
def test_real_pipeline_returns_frames_for_full_and_final_windows():
    audio = pipeline.AudioPipeline(os.environ["E3_YAMNET_MODEL"])
    time = np.arange(15600, dtype=np.float64) / 16000
    tone = 0.25 * np.sin(2 * np.pi * 400 * time)
    stereo = tone[:, None] + np.array([0.5, 0.25])

    full = audio.process_window(stereo, start_time=1700000000.0)
    final = audio.process_window(
        (tone[:160] + 0.5).astype(np.float32), start_time=1700000000.48, pad_end=True,
    )

    assert full.t == 1700000000.0 and final.t == 1700000000.48
    for frame in (full, final):
        assert isinstance(frame, Frame)
        assert frame.embedding.shape == (1024,) and frame.scores.shape == (521,)
        assert frame.embedding.dtype == frame.scores.dtype == np.float32
        assert np.isfinite(frame.embedding).all() and np.isfinite(frame.scores).all()
        assert np.all((frame.scores >= 0) & (frame.scores <= 1))
        assert frame.level_dbfs == pytest.approx(20 * np.log10(0.25 / np.sqrt(2)))
