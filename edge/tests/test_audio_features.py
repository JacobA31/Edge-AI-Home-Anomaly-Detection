"""Check the input contract for edge audio preparation without hardware."""

import numpy as np
import pytest

from e3_edge.audio.features import prepare_audio


def test_mono_preserves_quiet_audio_and_does_not_modify_input():
    samples = np.array([-0.025, 0.0, 0.025], dtype=np.float64)
    original = samples.copy()

    prepared = prepare_audio(samples, 16000)

    assert prepared.dtype == np.float32
    assert prepared.shape == (3,)
    np.testing.assert_allclose(prepared, original)
    np.testing.assert_array_equal(samples, original)


def test_stereo_averages_channels_for_each_sample():
    samples = np.array([[0.2, 0.6], [-0.8, 0.2], [0.5, -0.5]], dtype=np.float32)
    original = samples.copy()

    prepared = prepare_audio(samples, 16000)

    assert prepared.dtype == np.float32
    assert prepared.shape == (3,)
    np.testing.assert_allclose(prepared, [0.4, -0.3, 0.0], atol=1e-7)
    np.testing.assert_array_equal(samples, original)


def test_single_channel_matrix_becomes_a_vector():
    prepared = prepare_audio(np.array([[0.2], [-0.4]], dtype=np.float32), 16000)

    assert prepared.shape == (2,)
    np.testing.assert_allclose(prepared, [0.2, -0.4])


def test_clips_out_of_range_samples():
    prepared = prepare_audio(np.array([-1.2, -1.0, 0.5, 1.0, 1.2]), 16000)

    np.testing.assert_array_equal(prepared, [-1.0, -1.0, 0.5, 1.0, 1.0])


def test_silence_stays_silent():
    prepared = prepare_audio(np.zeros((32, 2), dtype=np.float32), 16000)

    np.testing.assert_array_equal(prepared, np.zeros(32, dtype=np.float32))


@pytest.mark.parametrize("sample_rate", [8000, 44100, 48000])
def test_rejects_audio_that_needs_resampling(sample_rate):
    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        prepare_audio(np.zeros(10, dtype=np.float32), sample_rate)


@pytest.mark.parametrize("samples", [
    np.array([], dtype=np.float32),
    np.empty((10, 0), dtype=np.float32),
    np.array(0.0),
    np.zeros((2, 3, 4), dtype=np.float32),
])
def test_rejects_empty_audio_and_invalid_dimensions(samples):
    with pytest.raises(ValueError, match="shaped"):
        prepare_audio(samples, 16000)


@pytest.mark.parametrize("samples", [
    np.array([0, 100], dtype=np.int16),
    np.array([True, False]),
    np.array([0.1 + 0.2j]),
    np.array(["sound"]),
    np.array([0.1, np.nan]),
    np.array([[0.1, np.inf]]),
    np.array([-np.inf, 0.1]),
])
def test_rejects_non_float_or_non_finite_samples(samples):
    with pytest.raises(ValueError, match="finite, normalized floating point"):
        prepare_audio(samples, 16000)
