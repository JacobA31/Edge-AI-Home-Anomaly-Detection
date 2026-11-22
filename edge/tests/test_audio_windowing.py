"""Sample-exact streaming windows, timing, ownership, and recording boundaries."""

from dataclasses import replace

import numpy as np
import pytest

from e3_edge.audio.capture import CapturedBlock
from e3_edge.audio.windowing import WindowAssembler

RATE = 16000
WINDOW = 15600
HOP = 7680
START = 1700000000.25


def block(samples, index=0, *, discontinuity=False):
    return CapturedBlock(
        samples=samples, start_time=START + index / RATE, adc_time=100.0 + index / RATE,
        sample_index=index, discontinuity=discontinuity,
    )


def test_waits_for_a_complete_window_without_padding():
    assembler = WindowAssembler()
    samples = np.arange(WINDOW, dtype=np.float32)
    assert assembler.push(block(samples[:-1])) == []
    assert assembler.pending_samples == WINDOW - 1
    windows = assembler.push(block(samples[-1:], WINDOW - 1))
    assert len(windows) == 1
    first = windows[0]
    np.testing.assert_array_equal(first.samples, samples)
    assert first.samples.dtype == np.float32 and first.samples.shape == (WINDOW,)
    assert first.start_time == START and first.sample_index == 0
    assert first.discontinuity
    assert assembler.pending_samples == WINDOW - HOP
    assert "samples=" not in repr(first)


@pytest.mark.parametrize("chunk_size", [1, 127, 1024, HOP, WINDOW, 45000])
def test_chunk_boundaries_do_not_change_windows_or_hop(chunk_size):
    assembler = WindowAssembler()
    samples = np.arange(WINDOW + 2 * HOP + 113, dtype=np.float32)
    windows = []
    for index in range(0, len(samples), chunk_size):
        windows.extend(assembler.push(block(samples[index:index + chunk_size], index)))
    assert len(windows) == 3
    for i, window in enumerate(windows):
        start = i * HOP
        np.testing.assert_array_equal(window.samples, samples[start:start + WINDOW])
        assert window.sample_index == start
        assert window.start_time == START + start / RATE
        assert window.discontinuity == (i == 0)
    np.testing.assert_array_equal(windows[0].samples[HOP:], windows[1].samples[:WINDOW - HOP])
    assert assembler.pending_samples == WINDOW - HOP + 113


def test_irregular_blocks_and_nonzero_source_index():
    assembler = WindowAssembler()
    samples = np.arange(WINDOW + 4 * HOP, dtype=np.float32)
    sizes = iter([13, 2048, 1, 20000, 67, 30000])
    offset = 0
    windows = []
    while offset < len(samples):
        size = next(sizes)
        windows.extend(assembler.push(block(samples[offset:offset + size], 987 + offset)))
        offset += size
    assert len(windows) == 5
    for i, window in enumerate(windows):
        assert window.sample_index == 987 + i * HOP
        assert window.start_time == START + 987 / RATE + i * HOP / RATE
        np.testing.assert_array_equal(window.samples, samples[i * HOP:i * HOP + WINDOW])


def test_retained_audio_and_emitted_windows_do_not_alias_input_or_each_other():
    assembler = WindowAssembler()
    original = np.arange(WINDOW + HOP, dtype=np.float32)
    incoming = original[:8000].copy()
    assert not assembler.push(block(incoming))
    incoming.fill(-1)
    first, second = assembler.push(block(original[8000:], 8000))
    np.testing.assert_array_equal(first.samples, original[:WINDOW])
    np.testing.assert_array_equal(second.samples, original[HOP:HOP + WINDOW])
    first.samples.fill(-2)
    assembler.reset()
    np.testing.assert_array_equal(second.samples, original[HOP:HOP + WINDOW])
    np.testing.assert_array_equal(original, np.arange(WINDOW + HOP, dtype=np.float32))
    assert second.samples.flags.c_contiguous


def test_noncontiguous_input_keeps_original_sample_order():
    source = np.arange(WINDOW * 2, dtype=np.float32)[::2]
    assert not source.flags.c_contiguous
    window, = WindowAssembler().push(block(source))
    np.testing.assert_array_equal(window.samples, source)
    assert window.samples.flags.c_contiguous


@pytest.mark.parametrize("partial_length", [5000, WINDOW])
@pytest.mark.parametrize("boundary", ["flag", "missing_samples", "repeated_index", "utc", "adc"])
def test_gaps_discard_partial_audio_and_old_overlap(partial_length, boundary):
    assembler = WindowAssembler()
    assembler.push(block(np.full(partial_length, -0.5, dtype=np.float32)))
    next_block = block(np.full(WINDOW + HOP, 0.25, dtype=np.float32), partial_length)
    if boundary == "flag":
        next_block = replace(next_block, discontinuity=True)
    elif boundary == "missing_samples":
        next_block = replace(next_block, sample_index=partial_length + 1)
    elif boundary == "repeated_index":
        next_block = replace(next_block, sample_index=0)
    elif boundary == "utc":
        next_block = replace(next_block, start_time=next_block.start_time + 0.1)
    else:
        next_block = replace(next_block, adc_time=next_block.adc_time + 0.1)
    first, second = assembler.push(next_block)
    assert first.discontinuity and not second.discontinuity
    assert first.start_time == next_block.start_time
    assert first.sample_index == next_block.sample_index
    assert second.sample_index == first.sample_index + HOP
    assert second.start_time == first.start_time + 0.48
    np.testing.assert_array_equal(first.samples, np.full(WINDOW, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(second.samples, first.samples)


def test_repeated_flags_keep_resetting_until_a_full_continuous_window_exists():
    assembler = WindowAssembler()
    for index in range(0, WINDOW * 2, 1000):
        assert assembler.push(block(np.zeros(1000, dtype=np.float32), index, discontinuity=True)) == []
    final_index = index + 1000
    windows = assembler.push(block(np.ones(WINDOW - 1000, dtype=np.float32), final_index))
    assert len(windows) == 1 and windows[0].discontinuity
    assert windows[0].sample_index == index


def test_small_timestamp_jitter_does_not_change_sample_derived_window_times():
    assembler = WindowAssembler()
    windows = []
    for index in range(0, 32768, 1024):
        captured = block(np.zeros(1024, dtype=np.float32), index)
        if index:
            jitter = 0.001 if index % 2048 else -0.001
            captured = replace(
                captured, start_time=captured.start_time + jitter, adc_time=captured.adc_time + jitter,
            )
        windows.extend(assembler.push(captured))
    assert len(windows) == 3
    assert [w.start_time for w in windows] == [START + i * HOP / RATE for i in range(3)]
    assert [w.discontinuity for w in windows] == [True, False, False]


def test_long_stream_has_fixed_retention_and_no_accumulated_timestamp_error():
    assembler = WindowAssembler()
    first, = assembler.push(block(np.zeros(WINDOW, dtype=np.float32)))
    original_buffer = assembler._samples
    incoming = np.zeros(HOP, dtype=np.float32)
    for i in range(1, 2001):
        window, = assembler.push(block(incoming, WINDOW + (i - 1) * HOP))
        assert window.sample_index == i * HOP
        assert window.start_time == START + i * HOP / RATE
        assert not window.discontinuity
        assert assembler.pending_samples == WINDOW - HOP
    assert assembler._samples is original_buffer
    assert original_buffer.nbytes == WINDOW * 4
    assert first.start_time == START


def test_reset_clears_audio_and_restart_accepts_a_new_clock_and_sample_counter():
    assembler = WindowAssembler()
    assembler.push(block(np.ones(9000, dtype=np.float32)))
    assembler.reset()
    assembler.reset()  # Safe on mute/shutdown even when already empty.
    assert assembler.pending_samples == 0
    assert not assembler._samples.any()
    restarted = replace(block(np.zeros(WINDOW, dtype=np.float32)), start_time=START - 10, adc_time=2.0)
    window, = assembler.push(restarted)
    assert window.discontinuity
    assert window.sample_index == 0 and window.start_time == START - 10
    assert not window.samples.any()


@pytest.mark.parametrize("audio", [
    np.array([], dtype=np.float32),
    np.zeros(WINDOW, dtype=np.float64),
    np.zeros(WINDOW, dtype=np.int16),
    np.zeros((WINDOW, 1), dtype=np.float32),
    np.array(0, dtype=np.float32),
    np.array([np.nan], dtype=np.float32),
    np.array([np.inf], dtype=np.float32),
])
def test_invalid_audio_is_rejected_before_mutating_retained_samples(audio):
    assembler = WindowAssembler()
    assembler.push(block(np.full(100, 0.25, dtype=np.float32)))
    with pytest.raises(ValueError, match="samples"):
        assembler.push(block(audio, 100))
    assert assembler.pending_samples == 100
    window, = assembler.push(block(np.full(WINDOW - 100, 0.25, dtype=np.float32), 100))
    np.testing.assert_array_equal(window.samples, np.full(WINDOW, 0.25, dtype=np.float32))


@pytest.mark.parametrize("name, value", [
    ("start_time", np.nan), ("start_time", np.inf), ("start_time", True), ("start_time", None),
    ("adc_time", np.nan), ("adc_time", -np.inf), ("adc_time", "100"),
    ("sample_index", -1), ("sample_index", 1.5), ("sample_index", True),
    ("discontinuity", "false"),
])
def test_invalid_capture_metadata_is_rejected(name, value):
    assembler = WindowAssembler()
    with pytest.raises(ValueError, match=name):
        assembler.push(replace(block(np.zeros(100, dtype=np.float32)), **{name: value}))
    assert assembler.pending_samples == 0
