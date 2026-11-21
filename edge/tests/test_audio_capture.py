"""Microphone acquisition checks without sounddevice, PortAudio, or hardware."""

from types import SimpleNamespace

import numpy as np
import pytest

from e3_edge.audio import capture


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(streams=[], settings=[], utc=1700000000.0, stream_time=100.0)

    class CallbackAbort(Exception):
        pass

    class Stream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.active = False
            self.aborts = self.closes = 0
            self.time = state.stream_time
            self.next_adc = state.stream_time
            state.streams.append(self)

        def start(self):
            self.active = True

        def abort(self):
            self.aborts += 1
            self.active = False

        def close(self):
            self.closes += 1
            self.active = False

        def emit(self, samples, *, adc=None, overflow=False):
            if samples.ndim == 1:
                samples = samples[:, None]
            adc = self.next_adc if adc is None else adc
            self.next_adc = adc + len(samples) / 16000
            status = SimpleNamespace(input_overflow=overflow)
            # SimpleNamespace is always truthy; emulate sounddevice.CallbackFlags.
            class Status:
                input_overflow = status.input_overflow

                def __bool__(self):
                    return self.input_overflow

            try:
                self.kwargs["callback"](
                    samples, len(samples), SimpleNamespace(inputBufferAdcTime=adc), Status(),
                )
            except CallbackAbort:
                self.active = False

    def check_input_settings(**kwargs):
        state.settings.append(kwargs)

    module = SimpleNamespace(
        InputStream=Stream, CallbackAbort=CallbackAbort, check_input_settings=check_input_settings,
    )
    state.module = module
    monkeypatch.setattr(capture, "import_module", lambda name: module)
    monkeypatch.setattr(capture, "time", lambda: state.utc)
    return state


def samples(start=0, count=4):
    return np.arange(start, start + count, dtype=np.float32) / 100


def test_explicit_input_format_device_and_lifecycle(backend):
    microphone = capture.MicrophoneCapture("ReSpeaker", block_size=4, buffer_blocks=2)
    with microphone as started:
        assert started is microphone
        assert microphone.read() is None
        stream = backend.streams[0]
        assert backend.settings == [{
            "device": "ReSpeaker", "channels": 1, "dtype": "float32", "samplerate": 16000,
        }]
        assert stream.kwargs == {
            "device": "ReSpeaker", "samplerate": 16000, "channels": 1, "dtype": "float32",
            "blocksize": 4, "callback": microphone._callback,
        }
    assert not stream.active
    assert stream.aborts == stream.closes == 1
    microphone.stop()
    assert stream.aborts == stream.closes == 1


def test_owned_mono_blocks_keep_adc_timing_despite_consumer_delay(backend):
    with capture.MicrophoneCapture(block_size=4) as microphone:
        stream = backend.streams[0]
        incoming = samples()
        stream.emit(incoming, adc=99.95)
        incoming[:] = -1  # PortAudio may reuse callback memory as soon as it returns.
        backend.utc += 100  # Wall-clock changes and delayed reads must not retime audio.
        first = microphone.read()
        assert first.start_time == pytest.approx(1699999999.95, rel=0, abs=1e-6)
        assert first.adc_time == 99.95 and first.sample_index == 0
        assert first.discontinuity
        assert first.samples.shape == (4,) and first.samples.dtype == np.float32
        np.testing.assert_array_equal(first.samples, samples())
        stream.emit(samples(4))
        second = microphone.read()
        assert second.start_time - first.start_time == pytest.approx(4 / 16000, abs=3e-7)
        assert second.sample_index == 4 and not second.discontinuity
        np.testing.assert_array_equal(first.samples, samples())
        assert "samples=" not in repr(first)


def test_ring_wraparound_discards_oldest_and_marks_the_gap(backend):
    with capture.MicrophoneCapture(block_size=4, buffer_blocks=2) as microphone:
        stream = backend.streams[0]
        stream.emit(samples())
        assert microphone.read().sample_index == 0
        for start in (4, 8, 12):
            stream.emit(samples(start))
        first = microphone.read()
        second = microphone.read()
        assert first.sample_index == 8 and first.discontinuity
        assert second.sample_index == 12 and not second.discontinuity
        np.testing.assert_array_equal(first.samples, samples(8))
        np.testing.assert_array_equal(second.samples, samples(12))
        assert microphone.dropped_samples == 4
        assert microphone.read() is None
        assert microphone._samples.nbytes == 2 * 4 * 4
        assert not microphone._samples.any()


def test_callback_does_not_wait_for_consumer_lock(backend):
    with capture.MicrophoneCapture(block_size=4) as microphone:
        stream = backend.streams[0]
        stream.emit(samples())
        microphone.read()
        with microphone._lock:
            stream.emit(samples(4))  # A blocking callback would deadlock this test.
        stream.emit(samples(8))
        block = microphone.read()
        assert block.sample_index == 8 and block.discontinuity
        assert microphone.dropped_samples == 4


@pytest.mark.parametrize("overflow, jump", [(True, 0), (False, 0.1)])
def test_driver_overflow_or_adc_gap_breaks_continuity(backend, overflow, jump):
    with capture.MicrophoneCapture(block_size=4) as microphone:
        stream = backend.streams[0]
        stream.emit(samples())
        microphone.read()
        stream.emit(samples(4), adc=stream.next_adc + jump, overflow=overflow)
        block = microphone.read()
        assert block.sample_index == 4 and block.discontinuity
        assert microphone.input_overflows == int(overflow)
        assert microphone.dropped_samples == 0  # Driver loss has unknown length.
        stream.emit(samples(8))
        assert not microphone.read().discontinuity


def test_variable_short_blocks_preserve_sample_counts(backend):
    with capture.MicrophoneCapture(block_size=4) as microphone:
        stream = backend.streams[0]
        stream.emit(samples(count=1))
        stream.emit(samples(1, count=3))
        first, second = microphone.read(), microphone.read()
        assert len(first.samples) == 1 and second.sample_index == 1
        assert not second.discontinuity
        np.testing.assert_array_equal(second.samples, samples(1, count=3))


@pytest.mark.parametrize("adc", [float("nan"), float("inf"), -float("inf"), 99.0, 100.0])
def test_bad_or_backwards_adc_time_surfaces_callback_failure(backend, adc):
    with capture.MicrophoneCapture(block_size=4) as microphone:
        stream = backend.streams[0]
        stream.emit(samples())
        stream.emit(samples(4), adc=adc)
        with pytest.raises(capture.CaptureError, match="callback failed") as error:
            microphone.read()
        assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("audio", [
    np.zeros((4, 2), dtype=np.float32),
    np.zeros(4, dtype=np.int16),
    np.zeros(5, dtype=np.float32),
    np.zeros(0, dtype=np.float32),
])
def test_unexpected_callback_format_is_not_silently_converted(backend, audio):
    with capture.MicrophoneCapture(block_size=4) as microphone:
        backend.streams[0].emit(audio)
        with pytest.raises(capture.CaptureError, match="callback failed"):
            microphone.read()


def test_unexpected_stream_termination_is_reported_even_with_queued_audio(backend):
    with capture.MicrophoneCapture(block_size=4) as microphone:
        stream = backend.streams[0]
        stream.emit(samples())
        stream.active = False
        with pytest.raises(capture.CaptureError, match="inactive"):
            microphone.read()


def test_stop_clears_audio_and_restart_uses_a_fresh_clock_anchor(backend):
    microphone = capture.MicrophoneCapture(block_size=4)
    with microphone:
        stream = backend.streams[0]
        stream.emit(samples())
    assert not microphone._samples.any()
    stream.emit(samples(4))  # A late callback after stop cannot repopulate the ring.
    assert microphone._count == 0
    backend.utc += 500
    backend.stream_time = 2.0
    with microphone:
        assert microphone.read() is None
        backend.streams[1].emit(samples())
        block = microphone.read()
        assert block.sample_index == 0 and block.discontinuity
        assert block.start_time == backend.utc


def test_invalid_lifecycle_calls_are_clear(backend):
    microphone = capture.MicrophoneCapture()
    with pytest.raises(RuntimeError, match="not started"):
        microphone.read()
    with microphone:
        with pytest.raises(RuntimeError, match="already started"):
            microphone.start()


def test_start_failure_closes_device_and_allows_retry(backend, monkeypatch):
    stream_class = backend.module.InputStream
    original = stream_class.start

    def fail_start(self):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(stream_class, "start", fail_start)
    microphone = capture.MicrophoneCapture()
    with pytest.raises(RuntimeError, match="device unavailable"):
        microphone.start()
    assert backend.streams[0].closes == 1
    assert microphone._stream is None
    monkeypatch.setattr(stream_class, "start", original)
    with microphone:
        assert backend.streams[1].active


def test_context_exception_still_closes_and_clears_capture(backend):
    microphone = capture.MicrophoneCapture(block_size=4)
    with pytest.raises(RuntimeError, match="consumer failed"), microphone:
        backend.streams[0].emit(samples())
        raise RuntimeError("consumer failed")
    assert backend.streams[0].closes == 1
    assert not microphone._samples.any()


@pytest.mark.parametrize("stream_time", [float("nan"), float("inf")])
def test_invalid_clock_anchor_closes_device_before_start(backend, stream_time):
    backend.stream_time = stream_time
    with pytest.raises(capture.CaptureError, match="clock mapping"):
        capture.MicrophoneCapture().start()
    assert backend.streams[0].closes == 1


def test_wall_clock_step_during_anchor_is_rejected(backend, monkeypatch):
    times = iter([1700000000.0, 1700000001.0])
    monkeypatch.setattr(capture, "time", lambda: next(times))
    with pytest.raises(capture.CaptureError, match="Clock mapping"):
        capture.MicrophoneCapture().start()
    assert backend.streams[0].closes == 1


def test_clock_mapping_uses_midpoint_of_bracket(backend, monkeypatch):
    times = iter([1700000000.0, 1700000000.002])
    monkeypatch.setattr(capture, "time", lambda: next(times))
    with capture.MicrophoneCapture(block_size=4) as microphone:
        backend.streams[0].emit(samples(), adc=99.9)
        assert microphone.read().start_time == pytest.approx(1699999999.901, rel=0, abs=1e-6)


def test_stop_still_closes_and_clears_if_abort_fails(backend, monkeypatch):
    microphone = capture.MicrophoneCapture(block_size=4).start()
    stream = backend.streams[0]
    stream.emit(samples())

    def fail_abort():
        raise RuntimeError("abort failed")

    monkeypatch.setattr(stream, "abort", fail_abort)
    with pytest.raises(RuntimeError, match="abort failed"):
        microphone.stop()
    assert stream.closes == 1
    assert microphone._stream is None
    assert not microphone._samples.any()
    microphone.stop()


def test_unsupported_device_settings_do_not_open_stream(backend):
    def reject(**kwargs):
        raise ValueError("16 kHz unsupported")

    backend.module.check_input_settings = reject
    with pytest.raises(ValueError, match="unsupported"):
        capture.MicrophoneCapture().start()
    assert not backend.streams


def test_optional_sounddevice_is_loaded_only_when_starting(monkeypatch):
    calls = []

    def missing(name):
        calls.append(name)
        raise ModuleNotFoundError("No sounddevice", name="sounddevice")

    monkeypatch.setattr(capture, "import_module", missing)
    microphone = capture.MicrophoneCapture()
    assert not calls
    with pytest.raises(ImportError, match="Microphone capture requires"):
        microphone.start()
    assert calls == ["sounddevice"]


@pytest.mark.parametrize("name", ["block_size", "buffer_blocks"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_buffer_settings(name, value):
    with pytest.raises(ValueError, match=name):
        capture.MicrophoneCapture(**{name: value})


@pytest.mark.parametrize("device", [True, -1, "", "  ", 1.5])
def test_invalid_device_selection(device):
    with pytest.raises(ValueError, match="device"):
        capture.MicrophoneCapture(device)
