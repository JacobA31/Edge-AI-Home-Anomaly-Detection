"""Live-loop orchestration without microphone hardware, model files, or HTTP."""

import signal
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from e3_edge import __main__ as entrypoint
from e3_edge import runtime
from e3_edge.anomaly.frames import Frame
from e3_edge.audio.capture import CapturedBlock, CaptureError
from e3_edge.config import Config
from e3_edge.outbox import Outbox


@pytest.fixture
def app(tmp_path, monkeypatch):
    stop = Event()
    state = SimpleNamespace(
        stop=stop, blocks=[], inputs=[], frames=[], gaps=[], loaded=[], capture_options=[],
        opened=False, closed=False, sender_started=False, sender_stopped=False,
        sender_error=None, clock=0.0, read_hook=None,
    )
    state.cfg = Config(
        device_id="test-pi", room="kitchen", backend_url="https://example.invalid",
        function_key="test-key", queue_db=str(tmp_path / "queue.sqlite"), audio_device="Test mic",
        heartbeat_seconds=2,
    )

    class FakePipeline:
        def __init__(self, path):
            state.loaded.append(path)

        def process_window(self, samples, *, start_time):
            state.inputs.append(samples.copy())
            return Frame(start_time, np.ones(1024, dtype=np.float32), np.zeros(521, dtype=np.float32))

    class FakeCapture:
        dropped_samples = 17
        input_overflows = 2

        def __init__(self, **kwargs):
            state.capture_options.append(kwargs)

        def __enter__(self):
            state.opened = True
            return self

        def __exit__(self, *args):
            state.closed = True

        def read(self):
            if state.read_hook is not None:
                state.read_hook()
            if state.blocks:
                value = state.blocks.pop(0)
                if isinstance(value, BaseException):
                    raise value
                return value
            stop.set()
            return None

    class FakeSender:
        def __init__(self, cfg):
            assert cfg is state.cfg

        def start(self):
            state.sender_started = True

        def check(self):
            if state.sender_error is not None:
                raise state.sender_error

        def stop(self):
            state.sender_stopped = True

    monkeypatch.setattr(runtime, "AudioPipeline", FakePipeline)
    monkeypatch.setattr(runtime, "MicrophoneCapture", FakeCapture)
    monkeypatch.setattr(runtime, "SenderWorker", FakeSender)
    # Replace runtime's module reference, not Python's shared time.monotonic.
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: state.clock))
    return state


def captured(samples, index=0, *, discontinuity=False):
    return CapturedBlock(
        samples=samples, start_time=1700000000.0 + index / 16000,
        adc_time=100.0 + index / 16000, sample_index=index, discontinuity=discontinuity,
    )


def test_run_connects_capture_windowing_pipeline_and_frame_consumer(app):
    samples = np.linspace(-0.5, 0.5, 15600 + 7680, dtype=np.float32)
    app.blocks = [captured(samples[i:i + 1024], i) for i in range(0, len(samples), 1024)]
    order = []

    def on_frame(frame):
        order.append("frame")
        app.frames.append(frame)

    count = runtime.run(
        app.cfg, stop_event=app.stop, on_frame=on_frame, on_gap=lambda: order.append("gap"),
    )
    assert count == len(app.frames) == 2
    assert order == ["gap", "frame", "frame"]
    assert app.loaded == [app.cfg.yamnet.model_path]
    assert app.capture_options == [{"device": "Test mic", "block_size": 1024, "buffer_blocks": 32}]
    assert [frame.t for frame in app.frames] == [1700000000.0, 1700000000.48]
    for i, values in enumerate(app.inputs):
        np.testing.assert_array_equal(values, samples[i * 7680:i * 7680 + 15600])
    assert app.closed and app.sender_started and app.sender_stopped
    with Outbox(app.cfg.queue_db) as outbox:
        event, = outbox.pending()
    assert event["event_type"] == "heartbeat"
    assert event["details"] == {
        "audio_active": True, "frames_processed": 0, "dropped_samples": 17, "input_overflows": 2,
    }


def test_heartbeats_follow_monotonic_time_and_do_not_repeat_for_missed_intervals(app, monkeypatch):
    app.blocks = [None] * 4
    times = iter([0.0, 2.0, 20.0, 20.1, 20.2])

    def advance():
        app.clock = next(times)

    app.read_hook = advance
    # No audio is supplied in this scheduling-only scenario.
    monkeypatch.setattr(runtime, "CAPTURE_STALL_SECONDS", 100.0)
    runtime.run(app.cfg, stop_event=app.stop)
    with Outbox(app.cfg.queue_db) as outbox:
        assert len(outbox.pending()) == 3


def test_capture_gap_interrupts_downstream_before_the_new_frame(app):
    app.blocks = [
        captured(np.zeros(15600, dtype=np.float32)),
        captured(np.ones(15600, dtype=np.float32), 20000, discontinuity=True),
    ]
    order = []
    runtime.run(
        app.cfg, stop_event=app.stop, on_frame=lambda frame: order.append(frame.t),
        on_gap=lambda: order.append("gap"),
    )
    assert order == ["gap", 1700000000.0, "gap", 1700000001.25]


@pytest.mark.parametrize("failure", [CaptureError("unplugged"), KeyboardInterrupt()])
def test_capture_failure_closes_devices_sender_database_and_partial_audio(app, monkeypatch, failure):
    assembler = runtime.WindowAssembler()
    monkeypatch.setattr(runtime, "WindowAssembler", lambda: assembler)
    app.blocks = [captured(np.ones(1000, dtype=np.float32)), failure]
    with pytest.raises(type(failure)):
        runtime.run(app.cfg, stop_event=app.stop)
    assert app.closed and app.sender_stopped
    assert assembler.pending_samples == 0
    assert not assembler._samples.any()
    with Outbox(app.cfg.queue_db) as outbox:
        assert len(outbox) == 1


def test_sender_failure_stops_capture_instead_of_silently_losing_events(app):
    app.sender_error = RuntimeError("sender failed")
    with pytest.raises(RuntimeError, match="sender failed"):
        runtime.run(app.cfg, stop_event=app.stop)
    assert app.closed and app.sender_stopped


def test_consumer_failure_is_propagated_after_cleanup(app):
    app.blocks = [captured(np.zeros(15600, dtype=np.float32))]

    def fail(frame):
        raise ValueError("consumer failed")

    with pytest.raises(ValueError, match="consumer failed"):
        runtime.run(app.cfg, stop_event=app.stop, on_frame=fail)
    assert app.closed and app.sender_stopped


def test_shutdown_during_inference_does_not_deliver_a_late_frame(app, monkeypatch):
    app.blocks = [captured(np.zeros(15600, dtype=np.float32))]
    pipeline = runtime.AudioPipeline("fake")
    original = pipeline.process_window

    def stop_during_model(samples, *, start_time):
        app.stop.set()
        return original(samples, start_time=start_time)

    monkeypatch.setattr(pipeline, "process_window", stop_during_model)
    monkeypatch.setattr(runtime, "AudioPipeline", lambda path: pipeline)
    assert runtime.run(app.cfg, stop_event=app.stop, on_frame=app.frames.append) == 0
    assert not app.frames
    assert app.closed and app.sender_stopped


def test_stopping_in_gap_handler_does_not_start_inference(app):
    app.blocks = [captured(np.zeros(15600, dtype=np.float32))]
    assert runtime.run(app.cfg, stop_event=app.stop, on_gap=app.stop.set) == 0
    assert not app.inputs


def test_stalled_active_capture_is_reported(app):
    def stall():
        app.clock += 6.0

    app.read_hook = stall
    app.blocks = [None] * 3
    with pytest.raises(CaptureError, match="No microphone audio"):
        runtime.run(app.cfg, stop_event=app.stop)
    assert app.closed and app.sender_stopped


def test_slow_inference_does_not_trigger_stall_when_audio_is_available(app):
    app.blocks = [
        captured(np.zeros(15600, dtype=np.float32)), captured(np.zeros(7680, dtype=np.float32), 15600),
    ]

    def consume(frame):
        app.clock += 10.0
        app.frames.append(frame)
        if len(app.frames) == 2:
            app.stop.set()

    assert runtime.run(app.cfg, stop_event=app.stop, on_frame=consume) == 2


def test_already_stopped_runtime_does_not_open_resources(app):
    app.stop.set()
    assert runtime.run(app.cfg, stop_event=app.stop) == 0
    assert not app.loaded and not app.opened and not app.sender_started


def test_missing_model_does_not_open_capture_or_start_sender(app, monkeypatch):
    def missing(path):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(runtime, "AudioPipeline", missing)
    with pytest.raises(FileNotFoundError, match="model missing"):
        runtime.run(app.cfg, stop_event=app.stop)
    assert not app.opened and not app.sender_started


def test_stop_during_model_loading_does_not_open_capture_or_sender(app, monkeypatch):
    pipeline = runtime.AudioPipeline("fake")

    def load(path):
        app.stop.set()
        return pipeline

    monkeypatch.setattr(runtime, "AudioPipeline", load)
    assert runtime.run(app.cfg, stop_event=app.stop) == 0
    assert not app.opened and not app.sender_started


def test_capture_startup_failure_stops_sender(app, monkeypatch):
    def unavailable(**kwargs):
        raise CaptureError("microphone unavailable")

    monkeypatch.setattr(runtime, "MicrophoneCapture", unavailable)
    with pytest.raises(CaptureError, match="microphone unavailable"):
        runtime.run(app.cfg, stop_event=app.stop)
    assert app.sender_started and app.sender_stopped
    with Outbox(app.cfg.queue_db) as outbox:
        assert len(outbox) == 0


def test_inference_failure_closes_capture_and_clears_window_buffer(app, monkeypatch):
    pipeline = runtime.AudioPipeline("fake")
    assembler = runtime.WindowAssembler()

    def fail(samples, *, start_time):
        raise RuntimeError("model failed")

    monkeypatch.setattr(pipeline, "process_window", fail)
    monkeypatch.setattr(runtime, "AudioPipeline", lambda path: pipeline)
    monkeypatch.setattr(runtime, "WindowAssembler", lambda: assembler)
    app.blocks = [captured(np.ones(15600, dtype=np.float32))]
    with pytest.raises(RuntimeError, match="model failed"):
        runtime.run(app.cfg, stop_event=app.stop)
    assert app.closed and app.sender_stopped
    assert assembler.pending_samples == 0 and not assembler._samples.any()


def test_blocked_http_does_not_stop_inference_and_shutdown_is_bounded(app, monkeypatch):
    from e3_edge import sender

    entered, release = Event(), Event()
    payloads = []

    def blocked_post(url, **kwargs):
        payloads.append(kwargs["json"])
        entered.set()
        release.wait(5)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(sender.requests, "post", blocked_post)
    worker = sender.SenderWorker(app.cfg)
    monkeypatch.setattr(runtime, "SenderWorker", lambda cfg: worker)
    app.blocks = [
        captured(np.zeros(15600, dtype=np.float32)), captured(np.zeros(7680, dtype=np.float32), 15600),
    ]

    def consume(frame):
        app.frames.append(frame)
        if len(app.frames) == 2:
            assert entered.is_set() and not release.is_set()
            app.stop.set()

    def wait_for_sender():
        if app.frames:
            assert entered.wait(2), "sender did not start HTTP"

    app.read_hook = wait_for_sender
    try:
        assert runtime.run(app.cfg, stop_event=app.stop, on_frame=consume) == 2
        assert app.closed and worker._thread.is_alive() and worker._thread.daemon
        assert len(payloads) == 1 and payloads[0]["event_type"] == "heartbeat"
        assert set(payloads[0]["details"]) == {
            "audio_active", "frames_processed", "dropped_samples", "input_overflows",
        }
        with Outbox(app.cfg.queue_db) as outbox:
            assert len(outbox) == 1
    finally:
        release.set()
        worker.stop(timeout=2)
    assert not worker._thread.is_alive()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_entrypoint_handles_shutdown_signals_and_restores_handlers(app, monkeypatch, sig):
    handlers = {}
    original = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(entrypoint.config, "load", lambda: app.cfg)
    monkeypatch.setattr(
        entrypoint.signal, "signal", lambda number, handler: handlers.update({number: handler}),
    )

    def fake_run(cfg, *, stop_event):
        assert cfg is app.cfg and not stop_event.is_set()
        handlers[sig](sig, None)
        assert stop_event.is_set()

    monkeypatch.setattr(entrypoint, "run", fake_run)
    entrypoint.main()
    assert handlers == original


def test_entrypoint_restores_signal_handlers_after_startup_failure(app, monkeypatch):
    handlers = {}
    original = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(entrypoint.config, "load", lambda: app.cfg)
    monkeypatch.setattr(
        entrypoint.signal, "signal", lambda number, handler: handlers.update({number: handler}),
    )

    def fail(cfg, **kwargs):
        raise RuntimeError("model missing")

    monkeypatch.setattr(entrypoint, "run", fail)
    with pytest.raises(RuntimeError, match="model missing"):
        entrypoint.main()
    assert handlers == original
