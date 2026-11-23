"""Event retries, worker isolation, and shutdown without external HTTP."""

from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import requests

from e3_edge import sender
from e3_edge.config import Config
from e3_edge.events import make_event
from e3_edge.outbox import Outbox


@pytest.fixture
def cfg(tmp_path):
    return Config(
        device_id="test-pi", room="kitchen", backend_url="https://example.invalid",
        function_key="test-key", queue_db=str(tmp_path / "queue.sqlite"),
    )


@pytest.mark.parametrize("status, removed", [(200, True), (202, True), (400, True), (503, False)])
def test_flush_retains_retryable_events_and_preserves_detection_metadata(cfg, monkeypatch, status, removed):
    payloads = []
    event = make_event(cfg.device_id, cfg.room, "heartbeat")

    def post(url, **kwargs):
        assert url == "https://example.invalid/api/events"
        assert kwargs["timeout"] == 10
        payloads.append(kwargs["json"])
        return SimpleNamespace(status_code=status, text="test response")

    monkeypatch.setattr(sender.requests, "post", post)
    with Outbox(cfg.queue_db) as outbox:
        outbox.put(event)
        assert sender.flush(outbox, cfg) == int(status in (200, 202))
        assert len(outbox) == int(not removed)
    assert payloads[0]["event_id"] == event["event_id"]
    assert payloads[0]["detected_at"] == event["detected_at"]
    assert payloads[0]["elapsed_seconds"] >= 0


def test_network_failure_keeps_event_queued(cfg, monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(sender.requests, "post", unavailable)
    with Outbox(cfg.queue_db) as outbox:
        event = make_event(cfg.device_id, cfg.room, "heartbeat")
        outbox.put(event)
        assert sender.flush(outbox, cfg) == 0
        assert outbox.pending() == [event]


def test_stop_event_interrupts_batch_before_the_next_request(cfg, monkeypatch):
    stop = Event()
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        stop.set()
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(sender.requests, "post", post)
    with Outbox(cfg.queue_db) as outbox:
        events = [make_event(cfg.device_id, cfg.room, "heartbeat") for _ in range(2)]
        for event in events:
            outbox.put(event)
        assert sender.flush(outbox, cfg, stop_event=stop) == 1
        assert outbox.pending() == [events[1]]
        assert sender.flush(outbox, cfg, stop_event=stop) == 0
    assert len(sent) == 1


def test_worker_creates_uses_and_closes_connection_on_its_own_thread(cfg, monkeypatch):
    used_threads, closed_threads = [], []
    sent = Event()

    class TrackedOutbox(Outbox):
        def __init__(self, path):
            used_threads.append(get_ident())
            super().__init__(path)

        def close(self):
            closed_threads.append(get_ident())
            super().close()

    def post(url, **kwargs):
        used_threads.append(get_ident())
        sent.set()
        return SimpleNamespace(status_code=200)

    with Outbox(cfg.queue_db) as outbox:
        outbox.put(make_event(cfg.device_id, cfg.room, "heartbeat"))
    monkeypatch.setattr(sender, "Outbox", TrackedOutbox)
    monkeypatch.setattr(sender.requests, "post", post)
    worker = sender.SenderWorker(cfg)
    worker.start()
    try:
        assert sent.wait(2), "worker did not send its queued event"
        worker.check()
    finally:
        worker.stop(timeout=2)
    assert not worker._thread.is_alive()
    assert used_threads == [worker._thread.ident, worker._thread.ident]
    assert closed_threads == [worker._thread.ident]
    assert worker._thread.ident != get_ident()
    with Outbox(cfg.queue_db) as outbox:
        assert len(outbox) == 0


def test_worker_failure_propagates_and_closes_database(cfg, monkeypatch):
    error = ValueError("invalid queued event")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(sender, "flush", fail)
    worker = sender.SenderWorker(cfg)
    worker.start()
    worker._thread.join(timeout=2)
    try:
        with pytest.raises(RuntimeError, match="Event sender failed") as raised:
            worker.check()
        assert raised.value.__cause__ is error
    finally:
        worker.stop()
    assert not worker._thread.is_alive()


def test_shutdown_during_http_preserves_the_unsent_remainder(cfg, monkeypatch):
    entered, release = Event(), Event()
    events = [make_event(cfg.device_id, cfg.room, "heartbeat") for _ in range(2)]
    with Outbox(cfg.queue_db) as outbox:
        for event in events:
            outbox.put(event)

    def blocked_post(*args, **kwargs):
        entered.set()
        release.wait(5)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(sender.requests, "post", blocked_post)
    worker = sender.SenderWorker(cfg)
    worker.start()
    try:
        assert entered.wait(2)
        worker.stop(timeout=0.01)
        assert worker._thread.is_alive() and worker._thread.daemon
        with Outbox(cfg.queue_db) as outbox:
            assert outbox.pending() == events
    finally:
        release.set()
        worker.stop(timeout=2)
    assert not worker._thread.is_alive()
    with Outbox(cfg.queue_db) as outbox:
        assert outbox.pending() == [events[1]]
