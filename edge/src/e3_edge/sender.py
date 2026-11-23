"""Sends queued events to the backend over HTTPS."""

import logging
from threading import Event, Thread

import requests

from .config import Config
from .outbox import Outbox, with_elapsed

log = logging.getLogger(__name__)


def flush(outbox: Outbox, cfg: Config, *, stop_event: Event | None = None) -> int:
    """Try to send everything in the outbox. Returns how many were sent."""
    sent = 0
    for event in outbox.pending():
        if stop_event is not None and stop_event.is_set():
            break
        try:
            r = requests.post(
                f"{cfg.backend_url}/api/events",
                json=with_elapsed(event),
                headers={"x-functions-key": cfg.function_key},
                timeout=10,
            )
        except requests.RequestException as exc:
            log.warning("Backend unreachable, will retry: %s", exc)
            break
        if r.status_code in (200, 202):
            outbox.mark_sent(event["event_id"])
            sent += 1
        elif 400 <= r.status_code < 500:
            log.error("Backend rejected %s (%s): %s", event["event_id"], r.status_code, r.text)
            outbox.mark_sent(event["event_id"])  # a bad event won't get better by retrying
        else:
            log.warning("Backend error %s, will retry", r.status_code)
            break
    return sent


class SenderWorker:
    """Send events independently of capture/inference, using a thread-owned DB.

    Requests retain the existing timeout and retry behavior. stop() interrupts
    retry waits and batch draining, then waits at most one second for an in-flight
    request. The daemon thread cannot hold process shutdown open if HTTP stalls;
    an unacknowledged event stays in SQLite for the next process to retry.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._stop_event = Event()
        self._error: Exception | None = None
        self._thread = Thread(target=self._run, name="event-sender", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def check(self) -> None:
        """Surface worker failures to the application's supervisor."""
        if self._error is not None:
            raise RuntimeError("Event sender failed") from self._error
        if not self._thread.is_alive() and not self._stop_event.is_set():
            raise RuntimeError("Event sender stopped unexpectedly")

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("Sender is still finishing a request; unacknowledged events remain queued")

    def _run(self) -> None:
        try:
            # sqlite3 connections must be created and used on the same thread.
            with Outbox(self._cfg.queue_db) as outbox:
                while not self._stop_event.is_set():
                    flush(outbox, self._cfg, stop_event=self._stop_event)
                    self._stop_event.wait(1.0)
        except Exception as exc:
            self._error = exc
