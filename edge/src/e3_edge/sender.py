"""Sends queued events to the backend over HTTPS."""

import logging

import requests

from .config import Config
from .outbox import Outbox, with_elapsed

log = logging.getLogger(__name__)


def flush(outbox: Outbox, cfg: Config) -> int:
    """Try to send everything in the outbox. Returns how many were sent."""
    sent = 0
    for event in outbox.pending():
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
