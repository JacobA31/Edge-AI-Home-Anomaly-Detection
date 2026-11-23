import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from e3_edge.events import make_event
from e3_edge.outbox import Outbox, with_elapsed

SCHEMA = Path(__file__).resolve().parents[2] / "shared" / "event.schema.json"


def test_queue_survives_until_marked_sent(tmp_path):
    box = Outbox(str(tmp_path / "q.sqlite"))
    event = make_event("pi-1", "kitchen", "fall", confidence=0.9)
    box.put(event)
    box.put(event)  # duplicate put is ignored
    assert len(box) == 1
    box.mark_sent(event["event_id"])
    assert len(box) == 0


def test_elapsed_seconds_filled_at_send_time():
    event = make_event("pi-1", "kitchen", "fall")
    later = datetime.now(UTC) + timedelta(seconds=30)
    assert 29 <= with_elapsed(event, later)["elapsed_seconds"] <= 31


def test_event_only_uses_schema_fields():
    allowed = set(json.loads(SCHEMA.read_text())["properties"])
    event = make_event("pi-1", "kitchen", "anomaly", confidence=0.5, details={"label": "silence"})
    assert set(event) <= allowed


def test_outbox_context_closes_connection_and_preserves_pending_events(tmp_path):
    path = str(tmp_path / "queue.sqlite")
    event = make_event("pi-1", "kitchen", "heartbeat")
    with pytest.raises(RuntimeError, match="shutdown"), Outbox(path) as outbox:
        outbox.put(event)
        raise RuntimeError("shutdown")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        outbox.pending()
    outbox.close()
    with Outbox(path) as reopened:
        assert reopened.pending() == [event]
