import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
