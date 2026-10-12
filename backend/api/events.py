"""Event endpoints.

POST /api/events  – called by the Raspberry Pi. Needs a function key (x-functions-key header).
GET  /api/events  – called by the dashboard to list recent events.
"""

import json
import logging

import azure.functions as func

from validation import EventValidationError, validate_event

bp = func.Blueprint()


@bp.route(route="events", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def post_event(req: func.HttpRequest) -> func.HttpResponse:
    try:
        event = req.get_json()
    except ValueError:
        return _json({"error": "body must be JSON"}, 400)

    try:
        validate_event(event)
    except EventValidationError as exc:
        return _json({"error": str(exc)}, 422)

    # TODO(member5): insert into the events table, ignoring duplicates by event_id
    #   from db.connection import get_connection
    #   with get_connection() as conn: conn.execute("INSERT ... ON CONFLICT (event_id) DO NOTHING", ...)
    logging.info("Received %s event from %s", event["event_type"], event["device_id"])
    return _json({"status": "accepted", "event_id": event["event_id"]}, 202)


@bp.route(route="events", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def list_events(req: func.HttpRequest) -> func.HttpResponse:
    # TODO(member5): read from the database, newest first, with ?limit= and ?since=
    # TODO: protect this endpoint before real data goes in (see docs/SETUP.md, "Security")
    return _json({"events": []})


def _json(body: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(body), status_code=status, mimetype="application/json")
