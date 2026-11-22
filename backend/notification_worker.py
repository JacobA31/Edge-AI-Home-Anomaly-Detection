"""Durably queued alert emails are submitted independently of Pi event ingestion."""

import logging

import azure.functions as func
from psycopg.errors import UndefinedTable

from notifications import process_pending

bp = func.Blueprint()


@bp.timer_trigger(schedule="*/10 * * * * *", arg_name="timer", run_on_startup=False, use_monitor=False)
def deliver_alert_emails(timer: func.TimerRequest) -> None:
    try:
        result = process_pending()
        if result["processed"]:
            logging.info("Alert email worker: %s", result)
    except UndefinedTable:
        logging.warning("Email queue is not initialized yet; apply the database migrations.")
    except Exception:
        logging.exception("Alert email worker failed; queued notifications remain available for inspection")
