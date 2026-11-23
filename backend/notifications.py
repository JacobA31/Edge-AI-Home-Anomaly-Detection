"""Email queue worker and operator commands, run from backend with its venv active.

python3 -m notifications list
python3 -m notifications preview
python3 -m notifications process
python3 -m notifications retry NOTIFICATION_ID [--confirm-not-sent]

Local preview messages stay in PostgreSQL and are never later converted to SMTP sends.
Production needs EMAIL_MODE=smtp and SMTP settings in Azure environment variables.
"""

import argparse
import json
import os
from pathlib import Path
from uuid import UUID

from db.notifications import claim_next, get_preview, list_notifications, process_claim, retry_notification
from notification_email import SMTPSettings, dashboard_url, email_mode


def process_pending(limit: int = 10) -> dict:
    mode = email_mode()
    if mode == "disabled":
        return {"mode": mode, "processed": 0}
    # Validate before taking any queue entries: configuration errors leave them pending.
    settings = SMTPSettings.load() if mode == "smtp" else None
    dashboard_url()
    results: dict[str, int | str] = {"mode": mode, "processed": 0}
    for _ in range(limit):
        claim = claim_next(mode)
        if claim is None:
            break
        outcome = process_claim(claim, mode, settings)
        results["processed"] += 1
        results[outcome] = results.get(outcome, 0) + 1
    return results


def _load_local_settings() -> None:
    path = Path(__file__).resolve().parent / "local.settings.json"
    if path.exists():
        for key, value in json.loads(path.read_text())["Values"].items():
            os.environ.setdefault(key, value)
        os.environ.setdefault("EMAIL_MODE", "preview")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or process the alert email queue.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show the latest notification delivery statuses")
    listing.add_argument("--limit", type=int, default=10)
    preview = commands.add_parser("preview", help="Read a locally previewed email; does not send anything")
    preview.add_argument("notification_id", nargs="?", type=UUID)
    commands.add_parser("process", help="Process pending jobs in the configured EMAIL_MODE")
    retry = commands.add_parser("retry", help="Requeue a failed notification after fixing its cause")
    retry.add_argument("notification_id", type=UUID)
    retry.add_argument("--confirm-not-sent", action="store_true")
    args = parser.parse_args()
    _load_local_settings()
    if args.command == "list":
        if not 1 <= args.limit <= 100:
            parser.error("limit must be between 1 and 100")
        print(json.dumps(list_notifications(args.limit), default=str, indent=2))
    elif args.command == "preview":
        item = get_preview(str(args.notification_id) if args.notification_id else None)
        if item is None:
            print("No email preview found. Send a new alert for a linked device, then wait for the worker.")
        else:
            print(f"To: {item['recipient_email']}\nSubject: {item['preview_subject']}\n\n"
                  f"{item['preview_body']}")
    elif args.command == "process":
        print(json.dumps(process_pending(), indent=2))
    else:
        try:
            requeued = retry_notification(str(args.notification_id), args.confirm_not_sent)
        except ValueError as exc:
            parser.error(str(exc))
        print("Notification requeued." if requeued else "No failed/uncertain notification found to retry.")


if __name__ == "__main__":
    main()
