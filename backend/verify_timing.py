"""Local cloud timing measurements using disposable, fictional preview-only data.

From backend with its virtual environment active:
  python3 -m verify_timing run
  python3 -m verify_timing report --backend-file reports/timing/run-ID.json --browser-file FILE
The run prints a temporary login and dashboard URL. No real Pi or SMTP is exercised.
"""

import argparse
import json
import math
import os
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID, uuid4

from psycopg import Error
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from consent_disclosure import DISCLOSURE_VERSION
from db import accounts, consent, devices
from db.connection import get_connection
from security import hash_password, token_hash

REPORTS = Path(__file__).resolve().parent / "reports" / "timing"


def local_settings() -> None:
    path = Path(__file__).resolve().parent / "local.settings.json"
    if path.exists():
        for key, value in json.loads(path.read_text())["Values"].items():
            os.environ.setdefault(key, value)
    url = os.environ.get("DATABASE_URL", "")
    if not url or conninfo_to_dict(url).get("host") not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("the timing helper requires a loopback local development database")


def local_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise RuntimeError("use an HTTP loopback site origin, such as http://localhost:7071")
    return value.rstrip("/")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("local timing requests must not redirect")


def request(base: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Cookie"] = "e3_session=" + token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    try:
        with build_opener(NoRedirect).open(Request(base + "/api" + path, data=data, headers=headers),
                                           timeout=3) as reply:
            return reply.status, json.load(reply)
    except HTTPError as exc:
        raise RuntimeError(f"local backend returned HTTP {exc.code} for {path.split('?')[0]}") from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError("could not read the local backend; check that func start is running") from exc


def create_fixture(run: dict) -> tuple[str, str]:
    suffix = UUID(run["run_id"]).hex
    password = secrets.token_urlsafe(24)
    stored = hash_password(password)
    account = accounts.create_account(f"timing-{suffix}@example.invalid", "Timing test (simulated)", stored)
    run["fixture"] = {"account_id": account["account_id"], "email": account["email"],
                      "device_id": f"timing-{suffix}"}
    code = devices.provision_device(run["fixture"]["device_id"], "Timing test (simulated)")
    devices.claim_device(account["account_id"], code)
    # This is a fictional device with no microphone/resident; never grant consent
    # or change permissions for any existing household device during verification.
    resident_code = consent.provision_code(run["fixture"]["device_id"])
    consent.change_consent(resident_code, "grant", 0, DISCLOSURE_VERSION)
    token = accounts.create_session(account["account_id"], account["email"], stored)
    if token is None:
        raise RuntimeError("could not create the temporary timing session")
    return password, token


def cleanup(run: dict) -> None:
    fixture = run.get("fixture")
    if not fixture:
        return
    suffix = UUID(run["run_id"]).hex
    if (fixture["device_id"] != f"timing-{suffix}"
            or fixture["email"] != f"timing-{suffix}@example.invalid"):
        raise RuntimeError("refusing to remove data that does not match the synthetic timing run")
    with get_connection() as conn:
        owner = conn.execute(
            "SELECT account_id FROM accounts WHERE email = %s", (fixture["email"],),
        ).fetchone()
        if owner is not None and str(owner[0]) != fixture["account_id"]:
            raise RuntimeError("temporary account ID does not match; no data was removed")
        device = conn.execute(
            "SELECT room FROM devices WHERE device_id = %s", (fixture["device_id"],),
        ).fetchone()
        if device is not None and (owner is None or device[0] != "Timing test (simulated)"):
            raise RuntimeError("temporary device does not match; no data was removed")
        conn.execute("DELETE FROM events WHERE device_id = %s", (fixture["device_id"],))
        conn.execute("DELETE FROM devices WHERE device_id = %s", (fixture["device_id"],))
        conn.execute("DELETE FROM accounts WHERE account_id = %s AND email = %s",
                     (fixture["account_id"], fixture["email"]))
        conn.execute("DELETE FROM auth_attempts WHERE attempt_key = %s", (token_hash(fixture["email"]),))


def notification_results(run: dict) -> dict:
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT event_id, delivery_mode, status, queued_at, finished_at FROM email_notifications "
            "WHERE event_id = ANY(%s::uuid[]) AND account_id = %s",
            ([sample["event_id"] for sample in run["samples"]], run["fixture"]["account_id"]),
        )
        return {str(row["event_id"]): row for row in cursor.fetchall()}


def collect(run: dict, token: str, samples: int) -> None:
    start = time.monotonic()
    wall_start = time.time()
    # Different phases relative to the dashboard's 5s poll and worker's 10s tick.
    offsets = (0, 1.7, 4.4, 7.8, 10.3)[:samples]
    created: dict[str, float] = {}
    next_sample = 0
    deadline = start + offsets[-1] + 45
    params = urlencode({"alerts_only": "true", "device_id": run["fixture"]["device_id"], "limit": 100})
    while time.monotonic() < deadline:
        now = time.monotonic()
        if abs((time.time() - wall_start) - (now - start)) > .25:
            raise RuntimeError("the system clock changed during measurement; repeat the run")
        if next_sample < samples and now >= start + offsets[next_sample]:
            event = {"event_id": str(uuid4()), "device_id": run["fixture"]["device_id"],
                     "event_type": "fall", "room": "Timing test (simulated)",
                     "detected_at": datetime.now(UTC).isoformat(), "elapsed_seconds": 0,
                     "confidence": .8, "details": {"timing_run_id": run["run_id"]}}
            created[event["event_id"]] = time.monotonic()
            sample = {"event_id": event["event_id"], "created_at": event["detected_at"],
                      "post_status": None, "first_api_seconds": None, "notification": None}
            run["samples"].append(sample)
            try:
                sample["post_status"], _ = request(run["backend_url"], "/events", event)
            except RuntimeError as exc:
                sample["post_error"] = str(exc)
            next_sample += 1
            print(f"Created synthetic alert {next_sample}/{samples}.", flush=True)
        if run["samples"]:
            try:
                _, page = request(run["backend_url"], "/events?" + params, token=token)
                observed = time.monotonic()
                visible_ids = {event["event_id"] for event in page["events"]}
                for sample in run["samples"]:
                    if sample["first_api_seconds"] is None and sample["event_id"] in visible_ids:
                        sample["first_api_seconds"] = observed - created[sample["event_id"]]
            except RuntimeError as exc:
                run["last_api_error"] = str(exc)
            statuses = notification_results(run)
            for sample in run["samples"]:
                row = statuses.get(sample["event_id"])
                if row:
                    if row["delivery_mode"] != "preview":
                        raise RuntimeError("email mode changed; refusing to continue the preview check")
                    completed = row["finished_at"]
                    seconds = ((completed - datetime.fromisoformat(sample["created_at"])).total_seconds()
                               if completed else None)
                    sample["notification"] = {
                        "mode": row["delivery_mode"], "status": row["status"],
                        "queued_at": row["queued_at"].isoformat(),
                        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
                        "preview_seconds": seconds,
                    }
            done = next_sample == samples and all(
                item["first_api_seconds"] is not None and item["notification"] is not None
                and item["notification"]["status"] == "previewed" for item in run["samples"]
            )
            # Keep collecting long enough for an active dashboard to cross the
            # full 10-second budget, even if API and preview work finish earlier.
            if done and time.monotonic() >= start + offsets[-1] + 12:
                break
        time.sleep(.25)


def timed_status(values: list[float | None], budget: float) -> dict:
    valid = [value for value in values if type(value) in (int, float) and math.isfinite(value) and value >= 0]
    return {"status": "PASS" if len(valid) == len(values) and values and max(valid) <= budget else "FAIL",
            "budget_seconds": budget, "samples": len(values), "measured": len(valid),
            "maximum_seconds": max(valid) if valid else None}


def summarize(run: dict, browser: dict | None = None) -> dict:
    if (run.get("kind") != "e3-local-cloud-timing" or run.get("schema_version") != 1
            or not isinstance(run.get("samples"), list) or not run["samples"]):
        raise RuntimeError("not a supported backend timing report")
    expected = {sample["event_id"]: sample for sample in run["samples"]}
    if str(UUID(run["run_id"])) != run["run_id"]:
        raise RuntimeError("invalid timing run ID")
    requested = run["requested_samples"]
    if type(requested) is not int or not 1 <= requested <= 5 or len(expected) > requested:
        raise RuntimeError("invalid number of timing samples")
    if len(expected) != len(run["samples"]):
        raise RuntimeError("backend report contains repeated event IDs")
    dashboard = {"status": "NOT_MEASURED", "budget_seconds": 10}
    if browser is not None:
        if (browser.get("kind") != "e3-browser-alert-timing" or browser.get("schema_version") != 1
                or browser.get("run_id") != run["run_id"]):
            raise RuntimeError("browser measurements must belong to this exact timing run")
        observed = {}
        for item in browser.get("observations", []):
            event_id = item["event_id"]
            if event_id not in expected or event_id in observed:
                raise RuntimeError("browser report contains an unexpected or repeated event ID")
            if (datetime.fromisoformat(item["detected_at"]) !=
                    datetime.fromisoformat(expected[event_id]["created_at"])):
                raise RuntimeError("browser and backend event creation times do not match")
            seconds = (datetime.fromisoformat(item["observed_at"]) -
                       datetime.fromisoformat(expected[event_id]["created_at"])).total_seconds()
            valid = item.get("foreground") is True and item.get("clock_consistent") is True
            observed[event_id] = seconds if valid and expected[event_id]["post_status"] == 202 else None
        dashboard = timed_status([observed.get(event_id) for event_id in expected]
                                 + [None] * (requested - len(expected)), 10)
    preview_values = []
    for sample in run["samples"]:
        notification = sample.get("notification") or {}
        seconds = None
        if (notification.get("status") == "previewed" and notification.get("mode") == "preview"
                and notification.get("finished_at") and sample["post_status"] == 202):
            seconds = (datetime.fromisoformat(notification["finished_at"]) -
                       datetime.fromisoformat(sample["created_at"])).total_seconds()
        preview_values.append(seconds)
    preview_values += [None] * (requested - len(expected))
    return {
        "run_id": run["run_id"], "scope": "Synthetic local cloud check; no real Pi or SMTP provider",
        "dashboard_render": dashboard,
        "api_availability_only": timed_status(
            [sample["first_api_seconds"] for sample in run["samples"]]
            + [None] * (requested - len(expected)), 10),
        "notification_preview_only": timed_status(preview_values, 30),
        "email_service_submission": {"status": "NOT_MEASURED", "budget_seconds": 30,
                                     "reason": "Preview completion does not measure SMTP or inbox delivery."},
        "full_pi_to_cloud_targets": "NOT_VERIFIED",
    }


def run_check(args) -> None:
    local_settings()
    base, frontend = local_origin(args.backend_url), local_origin(args.frontend_url)
    _, health = request(base, "/health?timing=1")
    if health.get("email_mode") != "preview":
        raise RuntimeError("start the local backend with EMAIL_MODE=preview before running this check")
    with get_connection() as conn:
        before = time.time()
        db_now = conn.execute("SELECT clock_timestamp()").fetchone()[0].timestamp()
        after = time.time()
        if not before - .25 <= db_now <= after + .25:
            raise RuntimeError("database clock differs from this computer; local timing is not comparable")
    run = {"kind": "e3-local-cloud-timing", "schema_version": 1, "run_id": str(uuid4()),
           "backend_url": base, "frontend_url": frontend, "requested_samples": args.samples, "samples": []}
    path = REPORTS / f"run-{run['run_id']}.json"
    try:
        password, token = create_fixture(run)
        # Save a credential-free manifest early so an interrupted run can also
        # be cleaned up explicitly using its report file.
        REPORTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run, indent=2) + "\n")
        # Prove the running backend sees the same database/session before posting.
        _, logged_in = request(base, "/auth/me", token=token)
        if logged_in["user"]["account_id"] != run["fixture"]["account_id"]:
            raise RuntimeError("backend is not connected to the selected local database")
        print(f"Open: {frontend}/?{urlencode({'timing_run': run['run_id']})}")
        print(f"Temporary email: {run['fixture']['email']}\nTemporary password: {password}")
        print("Log in using this fictional account. Keep that tab active; do not refresh during measurement.")
        input("Press Enter once its Local timing check panel says Ready: ")
        collect(run, token, args.samples)
        run["summary_without_browser"] = summarize(run)
        path.write_text(json.dumps(run, indent=2) + "\n")
        print(f"Backend report: {path}")
        print(json.dumps(run["summary_without_browser"], indent=2))
        print("Download browser timings from the dashboard before closing it.")
        input("After saving browser timings, press Enter to remove all temporary timing fixtures: ")
    finally:
        try:
            if run["samples"]:
                path.write_text(json.dumps(run, indent=2) + "\n")
        finally:
            cleanup(run)
            print("Removed temporary timing fixtures. Existing household records were not changed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Measure a live local backend using preview-only synthetic alerts")
    run.add_argument("--samples", type=int, choices=range(1, 6), default=5)
    run.add_argument("--backend-url", default="http://localhost:7071")
    run.add_argument("--frontend-url", default="http://localhost:5173")
    report = commands.add_parser("report", help="Combine the matching backend and browser measurements")
    report.add_argument("--backend-file", type=Path, required=True)
    report.add_argument("--browser-file", type=Path, required=True)
    remove = commands.add_parser("cleanup", help="Remove only fixtures from an interrupted synthetic run")
    remove.add_argument("--backend-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run_check(args)
        elif args.command == "cleanup":
            local_settings()
            cleanup(json.loads(args.backend_file.read_text()))
            print("Removed matching synthetic timing fixtures, if present.")
        else:
            result = summarize(json.loads(args.backend_file.read_text()),
                               json.loads(args.browser_file.read_text()))
            path = args.backend_file.with_name(f"summary-{result['run_id']}.json")
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            print(f"Combined report: {path}")
            if (result["dashboard_render"]["status"] != "PASS"
                    or result["notification_preview_only"]["status"] != "PASS"):
                parser.exit(1, "One or more local checks failed or lacked valid measurements.\n")
    except Error:
        parser.exit(1, "Local database operation failed; check settings and apply database migrations.\n")
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Timing check could not complete: {exc}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(1, "Timing check interrupted.\n")


if __name__ == "__main__":
    main()
