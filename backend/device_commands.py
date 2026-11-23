"""Operator provisioning and an explicitly simulated Pi for local control checks.

From backend with its virtual environment active:
  python3 -m device_commands provision pi-livingroom-01
  python3 -m device_commands simulate pi-livingroom-01
The simulator never operates a microphone or an actual Pi.
"""

import argparse
import json
import os
import time
from getpass import getpass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from db.device_commands import provision_token
from db.devices import DeviceError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("provision", "simulate"))
    parser.add_argument("device_id")
    args = parser.parse_args()
    if not 1 <= len(args.device_id) <= 64:
        parser.error("device ID must contain 1 to 64 characters")
    if args.action == "provision":
        if not os.environ.get("DATABASE_URL"):
            settings = Path(__file__).resolve().parent / "local.settings.json"
            if not settings.exists():
                parser.error("set DATABASE_URL or create backend/local.settings.json first")
            os.environ["DATABASE_URL"] = json.loads(settings.read_text())["Values"]["DATABASE_URL"]
        try:
            token = provision_token(args.device_id)
        except DeviceError as exc:
            parser.error(str(exc))
        print(f"Device: {args.device_id}\nDevice control token: {token}")
        print("Save privately. Re-running provision immediately invalidates the previous control token.")
        print("This is separate from the event function key and the caregiver linking code.")
        return

    print("LOCAL SIMULATION ONLY: no microphone or real Pi will be changed.")
    token = getpass("Device control token: ").strip()
    base = "http://localhost:7071/api/device-control/" + quote(args.device_id, safe="")

    def post(path: str, body: dict) -> dict:
        request = Request(base + path, method="POST", data=json.dumps(body).encode(), headers={
            "Authorization": "Bearer " + token, "Content-Type": "application/json",
        })
        with urlopen(request, timeout=15) as reply:
            return json.load(reply)

    try:
        collected = post("/next", {})
        policy = collected["consent"]
        command = collected["command"]
        print(f"Consent: {policy['state']} (version {policy['version']}).")
        if not policy["device_confirmed"]:
            if not policy["monitoring_allowed"]:
                print("STOP REQUIRED: the real Pi must stop capture, analysis and monitoring-data "
                      "collection.")
                outcome = input("Simulate stopping [success/failure/leave]: ").strip().lower()
                if outcome != "success":
                    print("Stopping remains unconfirmed. Resume is blocked.")
                    return
            else:
                print("Synchronizing granted permission. This simulation keeps monitoring muted "
                      "until a resume request.")
            post("/consent/ack", {
                "version": policy["version"], "monitoring_state": "muted", "elapsed_seconds": 0,
            })
            print("Simulated consent policy confirmed; monitoring is muted.")
            if not policy["monitoring_allowed"]:
                if command is not None and command["action"] == "mute":
                    post(f"/commands/{command['command_id']}/ack", {"succeeded": True, "elapsed_seconds": 0})
                return
        if command is None:
            print("No outstanding command. Request mute/resume in the dashboard first, then run again.")
            return
        print(f"Picked up {command['action']} request {command['command_id']}.")
        print("The dashboard should still say it is awaiting device confirmation.")
        outcome = input("Simulated outcome [success/failure/leave]: ").strip().lower()
        if outcome not in {"success", "failure"}:
            print("Left unconfirmed. Run the simulator again to pick up the same request.")
            return
        # Permission may have been withdrawn while the operator was answering.
        latest = post("/next", {})
        if latest["consent"]["version"] != policy["version"]:
            print("Consent changed. No command execution was simulated. Run again to apply "
                  "the current policy.")
            return
        executed_at = time.monotonic()
        reply = post(f"/commands/{command['command_id']}/ack", {
            "succeeded": outcome == "success", "elapsed_seconds": time.monotonic() - executed_at,
        })
        print("Simulated acknowledgement:", reply["command"]["status"])
    except HTTPError as exc:
        parser.exit(1, f"Backend returned {exc.code}: {exc.read().decode()}\n")
    except URLError:
        parser.exit(1, "Could not reach the local backend. Start it and try again.\n")


if __name__ == "__main__":
    main()
