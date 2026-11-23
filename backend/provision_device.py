"""Issue a one-time linking code using the setup operator's database credentials.

From backend, with its virtual environment active:
python3 -m provision_device pi-livingroom-01 --room "living room"
Uses DATABASE_URL, or the ignored local.settings.json for local development.
"""

import argparse
import json
import os
from pathlib import Path

from db.devices import DeviceError, provision_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue a private, single-use device linking code (24 hours).",
    )
    parser.add_argument("device_id")
    parser.add_argument("--room", required=True)
    args = parser.parse_args()
    if not 1 <= len(args.device_id) <= 64 or not 1 <= len(args.room.strip()) <= 64:
        parser.error("device ID and room must contain 1 to 64 characters")
    if not os.environ.get("DATABASE_URL"):
        settings = Path(__file__).resolve().parent / "local.settings.json"
        if not settings.exists():
            parser.error("set DATABASE_URL or create backend/local.settings.json first")
        os.environ["DATABASE_URL"] = json.loads(settings.read_text())["Values"]["DATABASE_URL"]
    try:
        code = provision_device(args.device_id, args.room.strip())
    except DeviceError as exc:
        parser.error(str(exc))
    print(f"Device: {args.device_id}")
    print(f"Linking code: {code}")
    print("Expires in 24 hours. Share privately with the first trusted caregiver; do not commit it.")


if __name__ == "__main__":
    main()
