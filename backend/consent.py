"""Issue/rotate a resident's private consent code using operator DB credentials.

From backend with the virtual environment active:
python3 -m consent provision pi-livingroom-01
No caregiver account can issue or retrieve this code through the website.
"""

import argparse
import json
import os
from pathlib import Path

from db.consent import provision_code
from db.devices import DeviceError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("provision",))
    parser.add_argument("device_id")
    args = parser.parse_args()
    if not 1 <= len(args.device_id) <= 64:
        parser.error("device ID must contain 1 to 64 characters")
    if not os.environ.get("DATABASE_URL"):
        settings = Path(__file__).resolve().parent / "local.settings.json"
        if not settings.exists():
            parser.error("set DATABASE_URL or create backend/local.settings.json first")
        os.environ["DATABASE_URL"] = json.loads(settings.read_text())["Values"]["DATABASE_URL"]
    try:
        code = provision_code(args.device_id)
    except DeviceError as exc:
        parser.error(str(exc))
    print(f"Device: {args.device_id}\nPrivate consent code: {code}")
    print("Give this code privately to the person being monitored. It is not a "
          "caregiver invitation or Pi token.")
    print("Use the Monitoring consent page (/consent). No dashboard account is needed.")
    print("Anyone with this code can give/withdraw permission. Re-running provision "
          "replaces the previous code.")
    print("Creating/replacing a code does not itself grant or withdraw consent.")


if __name__ == "__main__":
    main()
