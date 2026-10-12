from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class Config:
    device_id: str
    room: str
    backend_url: str
    function_key: str
    queue_db: str
    sample_rate: int = 16000
    heartbeat_seconds: int = 90


def load(path: Path = DEFAULT_PATH) -> Config:
    raw = yaml.safe_load(path.read_text())
    return Config(
        device_id=raw["device_id"],
        room=raw["room"],
        backend_url=raw["backend"]["url"].rstrip("/"),
        function_key=raw["backend"]["function_key"],
        queue_db=raw["storage"]["queue_db"],
        sample_rate=raw.get("audio", {}).get("sample_rate", 16000),
        heartbeat_seconds=raw.get("heartbeat_seconds", 90),
    )
