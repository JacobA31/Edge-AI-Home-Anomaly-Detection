from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class YAMNetConfig:
    """The release to install and the local file the runner will open.

    Reading these settings does not download anything or require the model
    file to exist. Use an absolute path so installation and inference agree
    regardless of the program's current working directory.
    """

    release: str = "yamnet-v1"
    model_path: str = "/var/lib/e3/models/yamnet.tflite"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.release, str)
            or not self.release
            or any(character.isspace() for character in self.release)
        ):
            raise ValueError("yamnet.release must be a nonempty release tag without whitespace")
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("yamnet.model_path must be a nonempty absolute .tflite file path")
        path = Path(self.model_path)
        if not path.is_absolute() or path.suffix.lower() != ".tflite":
            raise ValueError("yamnet.model_path must be an absolute .tflite file path")


@dataclass(frozen=True)
class Config:
    device_id: str
    room: str
    backend_url: str
    function_key: str
    queue_db: str
    sample_rate: int = 16000
    heartbeat_seconds: int = 90
    anomaly: dict = field(default_factory=dict)  # raw `anomaly:` section, see e3_edge.anomaly.config
    yamnet: YAMNetConfig = field(default_factory=YAMNetConfig)


def _load_yamnet(raw) -> YAMNetConfig:
    if raw is None:
        return YAMNetConfig()
    if not isinstance(raw, dict):
        raise ValueError("yamnet must be a mapping with release and model_path settings")
    unknown = raw.keys() - {"release", "model_path"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"Unknown yamnet settings: {names}")
    return YAMNetConfig(**raw)


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
        anomaly=raw.get("anomaly") or {},
        yamnet=_load_yamnet(raw.get("yamnet")),
    )
