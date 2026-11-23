from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
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
class FallClassifierConfig:
    """Release and local location of the classifier and its companion settings."""

    release: str = "fall-classifier-v1"
    model_path: str = "/var/lib/e3/models/fall_classifier.tflite"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.release, str)
            or not self.release
            or any(character.isspace() for character in self.release)
        ):
            raise ValueError("fall_classifier.release must be a nonempty release tag without whitespace")
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("fall_classifier.model_path must be an absolute fall_classifier.tflite path")
        path = Path(self.model_path)
        if not path.is_absolute() or path.name != "fall_classifier.tflite":
            raise ValueError("fall_classifier.model_path must be an absolute fall_classifier.tflite path")

    @property
    def metadata_path(self) -> str:
        """The JSON settings are installed beside the model, as the runner expects."""
        return str(Path(self.model_path).with_suffix(".json"))


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
    fall_classifier: FallClassifierConfig = field(default_factory=FallClassifierConfig)
    audio_device: int | str | None = None
    audio_block_size: int = 1024
    audio_buffer_blocks: int = 32

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate != 16000
        ):
            raise ValueError("audio.sample_rate must be 16000 Hz")
        device = self.audio_device
        if device is not None and (
            isinstance(device, bool)
            or not isinstance(device, (int, str))
            or (isinstance(device, int) and device < 0)
            or (isinstance(device, str) and not device.strip())
        ):
            raise ValueError("audio.device must be an input device index, name, or null")
        sizes = (("block_size", self.audio_block_size), ("buffer_blocks", self.audio_buffer_blocks))
        for name, value in sizes:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"audio.{name} must be a positive integer")
        if (
            isinstance(self.heartbeat_seconds, bool)
            or not isinstance(self.heartbeat_seconds, Real)
            or not isfinite(self.heartbeat_seconds)
            or self.heartbeat_seconds <= 0
        ):
            raise ValueError("heartbeat_seconds must be finite and positive")


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


def _load_fall_classifier(raw) -> FallClassifierConfig:
    if raw is None:
        return FallClassifierConfig()
    if not isinstance(raw, dict):
        raise ValueError("fall_classifier must be a mapping with release and model_path settings")
    unknown = raw.keys() - {"release", "model_path"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"Unknown fall_classifier settings: {names}")
    return FallClassifierConfig(**raw)


def load(path: Path = DEFAULT_PATH) -> Config:
    raw = yaml.safe_load(path.read_text())
    audio = raw.get("audio")
    if audio is None:
        audio = {}
    if not isinstance(audio, dict):
        raise ValueError("audio must be a mapping")
    unknown = audio.keys() - {"sample_rate", "device", "block_size", "buffer_blocks"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"Unknown audio settings: {names}")
    return Config(
        device_id=raw["device_id"],
        room=raw["room"],
        backend_url=raw["backend"]["url"].rstrip("/"),
        function_key=raw["backend"]["function_key"],
        queue_db=raw["storage"]["queue_db"],
        sample_rate=audio.get("sample_rate", 16000),
        heartbeat_seconds=raw.get("heartbeat_seconds", 90),
        anomaly=raw.get("anomaly") or {},
        yamnet=_load_yamnet(raw.get("yamnet")),
        fall_classifier=_load_fall_classifier(raw.get("fall_classifier")),
        audio_device=audio.get("device"),
        audio_block_size=audio.get("block_size", 1024),
        audio_buffer_blocks=audio.get("buffer_blocks", 32),
    )
