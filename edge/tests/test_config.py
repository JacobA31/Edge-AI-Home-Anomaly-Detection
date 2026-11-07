"""YAMNet configuration checks without a model, downloads, or Pi hardware."""

from pathlib import Path

import pytest
import yaml

from e3_edge import config


@pytest.fixture
def settings():
    return {
        "device_id": "test-pi", "room": "living room",
        "backend": {"url": "https://example.invalid/", "function_key": "test-key"},
        "storage": {"queue_db": "/tmp/test-queue.sqlite"},
        "audio": {"sample_rate": 16000}, "heartbeat_seconds": 45,
        "anomaly": {"timezone": "America/Toronto"},
    }


def write_config(tmp_path, settings):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_older_config_without_yamnet_keeps_working_with_defaults(tmp_path, settings):
    cfg = config.load(write_config(tmp_path, settings))

    assert cfg.yamnet.release == "yamnet-v1"
    assert cfg.yamnet.model_path == "/var/lib/e3/models/yamnet.tflite"
    assert cfg.device_id == "test-pi" and cfg.room == "living room"
    assert cfg.backend_url == "https://example.invalid"
    assert cfg.function_key == "test-key"
    assert cfg.queue_db == "/tmp/test-queue.sqlite"
    assert cfg.sample_rate == 16000 and cfg.heartbeat_seconds == 45
    assert cfg.anomaly == {"timezone": "America/Toronto"}


@pytest.mark.parametrize("section", [None, {}])
def test_empty_yamnet_section_uses_defaults(tmp_path, settings, section):
    settings["yamnet"] = section

    cfg = config.load(write_config(tmp_path, settings))

    assert cfg.yamnet.release == "yamnet-v1"
    assert cfg.yamnet.model_path == "/var/lib/e3/models/yamnet.tflite"


def test_reads_custom_release_and_path_without_requiring_or_creating_a_model(tmp_path, settings):
    model_path = tmp_path / "models" / "custom-yamnet.tflite"
    settings["yamnet"] = {"release": "yamnet-v2", "model_path": str(model_path)}

    cfg = config.load(write_config(tmp_path, settings))

    assert cfg.yamnet.release == "yamnet-v2"
    assert cfg.yamnet.model_path == str(model_path)
    assert not model_path.parent.exists()


@pytest.mark.parametrize("section", [
    {"release": "yamnet-v2"},
    {"model_path": "/opt/e3/models/custom.tflite"},
])
def test_partial_settings_keep_defaults_for_other_fields(tmp_path, settings, section):
    settings["yamnet"] = section

    cfg = config.load(write_config(tmp_path, settings))

    assert cfg.yamnet.release == section.get("release", "yamnet-v1")
    assert cfg.yamnet.model_path == section.get("model_path", "/var/lib/e3/models/yamnet.tflite")


def test_example_configuration_declares_the_pi_model_location():
    example = Path(__file__).resolve().parents[1] / "config.example.yaml"

    cfg = config.load(example)

    assert cfg.yamnet.release == "yamnet-v1"
    assert cfg.yamnet.model_path == "/var/lib/e3/models/yamnet.tflite"


@pytest.mark.parametrize("release", ["", " ", "yamnet v1", None, 1, True])
def test_rejects_invalid_release_tags(tmp_path, settings, release):
    settings["yamnet"] = {"release": release}

    with pytest.raises(ValueError, match="yamnet.release"):
        config.load(write_config(tmp_path, settings))


@pytest.mark.parametrize("model_path", [
    "", " ", "models/yamnet.tflite", "/var/lib/e3/models/",
    "/var/lib/e3/models/yamnet.onnx", 1, None,
])
def test_rejects_invalid_model_paths(tmp_path, settings, model_path):
    settings["yamnet"] = {"model_path": model_path}

    with pytest.raises(ValueError, match="yamnet.model_path"):
        config.load(write_config(tmp_path, settings))


@pytest.mark.parametrize("section", [[], "yamnet-v1", False, 17])
def test_rejects_yamnet_sections_that_are_not_mappings(tmp_path, settings, section):
    settings["yamnet"] = section

    with pytest.raises(ValueError, match="yamnet must be a mapping"):
        config.load(write_config(tmp_path, settings))


def test_rejects_misspelled_yamnet_settings(tmp_path, settings):
    settings["yamnet"] = {"modelpath": "/var/lib/e3/models/yamnet.tflite"}

    with pytest.raises(ValueError, match="Unknown yamnet settings: modelpath"):
        config.load(write_config(tmp_path, settings))


def test_direct_config_construction_still_has_yamnet_defaults():
    cfg = config.Config(
        device_id="test-pi", room="living room", backend_url="https://example.invalid",
        function_key="test-key", queue_db="/tmp/test-queue.sqlite",
    )

    assert cfg.yamnet.release == "yamnet-v1"
    assert cfg.yamnet.model_path == "/var/lib/e3/models/yamnet.tflite"
