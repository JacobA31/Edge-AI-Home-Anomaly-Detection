"""Model installation checks with simulated GitHub calls; no services or Git updates."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from e3_edge.config import YAMNetConfig

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "download_models.py"
EXAMPLE_CONFIG = SCRIPT.parents[1] / "config.example.yaml"
SPEC = spec_from_file_location("e3_download_models", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
download_models = module_from_spec(SPEC)
sys.modules[SPEC.name] = download_models
SPEC.loader.exec_module(download_models)


@pytest.fixture
def github(tmp_path, monkeypatch):
    payload = b"verified-model"
    state = SimpleNamespace(
        settings=YAMNetConfig(model_path=str(tmp_path / "installed" / "yamnet.tflite")),
        payload=payload, download_bytes=payload, calls=[], validations=[], directories=[],
        download_error=False, validation_error=False,
        metadata={
            "tag_name": "yamnet-v1", "draft": False,
            "assets": [{
                "name": "yamnet.tflite", "state": "uploaded", "size": len(payload),
                "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }],
        },
    )

    def run_gh(arguments):
        state.calls.append(arguments)
        if arguments[:2] == ["repo", "view"]:
            return "team/project\n"
        if arguments[0] == "api":
            return json.dumps(state.metadata)
        if arguments[:2] == ["release", "download"]:
            directory = Path(arguments[arguments.index("--dir") + 1])
            state.directories.append(directory)
            (directory / "yamnet.tflite").write_bytes(state.download_bytes)
            if state.download_error:
                raise RuntimeError("connection interrupted")
            return ""
        pytest.fail(f"Unexpected GitHub command: {arguments}")

    def validate(path):
        state.validations.append(path)
        assert path.read_bytes() == state.payload
        if state.validation_error:
            raise ValueError("incompatible model signature")

    monkeypatch.setattr(download_models, "_run_gh", run_gh)
    monkeypatch.setattr(download_models, "_validate_model", validate)
    return state


def install_existing(state, payload):
    destination = Path(state.settings.model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    destination.chmod(0o644)
    return destination


def download_calls(state):
    return [call for call in state.calls if call[:2] == ["release", "download"]]


def test_downloads_verifies_and_installs_readable_model(github):
    destination = download_models.download_yamnet(github.settings)

    assert destination == Path(github.settings.model_path)
    assert destination.read_bytes() == github.payload
    assert destination.stat().st_mode & 0o777 == 0o644
    assert len(download_calls(github)) == len(github.validations) == 1
    assert download_calls(github)[0][-2:] == ["--", "yamnet-v1"]
    assert not github.directories[0].exists()
    assert list(destination.parent.iterdir()) == [destination]


def test_matching_installed_model_is_checked_without_downloading_again(github):
    destination = install_existing(github, github.payload)

    assert download_models.download_yamnet(github.settings) == destination

    assert not download_calls(github)
    assert github.validations == [destination]


def test_matching_private_file_is_made_readable_for_the_service(github):
    destination = install_existing(github, github.payload)
    destination.chmod(0o600)

    download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == github.payload
    assert destination.stat().st_mode & 0o777 == 0o644
    assert not download_calls(github)


def test_corrupt_existing_model_is_replaced_even_if_its_size_matches(github):
    destination = install_existing(github, b"x" * len(github.payload))

    download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == github.payload
    assert len(download_calls(github)) == 1


def test_changed_release_downloads_its_model_instead_of_reusing_the_old_one(github):
    destination = install_existing(github, github.payload)
    github.settings = replace(github.settings, release="yamnet/v2")
    github.payload = github.download_bytes = b"new-model"
    github.metadata["tag_name"] = "yamnet/v2"
    github.metadata["assets"][0].update(
        size=len(github.payload), digest="sha256:" + hashlib.sha256(github.payload).hexdigest(),
    )

    download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == github.payload
    assert ["api", "repos/team/project/releases/tags/yamnet%2Fv2"] in github.calls
    assert download_calls(github)[0][-1] == "yamnet/v2"


@pytest.mark.parametrize("problem", [
    "draft", "wrong-tag", "missing-asset", "duplicate-asset", "not-uploaded",
    "empty-size", "missing-digest", "wrong-digest-format",
])
def test_invalid_release_metadata_preserves_the_installed_model(github, problem):
    destination = install_existing(github, b"previous-model")
    asset = github.metadata["assets"][0]
    if problem == "draft":
        github.metadata["draft"] = True
    elif problem == "wrong-tag":
        github.metadata["tag_name"] = "another-release"
    elif problem == "missing-asset":
        github.metadata["assets"] = []
    elif problem == "duplicate-asset":
        github.metadata["assets"].append(asset.copy())
    elif problem == "not-uploaded":
        asset["state"] = "new"
    elif problem == "empty-size":
        asset["size"] = 0
    elif problem == "missing-digest":
        asset["digest"] = None
    else:
        asset["digest"] = "not-a-sha256-digest"

    with pytest.raises(ValueError):
        download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == b"previous-model"
    assert not download_calls(github)


def test_interrupted_download_preserves_old_model_and_cleans_staging(github):
    destination = install_existing(github, b"previous-model")
    github.download_bytes = b"partial"
    github.download_error = True

    with pytest.raises(RuntimeError, match="connection interrupted"):
        download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == b"previous-model"
    assert not github.directories[0].exists()
    assert not github.validations


@pytest.mark.parametrize("payload", [b"truncated", b"x" * len(b"verified-model")])
def test_size_or_fingerprint_mismatch_is_rejected_before_loading_model(github, payload):
    destination = install_existing(github, b"previous-model")
    github.download_bytes = payload

    with pytest.raises(ValueError, match="size/SHA-256 differ"):
        download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == b"previous-model"
    assert not github.validations
    assert not github.directories[0].exists()


def test_unusable_model_preserves_the_previous_model(github):
    destination = install_existing(github, b"previous-model")
    github.validation_error = True

    with pytest.raises(ValueError, match="incompatible model signature"):
        download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == b"previous-model"
    assert not github.directories[0].exists()


def test_failed_copy_preserves_the_previous_model(github, monkeypatch):
    destination = install_existing(github, b"previous-model")

    def failed_copy(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(download_models.shutil, "copyfile", failed_copy)

    with pytest.raises(OSError, match="disk full"):
        download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == b"previous-model"
    assert list(destination.parent.iterdir()) == [destination]


def test_failed_atomic_replacement_keeps_the_previous_model(github, monkeypatch):
    destination = install_existing(github, b"previous-model")
    original_replace = Path.replace

    def failed_replace(source, target):
        if Path(target) == destination:
            raise OSError("replace failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", failed_replace)

    with pytest.raises(OSError, match="replace failed"):
        download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == b"previous-model"
    assert list(destination.parent.iterdir()) == [destination]


def test_privileged_install_stages_readable_model_then_renames_it(tmp_path, monkeypatch):
    source = tmp_path / "download.tflite"
    source.write_bytes(b"model")
    destination = tmp_path / "protected" / "yamnet.tflite"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(download_models.subprocess, "run", run)

    download_models._install_model(source, destination, sudo_install=True)

    assert len(commands) == 2
    assert commands[0][:6] == ["sudo", "install", "-D", "-m", "0644", "--"]
    assert commands[0][-2] == str(source)
    staged = commands[0][-1]
    assert Path(staged).parent == destination.parent
    assert commands[1] == ["sudo", "mv", "-f", "--", staged, str(destination)]


def test_failed_privileged_install_cleans_up_without_replacing_model(tmp_path, monkeypatch):
    source = tmp_path / "download.tflite"
    source.write_bytes(b"model")
    destination = tmp_path / "yamnet.tflite"
    destination.write_bytes(b"previous-model")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["sudo", "mv"]:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(download_models.subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        download_models._install_model(source, destination, sudo_install=True)

    assert commands[-1][:5] == ["sudo", "-n", "rm", "-f", "--"]
    assert destination.read_bytes() == b"previous-model"


@pytest.mark.parametrize("problem", ["missing", "authentication", "timeout"])
def test_github_cli_failures_have_clear_messages(monkeypatch, problem):
    def run(command, **kwargs):
        if problem == "missing":
            raise FileNotFoundError("gh")
        if problem == "authentication":
            raise subprocess.CalledProcessError(1, command, stderr="not logged in")
        raise subprocess.TimeoutExpired(command, 300)

    monkeypatch.setattr(download_models.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="GitHub"):
        download_models._run_gh(["repo", "view"])


def test_invalid_repository_is_rejected_before_any_github_calls(github):
    with pytest.raises(ValueError, match="OWNER/REPO"):
        download_models.download_yamnet(github.settings, repository="--not-a-repository")
    assert not github.calls


@pytest.mark.parametrize("flags", [[], ["--sudo-install"]])
def test_cli_uses_configuration_and_an_explicit_local_output(tmp_path, monkeypatch, flags):
    captured = []
    output = tmp_path / "local" / "yamnet.tflite"

    def download(settings, **kwargs):
        captured.append((settings, kwargs))
        return Path(settings.model_path)

    monkeypatch.setattr(download_models, "download_yamnet", download)

    assert download_models.main([
        "--config", str(EXAMPLE_CONFIG), "--repo", "team/project", "--output", str(output), *flags,
    ]) == 0

    settings, options = captured[0]
    assert settings.release == "yamnet-v1" and settings.model_path == str(output.resolve())
    assert options == {"repository": "team/project", "sudo_install": bool(flags)}
    assert not output.exists()


def test_cli_reports_failure_without_installing_a_model(github, capsys):
    destination = install_existing(github, b"previous-model")
    github.download_error = True

    with pytest.raises(SystemExit) as error:
        download_models.main(["--config", str(EXAMPLE_CONFIG), "--output", str(destination)])

    assert error.value.code == 1
    assert "Model installation failed" in capsys.readouterr().err
    assert destination.read_bytes() == b"previous-model"


@pytest.mark.skipif(
    not os.environ.get("E3_YAMNET_MODEL"),
    reason="Set E3_YAMNET_MODEL to check validation with a real local model",
)
def test_real_local_yamnet_passes_installer_validation():
    download_models._validate_model(Path(os.environ["E3_YAMNET_MODEL"]))
