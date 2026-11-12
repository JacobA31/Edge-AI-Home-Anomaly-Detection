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

from e3_edge.config import Config, FallClassifierConfig, YAMNetConfig

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
        payload=payload,
        download_bytes=payload,
        calls=[],
        validations=[],
        directories=[],
        download_error=False,
        validation_error=False,
        metadata={
            "tag_name": "yamnet-v1",
            "draft": False,
            "assets": [
                {
                    "name": "yamnet.tflite",
                    "state": "uploaded",
                    "size": len(payload),
                    "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                }
            ],
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
        size=len(github.payload),
        digest="sha256:" + hashlib.sha256(github.payload).hexdigest(),
    )

    download_models.download_yamnet(github.settings)

    assert destination.read_bytes() == github.payload
    assert ["api", "repos/team/project/releases/tags/yamnet%2Fv2"] in github.calls
    assert download_calls(github)[0][-1] == "yamnet/v2"


@pytest.mark.parametrize(
    "problem",
    [
        "draft",
        "wrong-tag",
        "missing-asset",
        "duplicate-asset",
        "not-uploaded",
        "empty-size",
        "missing-digest",
        "wrong-digest-format",
    ],
)
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

    assert (
        download_models.main(
            [
                "--config",
                str(EXAMPLE_CONFIG),
                "--repo",
                "team/project",
                "--output",
                str(output),
                *flags,
            ]
        )
        == 0
    )

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


@pytest.fixture
def model_releases(tmp_path, monkeypatch):
    yamnet = b"new-yamnet"
    classifier = b"new-fall-classifier"
    metadata = {
        "model_sha256": hashlib.sha256(classifier).hexdigest(),
        "yamnet_sha256": hashlib.sha256(yamnet).hexdigest(),
    }
    state = SimpleNamespace(
        settings=Config(
            device_id="test-pi",
            room="living room",
            backend_url="https://example.invalid",
            function_key="test",
            queue_db=str(tmp_path / "queue.sqlite"),
            yamnet=YAMNetConfig(model_path=str(tmp_path / "installed" / "yamnet.tflite")),
            fall_classifier=FallClassifierConfig(
                model_path=str(tmp_path / "installed" / "fall_classifier.tflite"),
            ),
        ),
        payloads={
            "yamnet-v1": {"yamnet.tflite": yamnet},
            "fall-classifier-v1": {
                "fall_classifier.tflite": classifier,
                "fall_classifier.json": json.dumps(metadata).encode(),
            },
        },
        calls=[],
        corrupt=None,
        interrupt=False,
        invalid_model=False,
        hide_metadata=False,
        missing_digest=False,
        validations=[],
    )

    def github_call(arguments):
        state.calls.append(arguments)
        if arguments[:2] == ["repo", "view"]:
            return "team/project\n"
        if arguments[0] == "api":
            tag = arguments[1].rsplit("/", 1)[-1]
            assets = []
            for name, payload in state.payloads[tag].items():
                if state.hide_metadata and name == "fall_classifier.json":
                    continue
                digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                assets.append(
                    {
                        "name": name,
                        "state": "uploaded",
                        "size": len(payload),
                        "digest": None if state.missing_digest and name.endswith(".json") else digest,
                    }
                )
            return json.dumps({"tag_name": tag, "draft": False, "assets": assets})
        if arguments[:2] == ["release", "download"]:
            name = arguments[arguments.index("--pattern") + 1]
            if state.interrupt and name == "fall_classifier.json":
                raise RuntimeError("download interrupted")
            target = Path(arguments[arguments.index("--dir") + 1]) / name
            payload = state.payloads[arguments[-1]][name]
            target.write_bytes(b"corrupt" if state.corrupt == name else payload)
            return ""
        pytest.fail(f"Unexpected GitHub command: {arguments}")

    def check_yamnet(path):
        assert path.read_bytes() == yamnet
        state.validations.append("yamnet")

    class FakeFallRunner:
        def __init__(self, path):
            settings = json.loads(path.with_suffix(".json").read_text())
            if settings["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("model and metadata differ")
            self.yamnet_sha256 = settings["yamnet_sha256"]

        def predict(self, embedding):
            if state.invalid_model:
                raise ValueError("inference failed")
            assert embedding.shape == (1024,)
            state.validations.append("fall")
            return 0.2

    monkeypatch.setattr(download_models, "_run_gh", github_call)
    monkeypatch.setattr(download_models, "_validate_model", check_yamnet)
    monkeypatch.setattr(download_models, "FallClassifierRunner", FakeFallRunner)
    return state


def model_destinations(state):
    return [
        Path(state.settings.yamnet.model_path),
        Path(state.settings.fall_classifier.model_path),
        Path(state.settings.fall_classifier.metadata_path),
    ]


def install_old_set(state):
    before = {}
    for path in model_destinations(state):
        path.parent.mkdir(parents=True, exist_ok=True)
        before[path] = ("previous-" + path.name).encode()
        path.write_bytes(before[path])
    return before


def test_complete_install_downloads_and_verifies_all_three_files(model_releases):
    state = model_releases
    paths = download_models.download_models(state.settings)
    assert paths == tuple(model_destinations(state)[:2])
    for path in model_destinations(state):
        tag = "yamnet-v1" if path.name == "yamnet.tflite" else "fall-classifier-v1"
        assert path.read_bytes() == state.payloads[tag][path.name]
        assert path.stat().st_mode & 0o777 == 0o644
    assert state.validations == ["yamnet", "fall"]
    assert len(download_calls(state)) == 3


def test_complete_install_reuses_matching_files_without_downloading(model_releases):
    state = model_releases
    download_models.download_models(state.settings)
    state.calls.clear()
    state.validations.clear()
    download_models.download_models(state.settings)
    assert not download_calls(state)
    assert state.validations == ["yamnet", "fall"]


@pytest.mark.parametrize(
    "problem",
    [
        "download",
        "corrupt_model",
        "corrupt_metadata",
        "mismatched_pair",
        "yamnet_version",
        "inference",
        "missing_metadata",
        "missing_digest",
    ],
)
def test_failed_model_set_validation_preserves_every_installed_file(model_releases, problem):
    state = model_releases
    before = install_old_set(state)
    if problem == "download":
        state.interrupt = True
    elif problem.startswith("corrupt"):
        state.corrupt = "fall_classifier.tflite" if problem == "corrupt_model" else "fall_classifier.json"
    elif problem in {"mismatched_pair", "yamnet_version"}:
        metadata = json.loads(state.payloads["fall-classifier-v1"]["fall_classifier.json"])
        metadata["model_sha256" if problem == "mismatched_pair" else "yamnet_sha256"] = "0" * 64
        state.payloads["fall-classifier-v1"]["fall_classifier.json"] = json.dumps(metadata).encode()
    elif problem == "inference":
        state.invalid_model = True
    elif problem == "missing_metadata":
        state.hide_metadata = True
    else:
        state.missing_digest = True
    with pytest.raises((ValueError, RuntimeError)):
        download_models.download_models(state.settings)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("privileged", [True, False])
def test_install_failure_restores_previous_files_or_removes_new_ones(
    model_releases,
    monkeypatch,
    existing,
    privileged,
):
    state = model_releases
    before = install_old_set(state) if existing else {}
    install = download_models._install_model
    failed = False
    install_options = []

    def fail_metadata_once(source, destination, **options):
        nonlocal failed
        install_options.append(options["sudo_install"])
        if destination.name == "fall_classifier.json" and not failed:
            failed = True
            raise OSError("disk full during metadata installation")
        # Simulate filesystem effects without ever invoking real sudo in tests.
        install(source, destination, sudo_install=False)

    def privileged_remove(command, **options):
        assert command[:4] == ["sudo", "rm", "-f", "--"]
        Path(command[-1]).unlink(missing_ok=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(download_models, "_install_model", fail_metadata_once)
    monkeypatch.setattr(download_models.subprocess, "run", privileged_remove)
    with pytest.raises(OSError, match="disk full"):
        download_models.download_models(state.settings, sudo_install=privileged)
    assert install_options and all(option == privileged for option in install_options)
    for path in model_destinations(state):
        if existing:
            assert path.read_bytes() == before[path]
        else:
            assert not path.exists()


def test_classifier_only_install_checks_existing_yamnet(model_releases):
    state = model_releases
    yamnet = Path(state.settings.yamnet.model_path)
    yamnet.parent.mkdir(parents=True)
    yamnet.write_bytes(state.payloads["yamnet-v1"]["yamnet.tflite"])
    destination = download_models.download_fall_classifier(state.settings.fall_classifier, yamnet_path=yamnet)
    assert destination == Path(state.settings.fall_classifier.model_path)
    assert len(download_calls(state)) == 2
    assert yamnet.read_bytes() == state.payloads["yamnet-v1"]["yamnet.tflite"]


def test_cli_default_uses_the_complete_installer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        download_models, "download_models", lambda settings, **options: calls.append(settings)
    )
    assert download_models.main(["--config", str(EXAMPLE_CONFIG)]) == 0
    assert calls[0].fall_classifier.release == "fall-classifier-v1"


def test_cli_classifier_output_keeps_the_yamnet_location(tmp_path, monkeypatch):
    calls = []
    output = tmp_path / "fall_classifier.tflite"
    monkeypatch.setattr(
        download_models,
        "download_fall_classifier",
        lambda settings, **options: calls.append((settings, options)),
    )
    assert (
        download_models.main(
            [
                "--config",
                str(EXAMPLE_CONFIG),
                "--model",
                "fall-classifier",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls[0][0].model_path == str(output)
    assert calls[0][1]["yamnet_path"] == Path("/var/lib/e3/models/yamnet.tflite")


@pytest.mark.skipif(
    not (os.environ.get("E3_YAMNET_MODEL") and os.environ.get("E3_FALL_CLASSIFIER_MODEL")),
    reason="Set both model paths to check the real local classifier/YAMNet pair",
)
def test_real_local_classifier_and_yamnet_pass_installer_validation():
    download_models._validate_fall_classifier(
        Path(os.environ["E3_FALL_CLASSIFIER_MODEL"]),
        Path(os.environ["E3_YAMNET_MODEL"]),
    )
