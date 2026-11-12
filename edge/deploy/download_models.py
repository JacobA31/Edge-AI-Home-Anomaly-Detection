"""Install the configured YAMNet and fall-classifier release assets before restart.

Run from edge/, using its environment with Pi or development dependencies:
    python deploy/download_models.py --config config.yaml
    python deploy/download_models.py --model yamnet --output /tmp/e3-model/yamnet.tflite

Requires GitHub CLI (gh), authenticated for this repository. The repository is
resolved from this clone unless --repo OWNER/REPO is supplied. Publish an asset
named yamnet.tflite under yamnet.release, and fall_classifier.tflite plus
fall_classifier.json under fall_classifier.release. The default installs all
three files, checking them together before changing any installed files.

The normal mode writes only to a destination writable by the current user.
update.sh uses --sudo-install for installation into the Pi's protected directory;
GitHub access and model verification still run as the invoking user. This helper
does not restart services or update Git. A failed download/validation preserves
the installed files. GitHub's release metadata must include a SHA-256 digest for
each asset. Installation errors trigger restoration of the previous files.

--model selects one component for focused checks. For backward compatibility,
--output alone still selects YAMNet; with --model fall-classifier it overrides
the classifier destination. The default all-model installation uses YAML paths.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from e3_edge import config
from e3_edge.audio.yamnet import EMBEDDING_SIZE, WINDOW_SAMPLES, YAMNetRunner
from e3_edge.fall_classifier import FallClassifierRunner

EDGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ASSET = "yamnet.tflite"
FALL_MODEL_ASSET = "fall_classifier.tflite"
FALL_METADATA_ASSET = "fall_classifier.json"


@dataclass(frozen=True)
class ReleaseAsset:
    size: int
    sha256: str


def _run_gh(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["gh", *arguments],
            cwd=EDGE_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Install GitHub CLI (gh) and authenticate it for this repository") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        detail = detail or "check gh authentication and repository access"
        raise RuntimeError(f"GitHub request failed: {detail}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("GitHub request timed out; the installed model was not replaced") from error
    return result.stdout


def _repository(repository: str | None) -> str:
    arguments = ["repo", "view"]
    if repository is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("--repo must use OWNER/REPO format")
        arguments.append(repository)
    arguments.extend(["--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    resolved = _run_gh(arguments).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", resolved):
        raise ValueError("GitHub CLI did not resolve an OWNER/REPO repository")
    return resolved


def _release_asset(repository: str, release: str, asset_name: str = MODEL_ASSET) -> ReleaseAsset:
    endpoint = f"repos/{repository}/releases/tags/{quote(release, safe='')}"
    metadata = json.loads(_run_gh(["api", endpoint]))
    if not isinstance(metadata, dict) or metadata.get("draft") or metadata.get("tag_name") != release:
        raise ValueError(f"Expected published GitHub Release {release}")
    assets = metadata.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("GitHub returned invalid release assets")
    matching = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == asset_name]
    if len(matching) != 1 or matching[0].get("state") != "uploaded":
        raise ValueError(f"Release {release} needs one uploaded asset named {asset_name}")
    asset = matching[0]
    size = asset.get("size")
    digest = asset.get("digest")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("GitHub reported an invalid model file size")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        raise ValueError(
            f"Release asset needs GitHub SHA-256 metadata; re-upload {asset_name} if it is missing",
        )
    return ReleaseAsset(size=size, sha256=digest.split(":", 1)[1].lower())


def _matches_asset(path: Path, asset: ReleaseAsset) -> bool:
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == asset.sha256


def _validate_model(path: Path) -> None:
    # The runner checks the signature, dimensions, data types, finite outputs,
    # and class-score range. A real invocation also rejects unusable models.
    runner = YAMNetRunner(path)
    runner.extract_features(np.zeros(WINDOW_SAMPLES, dtype=np.float32))


def _validate_fall_classifier(path: Path, yamnet_path: Path) -> None:
    runner = FallClassifierRunner(path)
    with yamnet_path.open("rb") as source:
        yamnet_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if runner.yamnet_sha256 != yamnet_sha256:
        raise ValueError("Fall classifier was trained with a different YAMNet model; nothing was installed")
    runner.predict(np.zeros(EMBEDDING_SIZE, dtype=np.float32))


def _install_model(source: Path, destination: Path, *, sudo_install: bool) -> None:
    """Stage beside the destination and rename only after the full copy exists."""
    if sudo_install:
        # GNU install -D creates missing parent directories on Raspberry Pi OS.
        # Keep GitHub credentials out of sudo: only these filesystem operations
        # are privileged. 0644 makes the model readable by the e3 service user.
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        installed = False
        try:
            subprocess.run(
                ["sudo", "install", "-D", "-m", "0644", "--", str(source), str(temporary)],
                check=True,
            )
            subprocess.run(["sudo", "mv", "-f", "--", str(temporary), str(destination)], check=True)
            installed = True
        finally:
            if not installed:
                # Cleanup must not prompt again or hide the original failure.
                with suppress(OSError, subprocess.SubprocessError):
                    subprocess.run(
                        ["sudo", "-n", "rm", "-f", "--", str(temporary)],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=".yamnet-",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.chmod(0o644)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _stage_yamnet(settings: config.YAMNetConfig, repository: str, directory: Path) -> Path:
    """Return a verified existing/downloaded file, without changing the destination."""
    destination = Path(settings.model_path)
    if destination.exists() and not destination.is_file():
        raise ValueError(f"Model destination is not a file: {destination}")
    asset = _release_asset(repository, settings.release)
    if _matches_asset(destination, asset):
        _validate_model(destination)
        print(
            f"Already installed and verified: {repository} {settings.release} -> {destination}",
            flush=True,
        )
        return destination

    print(f"Downloading {MODEL_ASSET} from {repository} release {settings.release}...", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    _run_gh(
        [
            "release",
            "download",
            "--repo",
            repository,
            "--pattern",
            MODEL_ASSET,
            "--dir",
            str(directory),
            "--",
            settings.release,
        ]
    )
    downloaded = directory / MODEL_ASSET
    if not _matches_asset(downloaded, asset):
        raise ValueError("Downloaded model size/SHA-256 differ from GitHub; installed model was not replaced")
    print("Checking downloaded YAMNet with LiteRT...", flush=True)
    _validate_model(downloaded)
    return downloaded


def download_yamnet(
    settings: config.YAMNetConfig,
    *,
    repository: str | None = None,
    sudo_install: bool = False,
) -> Path:
    """Keep the standalone YAMNet download API for focused checks."""
    repository = _repository(repository)
    destination = Path(settings.model_path)
    with TemporaryDirectory(prefix="e3-yamnet-download-") as temporary:
        source = _stage_yamnet(settings, repository, Path(temporary))
        if source != destination or not (destination.stat().st_mode & 0o004):
            _install_model(source, destination, sudo_install=sudo_install)
    print(f"Installed verified model: {settings.release} -> {destination}", flush=True)
    return destination


def _stage_fall_classifier(
    settings: config.FallClassifierConfig,
    yamnet_path: Path,
    repository: str,
    directory: Path,
) -> tuple[Path, Path]:
    destination, metadata_path = Path(settings.model_path), Path(settings.metadata_path)
    for path in (destination, metadata_path):
        if path.exists() and not path.is_file():
            raise ValueError(f"Classifier destination is not a file: {path}")
    assets = {
        name: _release_asset(repository, settings.release, name)
        for name in (FALL_MODEL_ASSET, FALL_METADATA_ASSET)
    }
    if _matches_asset(destination, assets[FALL_MODEL_ASSET]) and _matches_asset(
        metadata_path, assets[FALL_METADATA_ASSET]
    ):
        _validate_fall_classifier(destination, yamnet_path)
        print(f"Already installed and verified: {settings.release} -> {destination}", flush=True)
        return destination, metadata_path
    directory.mkdir(parents=True, exist_ok=True)
    for name, asset in assets.items():
        print(f"Downloading {name} from {repository} release {settings.release}...", flush=True)
        _run_gh(
            [
                "release",
                "download",
                "--repo",
                repository,
                "--pattern",
                name,
                "--dir",
                str(directory),
                "--",
                settings.release,
            ]
        )
        if not _matches_asset(directory / name, asset):
            raise ValueError(f"Downloaded {name} size/SHA-256 differ from GitHub; installed files unchanged")
    model, metadata = directory / FALL_MODEL_ASSET, directory / FALL_METADATA_ASSET
    print("Checking classifier/settings and YAMNet compatibility with LiteRT...", flush=True)
    _validate_fall_classifier(model, yamnet_path)
    return model, metadata


def _install_files(files: list[tuple[Path, Path]], *, sudo_install: bool) -> None:
    """Install fully validated files, restoring previous contents on installation errors."""
    destinations = [destination.resolve() for _, destination in files]
    if len(set(destinations)) != len(destinations):
        raise ValueError("YAMNet, fall model and metadata destinations must be different files")
    pending = []
    for source, destination in files:
        if destination.exists() and not destination.is_file():
            raise ValueError(f"Model destination is not a file: {destination}")
        if source != destination or not (destination.stat().st_mode & 0o004):
            pending.append((source, destination))
    with TemporaryDirectory(prefix="e3-model-backup-") as temporary:
        backups = {}
        for index, (_, destination) in enumerate(pending):
            backup = Path(temporary) / str(index) if destination.exists() else None
            if backup is not None:
                shutil.copyfile(destination, backup)
            backups[destination] = backup
        attempted = []
        try:
            for source, destination in pending:
                attempted.append(destination)
                _install_model(source, destination, sudo_install=sudo_install)
        except (Exception, KeyboardInterrupt) as error:
            failed = []
            for destination in reversed(attempted):
                try:
                    backup = backups[destination]
                    if backup is not None:
                        _install_model(backup, destination, sudo_install=sudo_install)
                    elif sudo_install:
                        subprocess.run(["sudo", "rm", "-f", "--", str(destination)], check=True)
                    else:
                        destination.unlink(missing_ok=True)
                except (OSError, subprocess.SubprocessError):
                    failed.append(str(destination))
            if failed:
                raise RuntimeError("Installation failed; could not restore: " + ", ".join(failed)) from error
            raise


def download_fall_classifier(
    settings: config.FallClassifierConfig,
    *,
    yamnet_path: Path,
    repository: str | None = None,
    sudo_install: bool = False,
) -> Path:
    """Install the model/settings pair using an already-installed compatible YAMNet."""
    repository = _repository(repository)
    destination, metadata_path = Path(settings.model_path), Path(settings.metadata_path)
    with TemporaryDirectory(prefix="e3-classifier-download-") as temporary:
        model, metadata = _stage_fall_classifier(settings, Path(yamnet_path), repository, Path(temporary))
        _install_files([(model, destination), (metadata, metadata_path)], sudo_install=sudo_install)
    print(f"Installed verified classifier/settings: {settings.release} -> {destination}", flush=True)
    return destination


def download_models(
    settings: config.Config,
    *,
    repository: str | None = None,
    sudo_install: bool = False,
) -> tuple[Path, Path]:
    """Check the complete model set before updating any installed file."""
    repository = _repository(repository)
    yamnet_destination = Path(settings.yamnet.model_path)
    fall_destination = Path(settings.fall_classifier.model_path)
    with TemporaryDirectory(prefix="e3-model-download-") as temporary:
        directory = Path(temporary)
        yamnet = _stage_yamnet(settings.yamnet, repository, directory / "yamnet")
        fall, metadata = _stage_fall_classifier(
            settings.fall_classifier,
            yamnet,
            repository,
            directory / "fall_classifier",
        )
        _install_files(
            [
                (yamnet, yamnet_destination),
                (fall, fall_destination),
                (metadata, Path(settings.fall_classifier.metadata_path)),
            ],
            sudo_install=sudo_install,
        )
    print("Installed and verified YAMNet and fall-classifier model/settings.", flush=True)
    return yamnet_destination, fall_destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=config.DEFAULT_PATH, help="Edge YAML configuration")
    parser.add_argument("--repo", help="GitHub OWNER/REPO (default: this clone's repository)")
    parser.add_argument("--model", choices=("all", "yamnet", "fall-classifier"), help="Default: all models")
    parser.add_argument("--output", type=Path, help="Single-model destination; alone this selects YAMNet")
    parser.add_argument(
        "--sudo-install",
        action="store_true",
        help="Use sudo only for installation on the Pi",
    )
    args = parser.parse_args(argv)
    try:
        settings = config.load(args.config)
        component = args.model or ("yamnet" if args.output is not None else "all")
        options = {"repository": args.repo, "sudo_install": args.sudo_install}
        if component == "all":
            if args.output is not None:
                raise ValueError(
                    "--output requires one model; configure both destinations in YAML for --model all"
                )
            download_models(settings, **options)
        elif component == "yamnet":
            yamnet = settings.yamnet
            if args.output is not None:
                yamnet = replace(yamnet, model_path=str(args.output.resolve()))
            download_yamnet(yamnet, **options)
        else:
            fall = settings.fall_classifier
            if args.output is not None:
                fall = replace(fall, model_path=str(args.output.resolve()))
            download_fall_classifier(fall, yamnet_path=Path(settings.yamnet.model_path), **options)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Model installation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
