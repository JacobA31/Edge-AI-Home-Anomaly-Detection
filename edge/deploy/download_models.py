"""Install the configured YAMNet GitHub Release asset before restarting edge.

Run from edge/, using its environment with Pi or development dependencies:
    python deploy/download_models.py --config config.example.yaml --output /tmp/e3-model/yamnet.tflite

Requires GitHub CLI (gh), authenticated for this repository. The repository is
resolved from this clone unless --repo OWNER/REPO is supplied. Publish an asset
named yamnet.tflite under the configured release tag before using this helper.

The normal mode writes only to a destination writable by the current user.
update.sh uses --sudo-install for installation into the Pi's protected directory;
GitHub access and model verification still run as the invoking user. This helper
does not restart services or update Git. A failed download/validation preserves
the installed model. GitHub's release asset metadata must include a SHA-256 digest.
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
from e3_edge.audio.yamnet import WINDOW_SAMPLES, YAMNetRunner

EDGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ASSET = "yamnet.tflite"


@dataclass(frozen=True)
class ReleaseAsset:
    size: int
    sha256: str


def _run_gh(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["gh", *arguments], cwd=EDGE_ROOT, check=True,
            capture_output=True, text=True, timeout=300,
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


def _release_asset(repository: str, release: str) -> ReleaseAsset:
    endpoint = f"repos/{repository}/releases/tags/{quote(release, safe='')}"
    metadata = json.loads(_run_gh(["api", endpoint]))
    if not isinstance(metadata, dict) or metadata.get("draft") or metadata.get("tag_name") != release:
        raise ValueError(f"Expected published GitHub Release {release}")
    assets = metadata.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("GitHub returned invalid release assets")
    matching = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == MODEL_ASSET]
    if len(matching) != 1 or matching[0].get("state") != "uploaded":
        raise ValueError(f"Release {release} needs one uploaded asset named {MODEL_ASSET}")
    asset = matching[0]
    size = asset.get("size")
    digest = asset.get("digest")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("GitHub reported an invalid model file size")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        raise ValueError(
            "Release asset needs GitHub SHA-256 metadata; re-upload yamnet.tflite if it is missing",
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
                ["sudo", "install", "-D", "-m", "0644", "--", str(source), str(temporary)], check=True,
            )
            subprocess.run(["sudo", "mv", "-f", "--", str(temporary), str(destination)], check=True)
            installed = True
        finally:
            if not installed:
                # Cleanup must not prompt again or hide the original failure.
                with suppress(OSError, subprocess.SubprocessError):
                    subprocess.run(
                        ["sudo", "-n", "rm", "-f", "--", str(temporary)], check=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
                    )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, prefix=".yamnet-", suffix=".tmp", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.chmod(0o644)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_yamnet(
    settings: config.YAMNetConfig, *, repository: str | None = None, sudo_install: bool = False,
) -> Path:
    """Reuse a verified matching model or download, verify, and install its replacement."""
    destination = Path(settings.model_path)
    if destination.exists() and not destination.is_file():
        raise ValueError(f"Model destination is not a file: {destination}")
    repository = _repository(repository)
    asset = _release_asset(repository, settings.release)
    if _matches_asset(destination, asset):
        _validate_model(destination)
        if not (destination.stat().st_mode & 0o004):
            _install_model(destination, destination, sudo_install=sudo_install)
        print(
            f"Already installed and verified: {repository} {settings.release} -> {destination}", flush=True,
        )
        return destination

    print(f"Downloading {MODEL_ASSET} from {repository} release {settings.release}...", flush=True)
    with TemporaryDirectory(prefix="e3-yamnet-download-") as temporary:
        directory = Path(temporary)
        _run_gh([
            "release", "download", "--repo", repository, "--pattern", MODEL_ASSET,
            "--dir", str(directory), "--", settings.release,
        ])
        downloaded = directory / MODEL_ASSET
        if not _matches_asset(downloaded, asset):
            raise ValueError(
                "Downloaded model size/SHA-256 differ from GitHub; installed model was not replaced",
            )
        print("Checking downloaded YAMNet with LiteRT...", flush=True)
        _validate_model(downloaded)
        _install_model(downloaded, destination, sudo_install=sudo_install)
    print(f"Installed verified model: {settings.release} -> {destination}", flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=config.DEFAULT_PATH, help="Edge YAML configuration")
    parser.add_argument("--repo", help="GitHub OWNER/REPO (default: this clone's repository)")
    parser.add_argument("--output", type=Path, help="Override destination for local download verification")
    parser.add_argument(
        "--sudo-install", action="store_true", help="Use sudo only for installation on the Pi",
    )
    args = parser.parse_args(argv)
    try:
        settings = config.load(args.config).yamnet
        if args.output is not None:
            settings = replace(settings, model_path=str(args.output.resolve()))
        download_yamnet(settings, repository=args.repo, sudo_install=args.sudo_install)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Model installation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
