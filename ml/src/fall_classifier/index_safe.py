"""Create a SAFE recording inventory without changing or processing the audio.

Run from ml/:
    python -m src.fall_classifier.index_safe data/safe

Writes inventory.csv in the supplied folder, with path,label columns. Paths are
relative to that folder. Labels describe recordings, not individual audio windows:
-01.wav means fall and -02.wav means non_fall. Subfolders are included.
Re-running replaces the generated inventory after all filenames are validated.
"""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

LABELS = {"01": "fall", "02": "non_fall"}


def index_safe(input_dir: Path) -> Path:
    """Write a sorted recording-to-label table and report the category counts."""
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Expected a SAFE dataset folder: {root}")

    recordings = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not recordings:
        raise ValueError(f"No WAV recordings found in {root}")

    rows = []
    counts = {"fall": 0, "non_fall": 0}
    for path in recordings:
        relative_path = path.relative_to(root).as_posix()
        _, separator, code = path.stem.rpartition("-")
        if not separator or code not in LABELS:
            raise ValueError(
                f"Unexpected SAFE filename: {relative_path}; expected -01.wav or -02.wav",
            )
        label = LABELS[code]
        rows.append((relative_path, label))
        counts[label] += 1

    destination = root / "inventory.csv"
    # Prepare the complete CSV on the same filesystem before replacing the old
    # inventory. Validation or write failures leave the existing table intact.
    with TemporaryDirectory(dir=root, prefix=".safe-index-") as temporary:
        staged = Path(temporary) / "inventory.csv"
        with staged.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["path", "label"])
            writer.writerows(rows)
        staged.replace(destination)

    print(f"Recordings: {len(rows)}")
    print(f"Falls: {counts['fall']}")
    print(f"Non-falls: {counts['non_fall']}")
    print(f"Inventory: {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="SAFE folder containing the WAV recordings")
    args = parser.parse_args(argv)
    try:
        index_safe(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
