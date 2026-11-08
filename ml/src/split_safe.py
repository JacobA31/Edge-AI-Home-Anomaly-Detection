"""Assign SAFE recordings to training, validation, and test sets without moving audio.

Run from ml/:
    python -m src.split_safe data/safe/inventory.csv

Writes splits.csv beside the inventory, with path,label,split columns. Paths
remain relative to that folder. The inventory and recordings stay unchanged.

SAFE's author documents AA-BBB-CC-DDD-FF filenames: AA is a supplied fold,
CC is the environment, DDD is the original video sequence, and FF is the class.
Source: https://www.kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset

Defaults keep folds 01-07 in train, 08 in validation, and 09-10 in test:
70/10/20 by recording count for the full dataset. Exact 70/15/15 would require
dividing the supplied folds. Assignments are fixed, not randomly regenerated.
Known source clips (class, environment, sequence) must stay in one split.
This is not an unseen-environment assessment, nor a check of background-mix provenance.
"""

import argparse
import csv
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from .index_safe import LABELS

DEFAULT_TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7)
DEFAULT_VALIDATION_FOLDS = (8,)
DEFAULT_TEST_FOLDS = (9, 10)
FILENAME = re.compile(
    r"(?P<fold>0[1-9]|10)-\d{3}-(?P<environment>\d{2})-(?P<source>\d{3})-(?P<class>01|02)",
)


def _fold_assignments(groups: dict[str, tuple[int, ...]]) -> dict[int, str]:
    assignments = {}
    for split, folds in groups.items():
        if not folds:
            raise ValueError(f"{split} must have at least one fold")
        for fold in folds:
            if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, 11):
                raise ValueError("Fold numbers must be integers from 1 to 10")
            if fold in assignments:
                raise ValueError(f"Fold {fold:02d} is assigned more than once")
            assignments[fold] = split
    missing = set(range(1, 11)) - assignments.keys()
    if missing:
        raise ValueError(f"Every supplied fold must be assigned; missing: {sorted(missing)}")
    return assignments


def split_safe(
    inventory_path: Path, *, train_folds: tuple[int, ...] = DEFAULT_TRAIN_FOLDS,
    validation_folds: tuple[int, ...] = DEFAULT_VALIDATION_FOLDS,
    test_folds: tuple[int, ...] = DEFAULT_TEST_FOLDS,
) -> Path:
    """Validate the inventory and write one deterministic split assignment per recording."""
    groups = {"train": train_folds, "validation": validation_folds, "test": test_folds}
    assignments = _fold_assignments(groups)
    inventory_path = Path(inventory_path).expanduser().resolve()
    root = inventory_path.parent
    rows = []
    seen_files = set()
    source_splits = {}
    counts = {split: {"fall": 0, "non_fall": 0} for split in groups}

    with inventory_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != 2 or set(reader.fieldnames) != {"path", "label"}:
            raise ValueError("Expected an inventory with path,label columns; create it with src.index_safe")
        for line, row in enumerate(reader, start=2):
            path_text = (row.get("path") or "").strip()
            label = (row.get("label") or "").strip()
            if None in row or not path_text or not label:
                raise ValueError(f"Invalid inventory row {line}: expected one path and label")
            relative = Path(path_text)
            path = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
                raise ValueError(f"Row {line}: recording paths must stay inside the inventory folder")
            if not path.is_file() or relative.suffix.lower() != ".wav":
                raise ValueError(f"Row {line}: WAV file not found: {path_text}")
            if path in seen_files:
                raise ValueError(f"Recording appears more than once in the inventory: {path_text}")
            seen_files.add(path)
            metadata = FILENAME.fullmatch(relative.stem)
            if metadata is None:
                raise ValueError(f"Unexpected SAFE filename: {path_text}; expected AA-BBB-CC-DDD-FF.wav")
            if label != LABELS[metadata["class"]]:
                raise ValueError(f"Row {line}: label does not match the filename: {path_text}")

            split = assignments[int(metadata["fold"])]
            # Sequence numbers are scoped by recording environment and class.
            source_group = (metadata["class"], metadata["environment"], metadata["source"])
            previous_split = source_splits.setdefault(source_group, split)
            if previous_split != split:
                raise ValueError(
                    f"Source clip {source_group} would appear in both {previous_split} and {split}; "
                    "assign its supplied folds to the same split",
                )
            rows.append((relative.as_posix(), label, split))
            counts[split][label] += 1

    if not rows:
        raise ValueError("The inventory contains no recordings")
    for split, labels in counts.items():
        if any(count == 0 for count in labels.values()):
            raise ValueError(f"{split} must contain both fall and non_fall recordings")

    destination = root / "splits.csv"
    # Validate everything before touching the old output, then replace it only
    # once a complete CSV exists. Original recordings and inventory are untouched.
    with TemporaryDirectory(dir=root, prefix=".safe-split-") as temporary:
        staged = Path(temporary) / "splits.csv"
        with staged.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["path", "label", "split"])
            writer.writerows(sorted(rows))
        staged.replace(destination)

    print(f"Recordings: {len(rows)}")
    for split, labels in counts.items():
        total = sum(labels.values())
        folds = ", ".join(f"{fold:02d}" for fold in sorted(groups[split]))
        print(
            f"{split}: {total} ({labels['fall']} falls, {labels['non_fall']} non-falls; "
            f"{100 * total / len(rows):.1f}%) | folds: {folds}",
        )
    print(f"Splits: {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inventory", type=Path, help="SAFE inventory.csv produced by src.index_safe")
    for name, default in (
        ("train", DEFAULT_TRAIN_FOLDS), ("validation", DEFAULT_VALIDATION_FOLDS), ("test", DEFAULT_TEST_FOLDS),
    ):
        parser.add_argument(
            f"--{name}-folds", type=int, nargs="+", choices=range(1, 11), default=default,
            help=f"Whole folds for {name} (default: {' '.join(map(str, default))})",
        )
    args = parser.parse_args(argv)
    try:
        split_safe(
            args.inventory, train_folds=tuple(args.train_folds),
            validation_folds=tuple(args.validation_folds), test_folds=tuple(args.test_folds),
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
