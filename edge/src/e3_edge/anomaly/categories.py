"""Household sound categories built from YAMNet's 521 class scores.

The behavioural detectors (running water, impacts, activity) need to know *what kind* of sound
is present, not just how unusual it is. YAMNet already scores the AudioSet classes we need, so a
category's score is the highest score among its member classes.

yamnet_class_map.csv is the official class map (tensorflow/models, research/audioset/yamnet,
Apache-2.0). Classes are referred to by display name so the mapping can be read and reviewed.
On/off thresholds are starting points: tune on SINS.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CategoryOverride
from .frames import NUM_CLASSES

CLASS_MAP = Path(__file__).with_name("yamnet_class_map.csv")
SILENCE = "Silence"


@dataclass(frozen=True)
class CategorySpec:
    classes: tuple[str, ...]
    on: float = 0.2  # an episode starts when the category score reaches this
    off: float = 0.1  # ...and continues while it stays at or above this
    gap_seconds: float = 2.0  # drop-outs shorter than this don't end an episode


DEFAULT_CATEGORIES: dict[str, CategorySpec] = {
    "water": CategorySpec(
        ("Water tap, faucet", "Sink (filling or washing)", "Bathtub (filling or washing)", "Water",
         "Pour", "Trickle, dribble", "Gush", "Fill (with liquid)"),
        gap_seconds=5.0,
    ),
    "impact": CategorySpec(
        ("Thump, thud", "Thunk", "Bang", "Slam", "Smash, crash", "Breaking", "Whack, thwack",
         "Shatter", "Clatter"),
        gap_seconds=1.0,
    ),
    "footsteps": CategorySpec(("Walk, footsteps", "Shuffle")),
    "speech": CategorySpec(
        ("Speech", "Child speech, kid speaking", "Conversation", "Hubbub, speech noise, speech babble")
    ),
    "household": CategorySpec(
        ("Door", "Sliding door", "Cupboard open or close", "Drawer open or close",
         "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)", "Frying (food)",
         "Microwave oven", "Blender", "Vacuum cleaner", "Hair dryer", "Toilet flush", "Boiling")
    ),
    "media": CategorySpec(("Television", "Radio", "Music")),
    "alarm": CategorySpec(("Alarm", "Smoke detector, smoke alarm", "Fire alarm", "Alarm clock")),
}


@lru_cache(maxsize=1)
def class_names() -> tuple[str, ...]:
    with CLASS_MAP.open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = tuple(r["display_name"] for r in sorted(rows, key=lambda r: int(r["index"])))
    if len(names) != NUM_CLASSES:
        raise RuntimeError(f"{CLASS_MAP.name} has {len(names)} classes, expected {NUM_CLASSES}")
    return names


def resolve(overrides: Mapping[str, CategoryOverride] | None = None) -> dict[str, CategorySpec]:
    """Default categories with config overrides applied (an override may also add a category)."""
    specs = dict(DEFAULT_CATEGORIES)
    for name, o in (overrides or {}).items():
        base = specs.get(name)
        if base is None and not o.classes:
            raise ValueError(f"new category {name!r} needs a list of classes")
        base = base or CategorySpec(())
        specs[name] = CategorySpec(
            tuple(o.classes) if o.classes else base.classes,
            base.on if o.on is None else o.on,
            base.off if o.off is None else o.off,
            base.gap_seconds if o.gap_seconds is None else o.gap_seconds,
        )
    return specs


class CategoryMapper:
    """Turns a frame's 521 YAMNet scores into one score per category."""

    def __init__(self, specs: Mapping[str, CategorySpec]):
        index = {name: i for i, name in enumerate(class_names())}
        unknown = sorted({c for s in specs.values() for c in s.classes} - set(index))
        if unknown:
            raise ValueError(f"not YAMNet class names: {unknown}")
        for name, spec in specs.items():
            if not spec.classes or not 0 <= spec.off <= spec.on <= 1:
                raise ValueError(f"category {name!r}: needs classes and 0 <= off <= on <= 1")
        self.names = tuple(specs)
        self.specs = dict(specs)
        self._members = [np.array([index[c] for c in s.classes]) for s in specs.values()]
        self.silence_index = index[SILENCE]

    def __len__(self) -> int:
        return len(self.names)

    def scores(self, class_scores: np.ndarray) -> np.ndarray:
        return np.array([class_scores[m].max() for m in self._members], dtype=np.float32)

    def dominant(self, class_scores: np.ndarray) -> str | None:
        """The strongest category at or above its `on` threshold in this frame, if any."""
        best, name = 0.0, None
        for category, score in zip(self.names, self.scores(class_scores), strict=True):
            if score >= self.specs[category].on and score > best:
                best, name = float(score), category
        return name

    def scores_batch(self, class_scores: np.ndarray) -> np.ndarray:
        """[N, 521] -> [N, categories]."""
        return np.stack([class_scores[:, m].max(axis=1) for m in self._members], axis=1)
