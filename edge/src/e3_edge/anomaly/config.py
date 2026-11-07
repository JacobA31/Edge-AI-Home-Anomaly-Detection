"""Settings for the anomaly module.

Every value has a default, so an empty `anomaly:` section in edge/config.yaml works. Durations
are seconds of *observed* audio: time spent muted does not count towards calibration.
The thresholds marked "tune on SINS" are starting points until the SINS replay is run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

DEFAULT_PERIODS = {"night": 0, "morning": 6, "afternoon": 12, "evening": 18}


@dataclass
class EmbeddingConfig:
    # "global":   one Gaussian over all frames (proposal 1.7.3.2 as written).
    # "category": the global Gaussian plus one per household sound category (water, impact, ...),
    #             fitted on the frames where that category is present. A frame's distance is the
    #             smallest of them ("selective Mahalanobis", as in the DCASE Task 2 baseline), so
    #             a rare but familiar sound (the cane thud of proposal 1.7.4) is not flagged just
    #             for being rare, while a sound unlike anything in the home still is.
    # "period":   the global Gaussian plus one per time-of-day period, same min rule.
    contexts: str = "category"
    n_components: int | None = None  # global component: None = exact Mahalanobis on all dims
    component_dims: int = 32  # other components: at most this many PCA dims (fewer frames)
    min_component_frames: int = 50  # a context with fewer frames gets no component of its own
    frame_quantile: float = 0.999  # frames above this quantile of the baseline are "anomalous"
    ridge: float = 1e-6  # added to the diagonal, relative to the mean variance


@dataclass
class CalibrationConfig:
    stats_seconds: float = 3 * 86400  # phase 1: learn the mean and covariance
    threshold_seconds: float = 86400  # phase 2: score unseen frames to set the threshold
    min_frames: int = 2000  # never fit a Gaussian on fewer frames
    min_threshold_frames: int = 500
    batch_size: int = 256
    score_reservoir: int = 50_000
    sample_reservoir: int = 5_000  # cap per learned list (durations, counts, levels)
    checkpoint_seconds: float = 3600  # each checkpoint is up to ~35 MB in category mode
    seed: int = 0


@dataclass
class ActivityConfig:
    margin_db: float = 6.0  # active if louder than the room's noise floor + margin
    noise_floor_quantile: float = 0.10
    default_noise_floor_dbfs: float = -60.0  # used while calibrating
    silence_score: float = 0.5  # fallback when frames carry no level: YAMNet "Silence" >= this


@dataclass
class ProlongedSoundConfig:
    category: str = "water"
    min_episode_seconds: float = 2.0  # shorter episodes are not learned (detector flicker)
    duration_quantile: float = 0.99
    margin: float = 1.5
    min_seconds: float = 120.0
    max_seconds: float = 3600.0
    default_seconds: float = 900.0  # used when the baseline saw fewer than min_episodes
    min_episodes: int = 5
    escalate_factor: float = 2.0  # second, stronger finding at this multiple of the threshold


@dataclass
class ImpactRateConfig:
    category: str = "impact"
    window_seconds: float = 300.0
    # Limit = this quantile of the per-window counts seen in calibration, + 1. The default 1.0
    # means "more impacts in one window than ever seen at that time of day while calibrating";
    # 0.995 would by construction alert on about 1 window in 200 (more than one a day).
    count_quantile: float = 1.0
    # Short calibrations rarely see the worst normal window, so the limit is also raised until
    # a Poisson model of the learned impact rate expects at most this many false alerts per day
    # (per period). Real impacts are burstier than Poisson, hence max(empirical, Poisson).
    target_alerts_per_day: float = 0.05
    min_count: int = 4
    min_windows: int = 12  # per period; with fewer, the pooled threshold is used
    default_count: int = 6
    cooldown_seconds: float = 300.0


@dataclass
class PostImpactConfig:
    # Quiet is measured from when the impact itself is over (impact no longer heard), and at
    # least grace_seconds after it. Evidence therefore comes about impact length + 1 s (episode
    # gap tolerance) + short_seconds after the fall sound: 7-8 s, inside NF1's 10 s budget.
    grace_seconds: float = 1.0
    # Quiet starts at the first quiet frame after the trigger. If none comes this soon (NF1's
    # 10 s fall budget), the room never went quiet, so someone is moving and the watch ends.
    settle_max_seconds: float = 10.0
    short_seconds: float = 5.0
    max_active_fraction: float = 0.2
    quiet_quantile: float = 0.95
    margin: float = 1.5
    min_seconds: float = 60.0
    max_seconds: float = 3600.0
    default_seconds: float = 180.0
    min_samples: int = 10
    category_strength: float = 0.3  # confidence when the trigger is a familiar impact
    merge_seconds: float = 2.0  # a fall candidate this close to an impact upgrades it
    # The learned limit describes quiet after *ordinary* impacts (a door, a dropped spoon), which
    # in a home where someone lives alone can be long. After a strong trigger (fall candidate or
    # an impact unlike the baseline) the limit is capped at strong_max_seconds instead.
    strong_level: float = 0.5
    strong_max_seconds: float = 120.0


@dataclass
class SustainedConfig:
    window_seconds: float = 30.0
    min_fraction: float = 0.5
    min_frames: int = 10
    cooldown_seconds: float = 300.0


@dataclass
class AdaptationConfig:
    enabled: bool = False  # proposal 1.7.3.3: the baseline stays static during evaluation
    interval_seconds: float = 7 * 86400
    weight: float = 0.2  # share of the new data in the blended baseline
    gate_level: float = 0.99  # frames above this baseline level are not learned
    holdoff_seconds: float = 600.0  # nothing is learned this long after an alert


@dataclass
class CategoryOverride:
    classes: list[str] | None = None
    on: float | None = None
    off: float | None = None
    gap_seconds: float | None = None


@dataclass
class AnomalyConfig:
    timezone: str = "UTC"
    periods: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PERIODS))
    frame_seconds: float = 0.96  # YAMNet window length
    max_gap_seconds: float = 5.0  # a longer gap between frames (mute, restart) breaks continuity
    baseline_path: str = "/var/lib/e3/baseline.npz"
    checkpoint_path: str = "/var/lib/e3/calibration.npz"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    categories: dict[str, CategoryOverride] = field(default_factory=dict)
    water: ProlongedSoundConfig = field(default_factory=ProlongedSoundConfig)
    impacts: ImpactRateConfig = field(default_factory=ImpactRateConfig)
    post_impact: PostImpactConfig = field(default_factory=PostImpactConfig)
    sustained: SustainedConfig = field(default_factory=SustainedConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)

    def __post_init__(self) -> None:
        if self.embedding.contexts not in ("global", "category", "period"):
            raise ValueError("anomaly.embedding.contexts must be 'global', 'category' or 'period'")
        hours = sorted(self.periods.values())
        if not hours or hours[0] != 0 or len(set(hours)) != len(hours) or hours[-1] > 23:
            raise ValueError("anomaly.periods needs distinct start hours in 0..23, one of them 0")
        if not 0 < self.embedding.frame_quantile < 1:
            raise ValueError("anomaly.embedding.frame_quantile must be between 0 and 1")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> AnomalyConfig:
        raw = dict(raw or {})
        overrides = raw.pop("categories", None) or {}
        cfg = _build(cls, raw)
        cfg.categories = {name: _build(CategoryOverride, v) for name, v in overrides.items()}
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = dict(raw or {})
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(f"unknown anomaly setting(s) in {cls.__name__}: {unknown}")
    kwargs = {}
    for name, value in raw.items():
        f = fields[name]
        nested = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(nested) and isinstance(value, dict):
            kwargs[name] = _build(type(nested), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
