"""Behavioural analysis over time: how long sounds last, how often they happen, what follows.

Proposal 1.6.1 limits the prototype to three temporal anomalies, all implemented here:

* prolonged running water ............................ ProlongedSound
* unusually frequent impacts within a time window ..... ImpactRate
* a fall-like impact followed by reduced activity ..... PostImpactMonitor

plus SustainedUnusual: the embedding score stays anomalous for a while (e.g. an alarm the
household never makes). Each threshold is learned from the calibration period (FR4, FR5), with
a documented default when the baseline saw too few examples. Detectors only report Findings;
the Event Manager decides what becomes a caregiver alert (FR7). Silence is never treated as
proof of an emergency, only as supporting evidence (proposal 1.7.4).

All timing uses frame timestamps, never the wall clock, so a week of SINS can be replayed in
minutes and give the same results as live audio.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .categories import CategoryMapper, CategorySpec
from .config import (
    ActivityConfig,
    AnomalyConfig,
    ImpactRateConfig,
    PostImpactConfig,
    ProlongedSoundConfig,
    SustainedConfig,
)


@dataclass
class Finding:
    """Something the anomaly module noticed. Maps onto shared/event.schema.json via to_event_fields()."""

    kind: str  # prolonged_water, repeated_impacts, post_impact_silence, post_impact_inactivity, ...
    event_type: str  # "anomaly" or "inactivity" (schema enum)
    role: str  # "alert": may justify an alert on its own; "evidence": only supports another event
    detected_at: float  # Unix seconds
    started_at: float
    confidence: float  # 0..1
    value: float  # what was measured...
    threshold: float  # ...and the learned limit it crossed
    unit: str
    extra: dict[str, Any] = field(default_factory=dict)

    def details(self) -> dict[str, Any]:
        """Small, audio-free summary for the event's `details` (the schema allows 10 keys)."""
        details = {
            "kind": self.kind,
            "role": self.role,
            "value": round(float(self.value), 2),
            "threshold": round(float(self.threshold), 2),
            "unit": self.unit,
            "started_at": _iso(self.started_at),
        } | self.extra
        if len(details) > 10:
            raise ValueError(f"{self.kind}: too many details for the event schema")
        return details

    def to_event_fields(self) -> dict[str, Any]:
        """Keyword arguments for e3_edge.events.make_event(device_id, room, **fields)."""
        return {"event_type": self.event_type, "confidence": round(float(self.confidence), 3),
                "details": self.details()}


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Reservoir:
    """Keeps a uniform random sample of at most `capacity` values (Vitter's algorithm R)."""

    def __init__(self, capacity: int, rng: np.random.Generator, items: list | None = None, seen: int = 0):
        self.capacity = capacity
        self.rng = rng
        self.items = list(items or [])
        self.seen = max(seen, len(self.items))

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(value)
        else:
            j = int(self.rng.integers(self.seen))
            if j < self.capacity:
                self.items[j] = value

    def state(self) -> dict[str, Any]:
        return {"items": self.items, "seen": self.seen}


def hours_by_period(periods: dict[str, int]) -> dict[str, list[int]]:
    """{period: local hours it covers}; each period runs until the next one starts."""
    starts = sorted((h, name) for name, h in periods.items())
    out: dict[str, list[int]] = {name: [] for _, name in starts}
    for hour in range(24):
        out[[name for start, name in starts if hour >= start][-1]].append(hour)
    return out


class Clock:
    """Local hour and time-of-day period of a timestamp (period starts come from the config)."""

    def __init__(self, timezone: str, periods: dict[str, int]):
        self.tz = ZoneInfo(timezone)
        by_period = hours_by_period(periods)
        self.names = list(by_period)
        self._period_of_hour = {h: name for name, hours in by_period.items() for h in hours}
        self._minute = None
        self._hour = 0

    def hour(self, t: float) -> int:
        minute = int(t // 60)
        if minute != self._minute:
            self._minute = minute
            self._hour = datetime.fromtimestamp(t, self.tz).hour
        return self._hour

    def period(self, t: float) -> str:
        return self._period_of_hour[self.hour(t)]


class EpisodeTracker:
    """Turns a category score per frame into episodes, with hysteresis and gap tolerance."""

    def __init__(self, spec: CategorySpec, frame_seconds: float):
        self.spec = spec
        self.frame_seconds = frame_seconds
        self.active = False
        self.start = 0.0
        self.last_on = 0.0

    def update(self, t: float, score: float) -> tuple[bool, float | None]:
        """Returns (episode started on this frame, duration of an episode that just ended)."""
        ended = None
        if self.active:
            if score >= self.spec.off:
                self.last_on = t
            elif t - self.last_on > self.spec.gap_seconds:
                ended = self.duration()
                self.active = False
        if not self.active and score >= self.spec.on:
            self.active, self.start, self.last_on = True, t, t
            return True, ended
        return False, ended

    def duration(self) -> float:
        return self.last_on + self.frame_seconds - self.start if self.active else 0.0

    def interrupt(self) -> float | None:
        """Ends the current episode at a discontinuity; returns its duration if there was one."""
        if not self.active:
            return None
        self.active = False
        return self.last_on + self.frame_seconds - self.start


def is_active(level_dbfs: float | None, any_category: bool, silence_score: float | None,
              noise_floor_dbfs: float, cfg: ActivityConfig) -> bool:
    """Is there household sound in this frame? Unknown counts as active (no silence claimed)."""
    if any_category:
        return True
    if level_dbfs is not None:
        return level_dbfs > noise_floor_dbfs + cfg.margin_db
    if silence_score is not None:
        return silence_score < cfg.silence_score
    return True


class ProlongedSound:
    """A category (running water by default) lasting much longer than the household's usual."""

    def __init__(self, cfg: ProlongedSoundConfig, threshold_seconds: float):
        self.cfg = cfg
        self.threshold = threshold_seconds
        self._episode: float | None = None
        self._stage = 0

    def update(self, t: float, tracker: EpisodeTracker) -> Finding | None:
        if not tracker.active:
            self._episode = None
            return None
        if tracker.start != self._episode:
            self._episode, self._stage = tracker.start, 0
        duration = tracker.duration()
        limit = self.threshold * (1 if self._stage == 0 else self.cfg.escalate_factor)
        if self._stage >= 2 or duration < limit:
            return None
        self._stage += 1
        return Finding(f"prolonged_{self.cfg.category}", "anomaly", "alert", t, tracker.start,
                       min(1.0, 0.5 * duration / self.threshold), duration, self.threshold, "s",
                       {"stage": self._stage})


class ImpactRate:
    """More impact sounds within a sliding window than the household normally makes.

    While learning, it records for each window-sized block of time the highest trailing-window
    count seen in it (zeros included), which is exactly what the detector compares later.
    """

    def __init__(self, cfg: ImpactRateConfig, thresholds: dict[str, int] | None = None,
                 default: int | None = None, samples: dict[str, Reservoir] | None = None):
        self.cfg = cfg
        self.thresholds = thresholds
        self.default = cfg.default_count if default is None else default
        self.samples = samples  # learning mode when set
        self.onsets: deque[float] = deque()
        self.cooldown_until = -math.inf
        self._block_start: float | None = None
        self._block_period = ""
        self._block_max = 0

    def update(self, t: float, onset: bool, period: str) -> Finding | None:
        window = self.cfg.window_seconds
        while self.onsets and self.onsets[0] <= t - window:
            self.onsets.popleft()
        count = 0
        if onset:
            self.onsets.append(t)
            count = len(self.onsets)
        if self.samples is not None:
            if self._block_start is None:
                self._block_start, self._block_period, self._block_max = t, period, 0
            elif t - self._block_start >= window:
                self.samples[self._block_period].add(self._block_max)
                self._block_start, self._block_period, self._block_max = t, period, 0
            self._block_max = max(self._block_max, count)
        if not onset or self.thresholds is None:
            return None
        limit = self.thresholds.get(period, self.default)
        if count < limit or t < self.cooldown_until:
            return None
        self.cooldown_until = t + self.cfg.cooldown_seconds
        return Finding("repeated_impacts", "anomaly", "alert", t, self.onsets[0],
                       min(1.0, 0.5 * count / limit), count, limit, "impacts",
                       {"window_s": window, "period": period})

    def interrupt(self) -> None:
        self.onsets.clear()
        self._block_start = None


INHERIT_SECONDS = 10.0  # the fall-event budget of NF1
MIN_FLOOR_FRAMES = 100  # frame levels needed before the room's noise floor is trusted
FLOOR_UPDATE_FRAMES = 500  # while learning, re-estimate the floor this often (~4 min of frames)


@dataclass
class _QuietRun:
    t0: float
    strength: float
    source: str
    observed: float = 0.0
    active: float = 0.0
    short_done: bool = False
    recent: deque = field(default_factory=deque)  # (t, active seconds) over the last recent_seconds


class PostImpactMonitor:
    """Watches household activity after an impact or a fall candidate.

    After `short_seconds` of mostly quiet audio it reports `post_impact_silence` (evidence the
    Event Manager can attach to a fall). If the quiet lasts past the learned threshold, it
    reports `post_impact_inactivity`. Activity resuming ends the watch.
    """

    def __init__(self, cfg: PostImpactConfig, extended_seconds: float | None = None,
                 samples: Reservoir | None = None):
        self.cfg = cfg
        self.extended = extended_seconds
        self.samples = samples  # learning mode when set
        self.run: _QuietRun | None = None

    def trigger(self, t: float, strength: float, source: str) -> None:
        old = self.run
        if old is not None and abs(t - old.t0) <= self.cfg.merge_seconds:
            if strength >= old.strength:
                old.strength, old.source = strength, source
            return
        self._finish()
        self.run = _QuietRun(t, strength, source)
        # A weaker thud soon after (table, then floor) must not downgrade a fall.
        if old is not None and t - old.t0 <= INHERIT_SECONDS and old.strength > strength:
            self.run.strength, self.run.source = old.strength, old.source

    def update(self, t: float, dt: float, active: bool, settling: bool = False) -> list[Finding]:
        """`settling`: the impact is still being heard, so quiet can't be measured yet."""
        run = self.run
        if run is None or t < run.t0 + self.cfg.grace_seconds:
            return []
        if run.observed == 0 and (settling or active):
            # The trigger's own sound is still going: the rest of a fall clip, its tail, a second
            # thud. A fall-classifier trigger has no "impact" category to wait for, so quiet is
            # measured from the first quiet frame. Counting the fall itself as activity ended
            # every watch at the 5 s check (9 of 10 planted SAFE falls missed on SINS).
            if t - run.t0 > self.cfg.settle_max_seconds:
                self._finish()  # never went quiet: someone is moving
            return []
        run.observed += dt
        run.active += dt if active else 0.0
        run.recent.append((t, dt if active else 0.0))
        while run.recent[0][0] <= t - self.cfg.recent_seconds:
            run.recent.popleft()
        # The share since the impact barely moves late in a long watch (20 s of walking after
        # 90 s of quiet is 18%), so someone moving about now also ends it.
        if sum(a for _, a in run.recent) > self.cfg.max_recent_active_fraction * self.cfg.recent_seconds:
            self._finish()
            return []
        if run.observed < self.cfg.short_seconds:
            return []
        if run.active / run.observed > self.cfg.max_active_fraction:
            self._finish()
            return []
        out = []
        extra = {"trigger": run.source}
        if not run.short_done:
            run.short_done = True
            out.append(Finding("post_impact_silence", "inactivity", "evidence", t, run.t0,
                               run.strength, run.observed, self.cfg.short_seconds, "s", extra))
        limit = self.extended
        if limit is not None and run.strength >= self.cfg.strong_level:
            limit = min(limit, self.cfg.strong_max_seconds)
        if limit is not None and run.observed >= limit:
            out.append(Finding("post_impact_inactivity", "inactivity", "alert", t, run.t0,
                               run.strength, run.observed, limit, "s", extra))
            self._finish()
        elif run.observed >= self.cfg.max_seconds:
            self._finish()
        return out

    def _finish(self) -> None:
        if self.run is not None and self.samples is not None:
            self.samples.add(min(self.run.observed, self.cfg.max_seconds))
        self.run = None

    def interrupt(self) -> None:
        self.run = None  # what happened during the gap is unknown, so nothing is learned


class SustainedUnusual:
    """The embedding score stays above the anomaly threshold for much of a short window."""

    def __init__(self, cfg: SustainedConfig):
        self.cfg = cfg
        self.window: deque[tuple[float, bool, float]] = deque()
        self.n_anomalous = 0
        self.cooldown_until = -math.inf

    def update(self, t: float, anomalous: bool, level: float) -> Finding | None:
        self.window.append((t, anomalous, level))
        self.n_anomalous += anomalous
        while self.window[0][0] <= t - self.cfg.window_seconds:
            self.n_anomalous -= self.window.popleft()[1]
        fraction = self.n_anomalous / len(self.window)
        if (self.n_anomalous < self.cfg.min_frames or fraction < self.cfg.min_fraction
                or t < self.cooldown_until):
            return None
        self.cooldown_until = t + self.cfg.cooldown_seconds
        hits = [(ts, lv) for ts, a, lv in self.window if a]
        return Finding("sustained_unusual_sound", "anomaly", "alert", t, hits[0][0],
                       float(np.mean([lv for _, lv in hits])), fraction, self.cfg.min_fraction,
                       "fraction", {"window_s": self.cfg.window_seconds})

    def interrupt(self) -> None:
        self.window.clear()
        self.n_anomalous = 0


@dataclass
class TemporalStats:
    """What the calibration period taught about household routine (FR4)."""

    categories: list[str]
    periods: list[str]
    durations: dict[str, dict[str, Any]]  # category -> reservoir state of episode lengths (s)
    impact_counts: dict[str, dict[str, Any]]  # period -> reservoir state of window counts
    post_impact_quiet: dict[str, Any]  # reservoir state of quiet lengths after impacts (s)
    levels: dict[str, Any]  # reservoir state of frame levels (dBFS)
    hourly_observed: list[float]  # [24] seconds of audio per local hour
    hourly_active: list[float]  # [24] seconds with household activity
    hourly_category: list[list[float]]  # [24][C] seconds each category was present
    hourly_onsets: list[list[int]]  # [24][C] episodes started

    @classmethod
    def empty(cls, categories: list[str], periods: list[str]) -> TemporalStats:
        blank = {"items": [], "seen": 0}
        return cls(categories, periods, {c: dict(blank) for c in categories},
                   {p: dict(blank) for p in periods}, dict(blank), dict(blank), [0.0] * 24,
                   [0.0] * 24, [[0.0] * len(categories) for _ in range(24)],
                   [[0] * len(categories) for _ in range(24)])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemporalStats:
        return cls(**d)

    def merged(self, other: TemporalStats, weight: float, rng: np.random.Generator,
               capacity: int) -> TemporalStats:
        """Blend with newer stats; `weight` is the share of `other` (used by adaptation)."""
        def mix(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
            old, new = a["items"], b["items"]
            if not old or not new:
                items = list(old or new)[:capacity]
                return {"items": items, "seen": len(items)}
            size = min(capacity, max(len(old), len(new)))
            n_new = round(weight * size)
            items = ([old[i] for i in rng.choice(len(old), size - n_new, replace=len(old) < size - n_new)]
                     + [new[i] for i in rng.choice(len(new), n_new, replace=len(new) < n_new)])
            return {"items": items, "seen": len(items)}

        def mix_arr(a: list, b: list) -> list:
            return ((1 - weight) * np.asarray(a, dtype=float) + weight * np.asarray(b, dtype=float)).tolist()

        return TemporalStats(
            self.categories, self.periods,
            {c: mix(self.durations[c], other.durations.get(c, {"items": []})) for c in self.categories},
            {p: mix(self.impact_counts[p], other.impact_counts.get(p, {"items": []})) for p in self.periods},
            mix(self.post_impact_quiet, other.post_impact_quiet), mix(self.levels, other.levels),
            mix_arr(self.hourly_observed, other.hourly_observed),
            mix_arr(self.hourly_active, other.hourly_active),
            mix_arr(self.hourly_category, other.hourly_category),
            mix_arr(self.hourly_onsets, other.hourly_onsets),
        )


@dataclass
class TemporalThresholds:
    """Limits derived from TemporalStats. `sources` says whether each was learned or defaulted."""

    noise_floor_dbfs: float
    prolonged_seconds: float
    impact_counts: dict[str, int]
    impact_default: int
    post_impact_seconds: float
    sources: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemporalThresholds:
        return cls(**d)


def learned_noise_floor(levels: list[float], cfg: AnomalyConfig) -> float | None:
    """The room's noise floor from frame levels (dBFS), or None while there are too few."""
    if len(levels) < MIN_FLOOR_FRAMES:
        return None
    return float(np.quantile(levels, cfg.activity.noise_floor_quantile))


def derive_thresholds(stats: TemporalStats, cfg: AnomalyConfig) -> TemporalThresholds:
    sources = {}

    levels = stats.levels["items"]
    floor = learned_noise_floor(levels, cfg)
    if floor is not None:
        sources["noise_floor"] = f"learned from {len(levels)} frames"
    else:
        floor = cfg.activity.default_noise_floor_dbfs
        sources["noise_floor"] = "default (no frame levels)"

    w = cfg.water
    durations = stats.durations.get(w.category, {"items": []})["items"]
    if len(durations) >= w.min_episodes:
        raw = float(np.quantile(durations, w.duration_quantile)) * w.margin
        prolonged = float(np.clip(raw, w.min_seconds, w.max_seconds))
        sources["prolonged"] = f"learned from {len(durations)} {w.category} episodes"
    else:
        prolonged = w.default_seconds
        sources["prolonged"] = f"default (only {len(durations)} {w.category} episodes)"

    ic = cfg.impacts
    pooled = [c for p in stats.periods for c in stats.impact_counts[p]["items"]]
    hours_of = hours_by_period(cfg.periods)
    impact_col = stats.categories.index(ic.category) if ic.category in stats.categories else None

    def rate_limit(hours: list[int]) -> int:
        observed = sum(stats.hourly_observed[h] for h in hours)
        if impact_col is None or observed <= 0:
            return 0
        onsets = sum(stats.hourly_onsets[h][impact_col] for h in hours)
        per_day = onsets / (observed / (len(hours) * 3600))  # onsets in one full day of these hours
        return poisson_limit(onsets / observed * ic.window_seconds, per_day, ic.target_alerts_per_day)

    def count_limit(counts: list[int], hours: list[int]) -> int:
        empirical = math.ceil(float(np.quantile(counts, ic.count_quantile))) + 1
        return max(ic.min_count, empirical, rate_limit(hours))

    if len(pooled) >= ic.min_windows:
        default_count = count_limit(pooled, list(range(24)))
        sources["impacts"] = f"learned from {len(pooled)} windows"
    else:
        default_count = ic.default_count
        sources["impacts"] = f"default (only {len(pooled)} windows)"
    per_period = {p: count_limit(stats.impact_counts[p]["items"], hours_of.get(p, []))
                  for p in stats.periods if len(stats.impact_counts[p]["items"]) >= ic.min_windows}

    pi = cfg.post_impact
    quiet = stats.post_impact_quiet["items"]
    if len(quiet) >= pi.min_samples:
        raw = float(np.quantile(quiet, pi.quiet_quantile)) * pi.margin
        post = float(np.clip(raw, max(pi.min_seconds, pi.short_seconds), pi.max_seconds))
        sources["post_impact"] = f"learned from {len(quiet)} impacts"
    else:
        post = max(pi.default_seconds, pi.short_seconds)
        sources["post_impact"] = f"default (only {len(quiet)} impacts)"

    return TemporalThresholds(floor, prolonged, per_period, default_count, post, sources)


def poisson_limit(expected_in_window: float, onsets_per_day: float, target_per_day: float) -> int:
    """Smallest count k such that, if impacts arrive as a Poisson process, an onset finds k-1
    others in its window rarely enough to cause at most target_per_day false alerts."""
    if onsets_per_day <= 0:
        return 1
    lam = expected_in_window
    term, cdf, m, tail = math.exp(-lam), 0.0, 0, 1.0  # tail = P(X >= m)
    while onsets_per_day * tail > target_per_day and m < 1000:
        cdf += term
        m += 1
        term *= lam / m
        tail = max(0.0, 1.0 - cdf)
    return m + 1


@dataclass
class TemporalResult:
    active: bool
    categories: tuple[str, ...]  # categories present in this frame
    findings: list[Finding]


class TemporalEngine:
    """Runs every temporal tracker for one microphone, either learning or detecting."""

    def __init__(self, cfg: AnomalyConfig, mapper: CategoryMapper, clock: Clock,
                 thresholds: TemporalThresholds | None = None, learn: bool = False,
                 rng: np.random.Generator | None = None, stats: TemporalStats | None = None):
        self.cfg = cfg
        self.mapper = mapper
        self.clock = clock
        self.rng = rng or np.random.default_rng(cfg.calibration.seed)
        cap = cfg.calibration.sample_reservoir
        self.trackers = {n: EpisodeTracker(s, cfg.frame_seconds) for n, s in mapper.specs.items()}
        self.last_t: float | None = None
        self.stats: TemporalStats | None = None
        self._res: dict[str, Any] = {}
        if learn:
            self.stats = stats or TemporalStats.empty(list(mapper.names), clock.names)

            def reservoir(state: dict[str, Any]) -> Reservoir:
                return Reservoir(cap, self.rng, state["items"], state["seen"])

            self._res = {
                "durations": {c: reservoir(self.stats.durations[c]) for c in self.stats.categories},
                "impacts": {p: reservoir(self.stats.impact_counts[p]) for p in self.stats.periods},
                "quiet": reservoir(self.stats.post_impact_quiet),
                "levels": reservoir(self.stats.levels),
            }
        th = thresholds
        self.noise_floor = th.noise_floor_dbfs if th else cfg.activity.default_noise_floor_dbfs
        # While learning, "active" uses the floor measured so far, as detection will, instead of
        # the default: in a room with a fan at -40 dBFS every frame would count as activity.
        self._floor_countdown = 0
        if learn:
            self._update_floor()
        self.water = ProlongedSound(cfg.water, th.prolonged_seconds) if th else None
        self.impacts = ImpactRate(cfg.impacts, th.impact_counts if th else None,
                                  th.impact_default if th else None, self._res.get("impacts"))
        self.post = PostImpactMonitor(cfg.post_impact, th.post_impact_seconds if th else None,
                                      self._res.get("quiet"))
        self.sustained = SustainedUnusual(cfg.sustained)
        if cfg.water.category not in self.trackers or cfg.impacts.category not in self.trackers:
            raise ValueError("water/impact detectors refer to categories that don't exist")

    def set_thresholds(self, th: TemporalThresholds) -> None:
        self.noise_floor = th.noise_floor_dbfs
        if self.water is None:
            self.water = ProlongedSound(self.cfg.water, th.prolonged_seconds)
        self.water.threshold = th.prolonged_seconds
        self.impacts.thresholds, self.impacts.default = th.impact_counts, th.impact_default
        self.post.extended = th.post_impact_seconds

    def interrupt(self) -> None:
        """A gap in the audio (mute, restart): nothing carries over it."""
        for name, tracker in self.trackers.items():
            duration = tracker.interrupt()
            if duration is not None:
                self._learn_episode(name, duration)
        self.impacts.interrupt()
        self.post.interrupt()
        self.sustained.interrupt()
        self.last_t = None

    def notify_fall_candidate(self, t: float, confidence: float) -> None:
        self.post.trigger(t, float(confidence), "fall_candidate")

    def process(self, t: float, class_scores: np.ndarray | None, level_dbfs: float | None,
                anomalous: bool | None = None, anomaly_level: float = 0.0) -> TemporalResult:
        # A long gap, or the clock stepping back (an NTP correction), breaks the patterns in
        # progress; a negative time step would otherwise be counted as quiet or episode time.
        if self.last_t is not None and (t <= self.last_t or t - self.last_t > self.cfg.max_gap_seconds):
            self.interrupt()
        dt = self.cfg.frame_seconds if self.last_t is None else t - self.last_t
        self.last_t = t
        hour = self.clock.hour(t)
        period = self.clock.period(t)

        impact_onset = False
        present: list[str] = []
        silence = None
        if class_scores is not None:
            silence = float(class_scores[self.mapper.silence_index])
            for (name, tracker), score in zip(self.trackers.items(), self.mapper.scores(class_scores),
                                              strict=True):
                onset, ended = tracker.update(t, float(score))
                if ended is not None:
                    self._learn_episode(name, ended)
                if tracker.active:
                    present.append(name)
                if onset:
                    impact_onset |= name == self.cfg.impacts.category
                    if self.stats is not None:
                        self.stats.hourly_onsets[hour][self.mapper.names.index(name)] += 1
        active = is_active(level_dbfs, bool(present), silence, self.noise_floor, self.cfg.activity)

        if self.stats is not None:
            self.stats.hourly_observed[hour] += dt
            self.stats.hourly_active[hour] += dt if active else 0.0
            row = self.stats.hourly_category[hour]
            for name in present:
                row[self.mapper.names.index(name)] += dt
            if level_dbfs is not None:
                self._res["levels"].add(float(level_dbfs))
                self._floor_countdown -= 1
                if self._floor_countdown <= 0:
                    self._update_floor()

        findings: list[Finding] = []
        hit = self.impacts.update(t, impact_onset, period)
        if hit:
            findings.append(hit)
        if impact_onset:
            strength = max(self.cfg.post_impact.category_strength, anomaly_level if anomalous else 0.0)
            self.post.trigger(t, strength, "impact")
        findings += self.post.update(t, dt, active, self.trackers[self.cfg.impacts.category].active)
        if self.water is not None:
            hit = self.water.update(t, self.trackers[self.cfg.water.category])
            if hit:
                findings.append(hit)
        if anomalous is not None:
            hit = self.sustained.update(t, anomalous, anomaly_level)
            if hit:
                findings.append(hit)
        return TemporalResult(active, tuple(present), findings)

    def _update_floor(self) -> None:
        self._floor_countdown = FLOOR_UPDATE_FRAMES
        floor = learned_noise_floor(self._res["levels"].items, self.cfg)
        if floor is not None:
            self.noise_floor = floor

    def _learn_episode(self, category: str, duration: float) -> None:
        if self.stats is None:
            return
        if category == self.cfg.water.category and duration < self.cfg.water.min_episode_seconds:
            return
        self._res["durations"][category].add(float(duration))

    def snapshot_stats(self) -> TemporalStats:
        """Learned statistics so far (reservoir contents synced into the stats object)."""
        if self.stats is None:
            raise RuntimeError("this engine is not learning")
        s = self.stats
        s.durations = {c: self._res["durations"][c].state() for c in s.categories}
        s.impact_counts = {p: self._res["impacts"][p].state() for p in s.periods}
        s.post_impact_quiet = self._res["quiet"].state()
        s.levels = self._res["levels"].state()
        return TemporalStats.from_dict(asdict(s))  # asdict deep-copies
