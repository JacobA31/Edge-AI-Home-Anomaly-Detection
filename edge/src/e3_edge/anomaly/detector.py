"""Scores live frames against a learned baseline: the "anomaly handler" of the edge pipeline.

One AnomalyDetector per microphone stream (proposal 1.7.2: streams are processed
independently). For every frame it returns an AnomalyOutput for the Event Manager:

* distance / level / is_anomalous: how unusual this sound is for this home (Mahalanobis),
* findings: behavioural patterns completed on this frame (prolonged water, repeated impacts,
  post-impact silence/inactivity, sustained unusual sound).

The detector never creates or sends events itself; FR7's risk decision belongs to the Event
Manager, which combines these outputs with the fall classifier's confidence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from .baseline import Baseline
from .calibration import STATS, BaselineCalibrator, shadow_config
from .categories import CategoryMapper, resolve
from .config import AnomalyConfig
from .frames import Frame
from .temporal import Clock, Finding, TemporalEngine, derive_thresholds

log = logging.getLogger(__name__)


@dataclass
class AnomalyOutput:
    t: float
    distance: float  # Mahalanobis distance from the baseline
    level: float  # 0..1: share of baseline frames that were closer to normal
    is_anomalous: bool  # distance above the baseline's frame-quantile threshold
    component: str  # baseline component that matched best ("global" or a period)
    active: bool  # household sound present in this frame
    categories: tuple[str, ...]  # sound categories present (water, impact, ...)
    findings: list[Finding] = field(default_factory=list)


class AnomalyDetector:
    def __init__(self, baseline: Baseline, cfg: AnomalyConfig | None = None, *,
                 mic_id: str | None = None, baseline_path: str | Path | None = None):
        # Without an explicit config, behave exactly as calibrated. With one, thresholds are
        # re-derived from the learned statistics, so tuning needs no recalibration.
        self.cfg = cfg or AnomalyConfig.from_dict(baseline.meta.get("config"))
        self.mic_id = mic_id
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.mapper = CategoryMapper(resolve(self.cfg.categories))
        self.clock = Clock(self.cfg.timezone, self.cfg.periods)
        self.temporal = TemporalEngine(self.cfg, self.mapper, self.clock)
        self.baseline_updates = 0
        self._holdoff_until = -math.inf
        self._shadow: BaselineCalibrator | None = None
        self._use(baseline)

    def _use(self, baseline: Baseline) -> None:
        self.baseline = baseline
        self.threshold = baseline.distance_threshold(self.cfg.embedding.frame_quantile)
        self.thresholds = derive_thresholds(baseline.temporal, self.cfg)
        self.temporal.set_thresholds(self.thresholds)
        if self.cfg.adaptation.enabled:
            self._shadow = BaselineCalibrator(shadow_config(self.cfg), prior=baseline,
                                              prior_weight=self.cfg.adaptation.weight, mic_id=self.mic_id)

    def process(self, frame: Frame) -> AnomalyOutput:
        if len(frame.embedding) != self.baseline.embedding.dim:
            raise ValueError(f"embedding has {len(frame.embedding)} dims, baseline has "
                             f"{self.baseline.embedding.dim}")
        distance, component = self.baseline.embedding.score(frame.embedding)
        level = self.baseline.level(distance)
        anomalous = distance > self.threshold
        result = self.temporal.process(frame.t, frame.scores, frame.level_dbfs, anomalous, level)
        if any(f.role == "alert" for f in result.findings):
            self._holdoff_until = frame.t + self.cfg.adaptation.holdoff_seconds
        if self._shadow is not None:
            self._adapt(frame, level)
        return AnomalyOutput(frame.t, distance, level, anomalous, component, result.active,
                             result.categories, result.findings)

    def notify_fall_candidate(self, t: float, confidence: float) -> None:
        """Called when the fall classifier sees a fall-like sound at time t: starts the silence watch."""
        self.temporal.notify_fall_candidate(t, confidence)

    def pause(self) -> None:
        """Monitoring muted (FR10): forget in-progress patterns rather than count the mute as silence."""
        self.temporal.interrupt()

    def _adapt(self, frame: Frame, level: float) -> None:
        if frame.t < self._holdoff_until:
            return
        # Unusual frames are kept out of what is learned, but not out of the threshold sample:
        # cutting off its tail would set the new threshold far too low.
        if self._shadow.phase == STATS and level >= self.cfg.adaptation.gate_level:
            return
        self._shadow.add(frame)
        if not self._shadow.done:
            return
        updated = self._shadow.finalize()
        self.baseline_updates += 1
        log.info("Baseline adapted (update %d): threshold %.2f -> %.2f", self.baseline_updates,
                 self.threshold, updated.distance_threshold(self.cfg.embedding.frame_quantile))
        if self.baseline_path is not None:
            if self.baseline_path.exists():
                self.baseline_path.replace(self.baseline_path.with_suffix(".prev.npz"))
            updated.save(self.baseline_path)
        self._use(updated)
