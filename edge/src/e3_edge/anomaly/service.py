"""The baseline routine manager: one object per microphone for the edge main loop.

On first start it calibrates (checkpointing as it goes); once the calibration period is over it
saves the baseline and switches to detection. After a restart it picks up wherever it was:
a saved baseline means detect, a checkpoint means resume calibrating.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .baseline import Baseline
from .calibration import BaselineCalibrator
from .config import AnomalyConfig
from .detector import AnomalyDetector, AnomalyOutput
from .frames import Frame

log = logging.getLogger(__name__)


class AnomalyService:
    def __init__(self, cfg: AnomalyConfig, mic_id: str | None = None):
        self.cfg = cfg
        self.mic_id = mic_id
        self.baseline_path = _per_mic(Path(cfg.baseline_path), mic_id)
        self.checkpoint_path = _per_mic(Path(cfg.checkpoint_path), mic_id)
        self.detector: AnomalyDetector | None = None
        self.calibrator: BaselineCalibrator | None = None
        if self.baseline_path.exists():
            self._start_detecting(Baseline.load(self.baseline_path))
            log.info("Anomaly detection using baseline %s", self.baseline_path)
        elif self.checkpoint_path.exists():
            try:
                self.calibrator = BaselineCalibrator.resume(self.checkpoint_path, cfg)
                log.info("Resumed calibration: %s", self.calibrator.progress())
            except (ValueError, KeyError, OSError) as exc:
                log.warning("Can't resume calibration (%s); starting over", exc)
        if self.detector is None and self.calibrator is None:
            self.calibrator = BaselineCalibrator(cfg, mic_id=mic_id, checkpoint_path=self.checkpoint_path)
            log.info("No baseline yet: calibrating for %.1f h of audio", self.calibrator.total_seconds / 3600)

    @property
    def state(self) -> str:
        return "detecting" if self.detector else "calibrating"

    def process(self, frame: Frame) -> AnomalyOutput | None:
        """Scores the frame, or returns None while the baseline is still being learned."""
        if self.detector is not None:
            return self.detector.process(frame)
        self.calibrator.add(frame)
        if self.calibrator.done:
            baseline = self.calibrator.finalize()
            size = baseline.save(self.baseline_path)
            self.checkpoint_path.unlink(missing_ok=True)
            log.info("Baseline saved to %s (%.1f MB)", self.baseline_path, size / 1e6)
            self._start_detecting(baseline)
        return None

    def notify_fall_candidate(self, t: float, confidence: float) -> None:
        if self.detector is not None:
            self.detector.notify_fall_candidate(t, confidence)

    def pause(self) -> None:
        """Call when monitoring is muted so the gap isn't mistaken for silence."""
        if self.detector is not None:
            self.detector.pause()
        else:
            self.calibrator.temporal.interrupt()

    def status(self) -> dict[str, Any]:
        if self.detector is None:
            return {"state": self.state, **self.calibrator.progress()}
        return {"state": self.state, "baseline_created_at": self.detector.baseline.meta.get("created_at"),
                "baseline_updates": self.detector.baseline_updates}

    def _start_detecting(self, baseline: Baseline) -> None:
        self.calibrator = None
        self.detector = AnomalyDetector(baseline, self.cfg, mic_id=self.mic_id,
                                        baseline_path=self.baseline_path)


def _per_mic(path: Path, mic_id: str | None) -> Path:
    return path.with_name(f"{path.stem}-{mic_id}{path.suffix}") if mic_id else path
