"""Acoustic and behavioural anomaly detection (proposal 1.7.3.2-1.7.3.3, FR4, FR5).

Learns a household baseline from YAMNet embeddings and class scores, then scores each new frame
(Mahalanobis distance) and watches temporal patterns (running water, impacts, post-impact
silence). Input: Frame objects from the YAMNet stage. Output: AnomalyOutput for the Event Manager.
See docs/anomaly-detection.md.
"""

from .baseline import Baseline
from .calibration import BaselineCalibrator, CalibrationIncomplete
from .config import AnomalyConfig
from .detector import AnomalyDetector, AnomalyOutput
from .frames import Frame, FrameStream, frame_level_dbfs
from .service import AnomalyService
from .temporal import Finding

__all__ = [
    "AnomalyConfig",
    "AnomalyDetector",
    "AnomalyOutput",
    "AnomalyService",
    "Baseline",
    "BaselineCalibrator",
    "CalibrationIncomplete",
    "Finding",
    "Frame",
    "FrameStream",
    "frame_level_dbfs",
]
