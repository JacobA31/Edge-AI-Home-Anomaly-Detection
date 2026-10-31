"""Shared fixtures for the anomaly tests (synthetic frames; see e3_edge/anomaly/simulate.py)."""

import pytest

from e3_edge.anomaly import AnomalyConfig, BaselineCalibrator
from e3_edge.anomaly.simulate import HouseholdSimulator

DAY = 86400.0
HOUR = 3600.0


def calibration_config(stats_hours: float = 18, threshold_hours: float = 6, **sections) -> AnomalyConfig:
    """Config with a short calibration; keyword arguments add or extend `anomaly:` sections."""
    raw = {"calibration": {"stats_seconds": stats_hours * HOUR, "threshold_seconds": threshold_hours * HOUR}}
    for key, value in sections.items():
        if isinstance(value, dict):
            raw.setdefault(key, {}).update(value)
        else:
            raw[key] = value
    return AnomalyConfig.from_dict(raw)


def calibrate(sim: HouseholdSimulator, cfg: AnomalyConfig, start: float = 0.0, seconds: float = DAY):
    cal = BaselineCalibrator(cfg)
    for chunk in sim.chunks(start, seconds):
        for frame in chunk.frames():
            cal.add(frame)
    return cal


@pytest.fixture(scope="session")
def sim() -> HouseholdSimulator:
    return HouseholdSimulator(dim=32, hop=0.96, seed=7)


@pytest.fixture(scope="session")
def day_baseline(sim):
    """A baseline calibrated on synthetic days 0-1 (36 h fit + 12 h threshold), category mode.

    Two days, because rare sounds need enough frames for their own component: one synthetic
    day has only ~30 impact frames before the threshold phase starts.
    """
    cal = calibrate(sim, calibration_config(stats_hours=36, threshold_hours=12), seconds=2 * DAY)
    assert cal.done
    return cal.finalize()
