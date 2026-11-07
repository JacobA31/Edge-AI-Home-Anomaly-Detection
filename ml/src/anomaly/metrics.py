"""Detection metrics. Proposal NF3 asks for misses and false alerts reported separately, so
nothing here is a single accuracy number."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomalous))]
    return float(roc_auc_score(y, np.r_[normal, anomalous]))


def partial_auc(normal: np.ndarray, anomalous: np.ndarray, max_fpr: float = 0.1) -> float:
    """Area under the ROC curve for FPR in [0, max_fpr], divided by max_fpr (DCASE Task 2 pAUC,
    not scikit-learn's McClish-corrected version). 1.0 is perfect; chance is max_fpr / 2."""
    labels = np.r_[np.zeros(len(normal)), np.ones(len(anomalous))]
    fpr, tpr, _ = roc_curve(labels, np.r_[normal, anomalous])
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x = np.r_[fpr[:stop], max_fpr]
    y = np.r_[tpr[:stop], np.interp(max_fpr, fpr, tpr)]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / max_fpr)  # trapezoid rule


def tpr_at_fpr(normal: np.ndarray, anomalous: np.ndarray, fpr: float) -> float:
    """Share of anomalies caught at the threshold that flags `fpr` of normal frames."""
    return float(np.mean(anomalous > np.quantile(normal, 1 - fpr)))


def at_threshold(normal: np.ndarray, anomalous: np.ndarray, threshold: float, hours: float) -> dict:
    """What a fixed (calibrated) threshold does on test data."""
    flagged = normal > threshold
    return {
        "false_alarm_frame_share": round(float(flagged.mean()), 5),
        "false_alarm_frames_per_hour": round(float(flagged.sum() / hours), 2) if hours else None,
        "anomalies_caught": round(float(np.mean(anomalous > threshold)), 4) if len(anomalous) else None,
        "anomalies_missed": round(float(np.mean(anomalous <= threshold)), 4) if len(anomalous) else None,
    }
