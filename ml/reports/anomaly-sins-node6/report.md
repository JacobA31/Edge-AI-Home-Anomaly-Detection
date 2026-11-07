# Anomaly detection evaluation

## Data

```
{
  "calibration_hours": 72.0,
  "test_hours": 57.56,
  "fall_frames": 2849,
  "embedding_dim": 1024,
  "class_scores": true,
  "levels": true,
  "timezone": "Europe/Brussels"
}
```

## 1. Frame level: SAFE falls vs held-out household frames

| method | edge_ready | auc | pauc_0.1 | tpr_at_fpr_1% | anomalies_caught | anomalies_missed | false_alarm_frames_per_hour | model_mb | score_us_per_frame |
|---|---|---|---|---|---|---|---|---|---|
| edge_category | True | 0.8309 | 0.3403 | 0.1169 | 0.026 | 0.974 | 13.38 | 4.94 | 5.2 |
| edge_global | True | 0.834 | 0.3442 | 0.1425 | 0.0565 | 0.9435 | 13.53 | 4.41 | 1.1 |
| edge_global_pca64 | True | 0.909 | 0.4427 | 0.1474 | 0.0337 | 0.9663 | 11.89 | 0.47 | 0.5 |
| edge_period | True | 0.8589 | 0.3848 | 0.1362 | 0.034 | 0.966 | 12.79 | 4.95 | 3.1 |
| kmeans8_selective | False | 0.8321 | 0.3164 | 0.1099 | 0.0291 | 0.9709 | 11.6 | 5.29 | 10.8 |
| isolation_forest | False | 0.3716 | 0.0002 | 0.0 | 0.0 | 1.0 | 29.59 |  | 11.0 |
| knn5_pca64 | False | 0.8532 | 0.357 | 0.1207 | 0.0316 | 0.9684 | 9.81 | 5.38 | 9.8 |
| euclidean | False | 0.7652 | 0.0021 | 0.0 | 0.0 | 1.0 | 29.74 |  | 1.2 |

## 2. Behavioural detectors

Untouched test days (2.4):

```
{
  "alerts_per_day": 0.42,
  "confident_alerts_per_day": 0.42,
  "evidence_per_day": 5.42,
  "alerts_by_kind": {
    "repeated_impacts": 1
  }
}
```

### Planted anomalies, impact trigger only

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 10 | 10 | 1.0 | 119.5 | 119.7 |
| fall:post_impact_silence | 10 | 1 | 0.1 | 10.0 | 10.0 |
| fall:post_impact_inactivity | 10 | 0 | 0.0 |  |  |
| impact_burst:repeated_impacts | 8 | 8 | 1.0 | 80.2 | 80.4 |
| other_alerts |  |  |  |  |  |

Other alerts per day: 1.67 ({"by_kind": {"post_impact_inactivity": 3, "repeated_impacts": 1}, "confident": 1})

### Planted anomalies, with fall classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 10 | 10 | 1.0 | 119.5 | 119.7 |
| fall:post_impact_silence | 10 | 10 | 1.0 | 9.9 | 11.8 |
| fall:post_impact_inactivity | 10 | 10 | 1.0 | 124.9 | 126.6 |
| impact_burst:repeated_impacts | 8 | 8 | 1.0 | 80.2 | 80.4 |
| other_alerts |  |  |  |  |  |

Other alerts per day: 1.67 ({"by_kind": {"post_impact_inactivity": 3, "repeated_impacts": 1}, "confident": 1})

## 3. Calibration length

| calibration_hours | components | test_frames_flagged | target_flagged | auc | alerts_per_day | confident_alerts_per_day |
|---|---|---|---|---|---|---|
| 6 | 3 | 0.02092 | 0.001 | 0.8974 | 6.25 | 6.25 |
| 12 | 3 | 0.00024 | 0.001 | 0.922 | 0.42 | 0.42 |
| 24 | 5 | 0.00154 | 0.001 | 0.8556 | 0.83 | 0.83 |
| 48 | 6 | 0.00132 | 0.001 | 0.8699 | 0.83 | 0.83 |
| 72 | 6 | 0.0018 | 0.001 | 0.8309 | 0.42 | 0.42 |

## 4. Static vs adapting baseline

```
{
  "skipped": "needs at least three test days"
}
```
