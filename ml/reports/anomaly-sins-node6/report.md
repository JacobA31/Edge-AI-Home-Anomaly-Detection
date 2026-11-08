# Anomaly detection evaluation

## Data

```
{
  "calibration_hours": 72.0,
  "test_hours": 57.56,
  "fall_frames": 2849,
  "non_fall_frames": 2850,
  "embedding_dim": 1024,
  "class_scores": true,
  "levels": true,
  "timezone": "Europe/Brussels",
  "contexts": [
    "category",
    "global",
    "global_pca64",
    "period"
  ],
  "plant_seeds": [
    0,
    1,
    2
  ]
}
```

## 1. Frame level: SAFE falls vs held-out household frames

| method | edge_ready | auc | pauc_0.1 | tpr_at_fpr_1% | anomalies_caught | false_alarm_frames_per_hour | model_mb | score_us_per_frame |
|---|---|---|---|---|---|---|---|---|
| edge_category | True | 0.8309 | 0.3403 | 0.1169 | 0.026 | 13.38 | 4.94 | 4.2 |
| edge_global | True | 0.834 | 0.3442 | 0.1425 | 0.0565 | 13.53 | 4.41 | 1.1 |
| edge_global_pca64 | True | 0.909 | 0.4427 | 0.1474 | 0.0337 | 11.89 | 0.47 | 0.5 |
| edge_period | True | 0.8589 | 0.3848 | 0.1362 | 0.034 | 12.79 | 4.95 | 3.5 |
| kmeans8_selective | False | 0.8321 | 0.3164 | 0.1099 | 0.0291 | 11.6 | 5.29 | 7.9 |
| isolation_forest | False | 0.3716 | 0.0002 | 0.0 | 0.0 | 29.59 |  | 11.0 |
| knn5_pca64 | False | 0.8532 | 0.357 | 0.1207 | 0.0316 | 9.81 | 5.38 | 11.1 |
| euclidean | False | 0.7652 | 0.0021 | 0.0 | 0.0 | 29.74 |  | 1.1 |

SAFE non-falls (same recordings, no fall) as a control: if non-falls separate from SINS about as well as falls do, the separation is mostly the recording setup.

| method | auc | auc_non_falls | auc_falls_vs_non_falls | anomalies_caught | non_falls_flagged |
|---|---|---|---|---|---|
| edge_category | 0.8309 | 0.7842 | 0.5506 | 0.026 | 0.0449 |
| edge_global | 0.834 | 0.7732 | 0.5787 | 0.0565 | 0.0544 |
| edge_global_pca64 | 0.909 | 0.882 | 0.5541 | 0.0337 | 0.0512 |
| edge_period | 0.8589 | 0.8079 | 0.5647 | 0.034 | 0.0509 |
| kmeans8_selective | 0.8321 | 0.7767 | 0.5608 | 0.0291 | 0.0453 |
| isolation_forest | 0.3716 | 0.363 | 0.537 | 0.0 | 0.0053 |
| knn5_pca64 | 0.8532 | 0.8115 | 0.5234 | 0.0316 | 0.0726 |
| euclidean | 0.7652 | 0.7374 | 0.5533 | 0.0 | 0.0049 |

The home's own sounds flagged as anomalous (share of test frames where YAMNet hears the category; lower is better, ADR 0002):

| method | flagged_impact | flagged_water | flagged_household |
|---|---|---|---|
| edge_category | 0.0149 | 0.0 | 0.0013 |
| edge_global | 0.0149 | 0.0258 | 0.0134 |
| edge_global_pca64 | 0.0597 | 0.0554 | 0.033 |
| edge_period | 0.0299 | 0.048 | 0.0399 |
| kmeans8_selective | 0.0149 | 0.0443 | 0.0138 |
| isolation_forest | 0.0 | 0.0 | 0.0 |
| knn5_pca64 | 0.0597 | 0.0111 | 0.0162 |
| euclidean | 0.0 | 0.0 | 0.0 |

## 2. Behavioural detectors

By embedding context (ideal classifier stand-in):

| context | alerts_per_day | alerts_by_kind | fall_silence_rate | fall_inactivity_rate | water_rate | burst_rate | stray_alerts_per_day |
|---|---|---|---|---|---|---|---|
| category | 0.0 | {} | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| global | 0.0 | {} | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| global_pca64 | 0.0 | {} | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| period | 0.0 | {} | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

### `category`: untouched test days (2.4)

```
{
  "alerts_per_day": 0.0,
  "confident_alerts_per_day": 0.0,
  "evidence_per_day": 5.42,
  "alerts_by_kind": {}
}
```

### `category`, planted anomalies (seeds [0, 1, 2]), impact trigger only

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.8 |
| fall:post_impact_silence | 30 | 4 | 0.133 | 7.9 | 8.1 |
| fall:post_impact_inactivity | 30 | 1 | 0.033 | 893.4 | 893.4 |
| impact_burst:repeated_impacts | 24 | 24 | 1.0 | 100.2 | 120.3 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 11}

### `category`, planted anomalies (seeds [0, 1, 2]), ideal classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.8 |
| fall:post_impact_silence | 30 | 30 | 1.0 | 7.9 | 9.4 |
| fall:post_impact_inactivity | 30 | 30 | 1.0 | 124.8 | 126.6 |
| impact_burst:repeated_impacts | 24 | 24 | 1.0 | 100.2 | 120.3 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 11}

### `category`, planted anomalies (seeds [0, 1, 2]), weaker classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.8 |
| fall:post_impact_silence | 30 | 26 | 0.867 | 7.9 | 9.0 |
| fall:post_impact_inactivity | 30 | 24 | 0.8 | 124.7 | 125.2 |
| impact_burst:repeated_impacts | 24 | 24 | 1.0 | 100.2 | 120.3 |

Stray alerts per day: 0.14 ({"by_kind": {"post_impact_inactivity": 1}, "confident": 1}); side alerts caused by planted events: {"post_impact_inactivity": 11}; 1 of 34 false triggers became alerts

## 3. Calibration length

| calibration_hours | components | test_frames_flagged | target_flagged | auc | alerts_per_day | confident_alerts_per_day |
|---|---|---|---|---|---|---|
| 6 | 3 | 0.02092 | 0.001 | 0.8974 | 5.42 | 5.42 |
| 12 | 3 | 0.00024 | 0.001 | 0.922 | 0.0 | 0.0 |
| 24 | 5 | 0.00154 | 0.001 | 0.8556 | 0.42 | 0.42 |
| 48 | 6 | 0.00132 | 0.001 | 0.8699 | 0.42 | 0.42 |
| 72 | 6 | 0.0018 | 0.001 | 0.8309 | 0.0 | 0.0 |

## 4. Static vs adapting baseline

```
{
  "skipped": "needs at least three test days"
}
```

## 5. Fall classifier stand-ins

Each stand-in changes the ideal one in one way. False triggers land on the home's own impact/household sounds.

| stand_in | post_impact_silence_detection_rate | post_impact_silence_latency_s_max | post_impact_inactivity_detection_rate | post_impact_inactivity_latency_s_max | stray_alerts_per_day | false_triggers | false_trigger_alerts |
|---|---|---|---|---|---|---|---|
| ideal | 1.0 | 9.4 | 1.0 | 126.6 | 0.0 |  |  |
| delay_2s | 1.0 | 9.4 | 1.0 | 126.6 | 0.0 |  |  |
| delay_5s | 1.0 | 9.4 | 1.0 | 126.6 | 0.0 |  |  |
| recall_0.8 | 0.867 | 9.4 | 0.833 | 893.4 | 0.0 |  |  |
| recall_0.5 | 0.6 | 9.4 | 0.533 | 893.4 | 0.0 |  |  |
| false_1_per_day | 1.0 | 9.4 | 1.0 | 126.6 | 0.0 | 5 | 0 |
| false_5_per_day | 1.0 | 9.4 | 1.0 | 126.6 | 0.0 | 31 | 0 |
| false_20_per_day | 1.0 | 9.4 | 1.0 | 126.6 | 0.0 | 130 | 0 |
