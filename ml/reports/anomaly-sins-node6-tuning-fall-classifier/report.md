# Anomaly detection evaluation

## Data

```
{
  "calibration_hours": 72.0,
  "test_hours": 57.56,
  "fall_frames": 546,
  "non_fall_frames": 0,
  "embedding_dim": 1024,
  "class_scores": true,
  "levels": true,
  "timezone": "Europe/Brussels",
  "contexts": [
    "category"
  ],
  "plant_seeds": [
    0,
    1,
    2
  ],
  "safe_folds": [
    9,
    10
  ],
  "fall_classifier": {
    "model": "models/fall_classifier.tflite",
    "model_sha256": "84837ba61875db2ad8dd1490b1bfc5e9864e4bc851e2fc76856aa51e05399637",
    "yamnet_sha256": "3462e073dd7dd359afdbbefe273fed1cd29df2ca705f72a2af80d624095dc4c9",
    "threshold": 0.66
  }
}
```

## 6. Threshold tuning (each variant recalibrated, real fall classifier)

| variant | alerts_per_day | alerts_by_kind | evidence_per_day | fall_silence_rate | fall_silence_s_median | fall_silence_s_max | fall_inactivity_rate | fall_inactivity_s_median | fall_silence_rate_no_classifier | water_rate | water_s_median | burst_rate | burst_s_median | stray_alerts_per_day | notified_rate | notified_s_max | untouched_alerts_per_day | false_triggers_per_day | false_trigger_alerts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| defaults | 0.0 | {} | 5.42 | 0.933 | 7.9 | 8.2 | 0.933 | 124.8 | 0.267 | 1.0 | 119.6 | 1.0 | 100.2 | 1.39 | 0.933 | 2.8 | 1.67 | 51.29 | 4 |
| strong_max_60s | 0.0 | {} | 5.42 | 0.933 | 7.9 | 8.2 | 0.933 | 64.6 | 0.267 | 1.0 | 119.6 | 1.0 | 100.2 | 3.2 | 0.933 | 2.8 | 3.34 | 51.29 | 8 |
| strong_max_300s | 0.0 | {} | 5.42 | 0.933 | 7.9 | 8.2 | 0.933 | 304.7 | 0.267 | 1.0 | 119.6 | 1.0 | 100.2 | 0.0 | 0.933 | 2.8 | 0.0 | 51.29 | 0 |
