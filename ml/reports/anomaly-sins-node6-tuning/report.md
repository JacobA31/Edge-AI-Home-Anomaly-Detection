# Anomaly detection evaluation

> Threshold sweep, run against the defaults before tuning (`post_impact.short_seconds` 5,
> `impacts.min_count` 4). It led to 3 and 6; see docs/anomaly-detection.md, section 6.

## Data

```
{
  "calibration_hours": 72.0,
  "test_hours": 57.56,
  "fall_frames": 2849,
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
    1
  ]
}
```

## 6. Threshold tuning (each variant recalibrated, ideal classifier stand-in)

| variant | alerts_per_day | alerts_by_kind | evidence_per_day | fall_silence_rate | fall_silence_s_median | fall_silence_s_max | fall_inactivity_rate | fall_inactivity_s_median | fall_silence_rate_no_classifier | water_rate | water_s_median | burst_rate | burst_s_median | stray_alerts_per_day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| defaults | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| short_4s | 0.42 | {"repeated_impacts": 1} | 3.75 | 1.0 | 8.9 | 10.3 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| short_3s | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 7.9 | 9.4 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| strong_max_60s | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 64.7 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| strong_max_300s | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 304.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| quantile_0.995 | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| quantile_0.9995 | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| impact_on_0.1 | 6.25 | {"post_impact_inactivity": 11, "repeated_impacts": 4} | 15.01 | 1.0 | 9.9 | 11.8 | 1.0 | 109.5 | 0.4 | 1.0 | 119.6 | 0.375 | 60.4 | 5.21 |
| impact_on_0.3 | 0.0 | {} | 2.5 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 0.562 | 140.3 | 0.0 |
| impacts_target_0.01 | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| impacts_min_6 | 0.0 | {} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 100.3 | 0.0 |
| activity_margin_3db | 1.67 | {"post_impact_inactivity": 3, "repeated_impacts": 1} | 2.92 | 1.0 | 9.9 | 11.8 | 1.0 | 64.7 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 1.67 |
| activity_margin_10db | 0.42 | {"repeated_impacts": 1} | 6.25 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.2 | 0.42 |
| timezone_utc | 0.42 | {"repeated_impacts": 1} | 5.42 | 1.0 | 9.9 | 11.8 | 1.0 | 124.8 | 0.1 | 1.0 | 119.6 | 1.0 | 80.1 | 0.42 |
