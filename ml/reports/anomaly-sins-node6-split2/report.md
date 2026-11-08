# Anomaly detection evaluation

## Data

```
{
  "calibration_hours": 67.34,
  "test_hours": 81.56,
  "fall_frames": 2849,
  "non_fall_frames": 2850,
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
  ]
}
```

## 2. Behavioural detectors

### `category`: untouched test days (3.4)

```
{
  "alerts_per_day": 0.0,
  "confident_alerts_per_day": 0.0,
  "evidence_per_day": 5.0,
  "alerts_by_kind": {}
}
```

### `category`, planted anomalies (seeds [0, 1, 2]), impact trigger only

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.9 |
| impact_burst:repeated_impacts | 30 | 29 | 0.967 | 100.3 | 140.2 |
| fall:post_impact_silence | 30 | 6 | 0.2 | 7.9 | 8.0 |
| fall:post_impact_inactivity | 30 | 2 | 0.067 | 1557.9 | 1558.0 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 6}

### `category`, planted anomalies (seeds [0, 1, 2]), ideal classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.9 |
| impact_burst:repeated_impacts | 30 | 29 | 0.967 | 100.3 | 140.2 |
| fall:post_impact_silence | 30 | 30 | 1.0 | 7.8 | 8.1 |
| fall:post_impact_inactivity | 30 | 30 | 1.0 | 124.8 | 125.0 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 6}

### `category`, planted anomalies (seeds [0, 1, 2]), weaker classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.9 |
| impact_burst:repeated_impacts | 30 | 29 | 0.967 | 100.3 | 140.2 |
| fall:post_impact_silence | 30 | 25 | 0.833 | 7.9 | 8.1 |
| fall:post_impact_inactivity | 30 | 24 | 0.8 | 124.8 | 125.0 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 6}; 0 of 48 false triggers became alerts

## 4. Static vs adapting baseline

```
{
  "static": {
    "alerts_per_day": 0.0,
    "confident_alerts_per_day": 0.0,
    "evidence_per_day": 5.0,
    "alerts_by_kind": {},
    "baseline_updates": 0,
    "frames_flagged_by_day": [
      0.00097,
      0.00123,
      0.00143,
      0.00191
    ],
    "target_flagged": 0.001
  },
  "adaptive": {
    "alerts_per_day": 0.0,
    "confident_alerts_per_day": 0.0,
    "evidence_per_day": 5.0,
    "alerts_by_kind": {},
    "baseline_updates": 1,
    "frames_flagged_by_day": [
      0.00097,
      0.00123,
      0.00102,
      0.00172
    ],
    "target_flagged": 0.001
  }
}
```
