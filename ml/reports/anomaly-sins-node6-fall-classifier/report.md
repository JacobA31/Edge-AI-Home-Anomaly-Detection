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

## 2. Behavioural detectors

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
| fall:post_impact_silence | 30 | 8 | 0.267 | 8.0 | 8.2 |
| fall:post_impact_inactivity | 30 | 2 | 0.067 | 509.3 | 893.9 |
| impact_burst:repeated_impacts | 24 | 24 | 1.0 | 100.2 | 120.3 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 11}

### `category`, planted anomalies (seeds [0, 1, 2]), ideal classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| fall:classifier_notified | 30 | 30 | 1.0 | 0.0 | 0.0 |
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.8 |
| fall:post_impact_silence | 30 | 30 | 1.0 | 7.9 | 10.6 |
| fall:post_impact_inactivity | 30 | 30 | 1.0 | 124.8 | 127.5 |
| impact_burst:repeated_impacts | 24 | 24 | 1.0 | 100.2 | 120.3 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 11}

### `category`, planted anomalies (seeds [0, 1, 2]), weaker classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| fall:classifier_notified | 30 | 24 | 0.8 | 2.0 | 2.0 |
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.8 |
| fall:post_impact_silence | 30 | 26 | 0.867 | 7.9 | 10.6 |
| fall:post_impact_inactivity | 30 | 24 | 0.8 | 124.8 | 127.5 |
| impact_burst:repeated_impacts | 24 | 24 | 1.0 | 100.2 | 120.3 |

Stray alerts per day: 0.14 ({"by_kind": {"post_impact_inactivity": 1}, "confident": 1}); side alerts caused by planted events: {"post_impact_inactivity": 11}; 1 of 34 false triggers became alerts

### `category`, planted anomalies (seeds [0, 1, 2]), real classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| fall:classifier_notified | 30 | 28 | 0.933 | 1.8 | 2.8 |
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.8 |
| fall:post_impact_silence | 30 | 28 | 0.933 | 7.9 | 8.2 |
| fall:post_impact_inactivity | 30 | 28 | 0.933 | 124.8 | 125.1 |
| impact_burst:repeated_impacts | 24 | 24 | 1.0 | 100.2 | 120.3 |

Stray alerts per day: 1.39 ({"by_kind": {"post_impact_inactivity": 10}, "confident": 10}); side alerts caused by planted events: {"post_impact_inactivity": 11}; 10 of 2228 false triggers became alerts

### Real fall classifier on the untouched test days (threshold 0.66)

123 false triggers (51.29/day), 4 became alerts. With the classifier connected: 1.67 alerts/day ({"post_impact_inactivity": 4}).

| local_time | score | top_class | categories | level_dbfs | became_alert |
|---|---|---|---|---|---|
| Fri 09:47:40 | 0.936 | Thunderstorm | [] | -43.7 | False |
| Fri 09:57:30 | 0.703 | Outside, rural or natural | [] | -45.9 | False |
| Fri 09:59:04 | 0.748 | Vehicle | [] | -41.1 | False |
| Fri 09:59:10 | 0.993 | Thunderstorm | [] | -38.4 | False |
| Fri 09:59:48 | 0.993 | Thunderstorm | [] | -43.6 | False |
| Fri 10:46:23 | 0.715 | Clip-clop | ['footsteps'] | -40.1 | False |
| Fri 10:46:43 | 0.996 | Music | [] | -36.2 | False |
| Fri 10:46:58 | 0.856 | Thunderstorm | [] | -41.7 | False |
| Fri 10:48:40 | 0.788 | Inside, small room | [] | -28.6 | False |
| Fri 10:49:08 | 0.709 | Inside, large room or hall | [] | -42.8 | False |
| Fri 10:49:36 | 0.961 | Clip-clop | [] | -37.6 | False |
| Fri 10:49:54 | 0.893 | Eruption | [] | -39.4 | False |
| Fri 10:50:00 | 0.679 | Inside, large room or hall | [] | -31.3 | False |
| Fri 10:50:24 | 0.692 | Hammer | [] | -35.4 | False |
| Fri 11:07:31 | 0.77 | Inside, small room | [] | -35.1 | False |
| Fri 11:08:05 | 0.691 | Dishes, pots, and pans | ['household'] | -27.2 | False |
| Fri 11:58:01 | 0.911 | Inside, large room or hall | [] | -41.9 | False |
| Fri 11:58:07 | 0.957 | Thunderstorm | [] | -41.4 | False |
| Fri 11:58:49 | 0.793 | Vehicle | [] | -37.3 | False |
| Fri 12:44:17 | 0.669 | Tambourine | [] | -24.4 | False |
| Fri 12:48:19 | 0.667 | Rail transport | [] | -25.2 | False |
| Fri 12:50:36 | 0.954 | Inside, large room or hall | [] | -20.4 | False |
| Fri 13:06:44 | 0.705 | Percussion | ['media'] | -38.6 | False |
| Fri 13:09:11 | 0.752 | Inside, small room | [] | -35.3 | False |
| Fri 13:11:59 | 0.747 | Hammer | [] | -26.9 | False |
| Fri 13:20:43 | 0.989 | Percussion | [] | -24.1 | False |
| Fri 13:26:28 | 0.801 | Inside, small room | [] | -21.9 | False |
| Fri 13:31:04 | 0.986 | Speech | ['speech'] | -32.3 | False |
| Fri 13:31:11 | 0.958 | Speech | ['speech'] | -19.5 | False |
| Fri 14:24:36 | 0.992 | Rustle | [] | -41.2 | False |
| Fri 14:24:43 | 0.981 | Thunderstorm | [] | -41.1 | False |
| Fri 14:32:21 | 0.716 | Thunderstorm | [] | -47.6 | False |
| Fri 14:38:12 | 0.682 | Rustle | [] | -44.9 | False |
| Fri 14:38:26 | 1.0 | Thunderstorm | [] | -32.8 | False |
| Fri 14:39:04 | 0.898 | Inside, small room | [] | -44.8 | False |
| Fri 14:39:19 | 0.857 | Thunderstorm | [] | -36.4 | False |
| Fri 14:58:20 | 0.989 | Music | ['media'] | -20.5 | False |
| Fri 14:59:21 | 0.718 | Speech | ['speech'] | -21.1 | False |
| Fri 14:59:39 | 0.896 | Drawer open or close | ['speech', 'household'] | -37.6 | False |
| Fri 15:00:26 | 0.928 | Speech | ['speech'] | -27.7 | False |
| Fri 15:03:16 | 0.998 | Drum | [] | -23.7 | False |
| Fri 17:46:55 | 0.922 | Tambourine | [] | -31.2 | False |
| Fri 17:50:09 | 0.687 | Cutlery, silverware | ['household'] | -24.7 | False |
| Fri 17:51:21 | 0.755 | Microwave oven | [] | -27.3 | False |
| Fri 17:51:34 | 0.974 | Percussion | ['media'] | -30.8 | False |
| Fri 17:51:48 | 0.999 | Percussion | [] | -27.2 | False |
| Fri 17:54:24 | 0.75 | Sliding door | ['household'] | -29.7 | False |
| Fri 17:55:32 | 0.905 | Dishes, pots, and pans | [] | -33.0 | False |
| Fri 17:58:57 | 0.69 | Percussion | [] | -25.4 | False |
| Fri 18:09:34 | 0.931 | Drawer open or close | ['household'] | -29.4 | False |
| Fri 18:26:23 | 0.918 | Drum | ['media'] | -21.4 | False |
| Fri 20:42:06 | 0.704 | Speech | ['speech'] | -35.0 | False |
| Fri 20:50:48 | 0.84 | Music | ['media'] | -30.4 | False |
| Fri 20:53:33 | 0.693 | Music | ['media'] | -37.2 | False |
| Fri 21:02:46 | 0.871 | Music | ['media'] | -33.2 | False |
| Fri 21:03:04 | 0.914 | Music | ['media'] | -36.0 | False |
| Fri 22:51:41 | 0.696 | Inside, small room | [] | -32.6 | False |
| Fri 22:51:58 | 0.876 | Percussion | [] | -39.0 | False |
| Fri 22:53:10 | 0.797 | Music | ['media'] | -38.2 | False |
| Fri 22:57:00 | 0.968 | Percussion | [] | -21.6 | False |
| Sat 02:15:48 | 0.75 | Bathtub (filling or washing) | ['water'] | -23.9 | False |
| Sat 05:18:28 | 0.693 | Music | [] | -38.8 | False |
| Sat 11:44:19 | 0.895 | Sliding door | ['household'] | -29.6 | False |
| Sat 11:44:31 | 0.722 | Inside, small room | [] | -23.4 | False |
| Sat 11:49:10 | 0.758 | Stir | ['household'] | -41.0 | False |
| Sat 11:58:35 | 0.883 | Dishes, pots, and pans | [] | -20.2 | False |
| Sat 12:05:40 | 0.985 | Vehicle | [] | -26.8 | False |
| Sat 12:06:03 | 0.994 | Music | [] | -29.5 | False |
| Sat 12:10:30 | 0.855 | Toilet flush | ['household'] | -22.2 | False |
| Sat 13:41:23 | 0.99 | Percussion | ['media'] | -26.0 | False |
| Sat 13:42:30 | 0.994 | Percussion | [] | -37.4 | False |
| Sat 13:58:34 | 0.793 | Inside, small room | [] | -32.9 | False |
| Sat 14:36:33 | 0.668 | Inside, small room | [] | -28.0 | False |
| Sat 14:38:11 | 0.996 | Drum | [] | -29.1 | False |
| Sat 14:39:40 | 0.848 | Inside, large room or hall | [] | -34.5 | False |
| Sat 14:45:52 | 0.663 | Inside, small room | [] | -25.1 | False |
| Sat 14:56:20 | 0.997 | Tambourine | [] | -21.5 | False |
| Sat 14:58:20 | 0.963 | Vehicle | [] | -31.7 | False |
| Sat 15:11:22 | 0.97 | Rail transport | [] | -28.5 | False |
| Sat 15:11:37 | 0.771 | Dishes, pots, and pans | ['household'] | -27.0 | False |
| Sat 15:12:45 | 0.839 | Drum | ['media'] | -20.3 | False |
| Sat 15:13:04 | 0.726 | Door | ['household'] | -32.3 | False |
| Sat 15:13:17 | 0.936 | Inside, small room | [] | -20.0 | False |
| Sat 15:14:11 | 0.992 | Beatboxing | [] | -19.6 | False |
| Sat 15:16:22 | 0.976 | Inside, small room | [] | -21.6 | False |
| Sat 15:26:24 | 0.991 | Percussion | [] | -34.4 | False |
| Sat 15:26:33 | 0.95 | Fireworks | [] | -20.6 | False |
| Sat 16:19:23 | 0.848 | Toilet flush | ['household'] | -30.2 | True |
| Sat 16:31:41 | 0.687 | Inside, small room | [] | -29.0 | False |
| Sat 16:56:08 | 0.892 | Car | [] | -27.0 | False |
| Sat 17:17:59 | 0.985 | Animal | [] | -33.2 | False |
| Sat 17:34:04 | 0.942 | Music | ['media'] | -35.0 | False |
| Sat 18:29:55 | 0.824 | Music | ['media'] | -33.3 | False |
| Sat 19:08:20 | 0.807 | Music | ['media'] | -43.3 | False |
| Sat 20:06:45 | 0.732 | Inside, small room | [] | -38.2 | False |
| Sat 22:03:36 | 0.864 | Inside, small room | [] | -26.3 | False |
| Sat 22:04:51 | 0.883 | Percussion | [] | -22.3 | False |
| Sat 22:06:17 | 0.878 | Dishes, pots, and pans | ['household'] | -32.5 | False |
| Sat 22:42:41 | 0.968 | Inside, large room or hall | [] | -25.7 | False |
| Sat 22:44:37 | 0.965 | Speech | [] | -33.5 | False |
| Sat 22:48:46 | 0.81 | Inside, small room | [] | -31.9 | False |
| Sat 22:48:57 | 0.802 | Inside, small room | [] | -36.1 | True |
| Sat 22:51:18 | 0.741 | Percussion | [] | -44.0 | False |
| Sat 22:53:21 | 0.989 | Percussion | ['media'] | -22.6 | False |
| Sat 23:18:36 | 0.97 | Inside, small room | [] | -21.8 | False |
| Sat 23:18:49 | 0.919 | Drum | [] | -35.8 | True |
| Sun 00:04:47 | 0.966 | Thump, thud | ['impact', 'household'] | -33.1 | False |
| Sun 02:03:38 | 0.827 | Music | ['media'] | -43.2 | False |
| Sun 02:49:53 | 0.81 | Door | ['household'] | -22.8 | False |
| Sun 08:31:28 | 0.999 | Timpani | ['media'] | -35.0 | False |
| Sun 08:31:39 | 0.998 | Timpani | ['media'] | -40.0 | True |
| Sun 08:36:20 | 0.896 | Drum | [] | -20.5 | False |
| Sun 08:56:58 | 0.928 | Thump, thud | ['impact'] | -42.6 | False |
| Sun 08:57:16 | 0.97 | Thump, thud | ['impact'] | -46.4 | False |
| Sun 09:00:53 | 0.983 | Music | ['media'] | -19.4 | False |
| Sun 09:01:05 | 0.737 | Inside, small room | [] | -21.9 | False |
| Sun 09:02:09 | 0.989 | Music | ['media'] | -34.3 | False |
| Sun 09:02:22 | 0.957 | Drum | [] | -30.7 | False |
| Sun 09:03:23 | 0.915 | Percussion | [] | -39.4 | False |
| Sun 09:19:02 | 0.736 | Drum | ['media'] | -37.2 | False |
| Sun 09:20:17 | 0.796 | Thump, thud | ['impact'] | -40.1 | False |
| Sun 09:20:23 | 0.93 | Door | [] | -20.3 | False |
| Sun 09:32:46 | 0.857 | Percussion | [] | -21.0 | False |

## 5. Fall classifier thresholds

`real_<t>`: the real classifier notifying at threshold t (planted falls and planted test days; `untouched_*`: the same on the untouched test days). `ideal`: the stand-in, for reference.

| classifier | post_impact_silence_detection_rate | post_impact_silence_latency_s_max | post_impact_inactivity_detection_rate | post_impact_inactivity_latency_s_max | stray_alerts_per_day | false_triggers | false_trigger_alerts | notified_rate | notified_s_median | notified_s_max | untouched_alerts_per_day | false_triggers_per_day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ideal | 1.0 | 10.6 | 1.0 | 127.5 | 0.0 |  |  | 1.0 | 0.0 | 0.0 |  |  |
| real_0.5 | 0.967 | 10.6 | 0.967 | 127.5 | 2.22 | 3005 | 6 | 0.967 | 1.8 | 2.5 | 2.5 | 72.97 |
| real_0.66 | 0.933 | 8.2 | 0.933 | 125.1 | 1.39 | 2228 | 4 | 0.933 | 1.8 | 2.8 | 1.67 | 51.29 |
| real_0.8 | 0.867 | 8.2 | 0.867 | 125.1 | 1.81 | 1506 | 5 | 0.833 | 2.0 | 2.8 | 2.08 | 37.11 |
| real_0.9 | 0.867 | 8.2 | 0.867 | 125.1 | 0.97 | 1450 | 3 | 0.8 | 2.0 | 2.8 | 1.25 | 28.77 |
| real_0.95 | 0.767 | 8.2 | 0.7 | 893.9 | 0.83 | 1412 | 2 | 0.633 | 2.2 | 2.8 | 0.83 | 22.52 |
| real_0.99 | 0.533 | 8.2 | 0.4 | 893.9 | 0.42 | 1334 | 1 | 0.367 | 2.0 | 2.9 | 0.42 | 11.26 |
