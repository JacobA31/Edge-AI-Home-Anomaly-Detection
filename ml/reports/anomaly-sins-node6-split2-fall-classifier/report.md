# Anomaly detection evaluation

## Data

```
{
  "calibration_hours": 67.34,
  "test_hours": 81.56,
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
| fall:post_impact_silence | 30 | 9 | 0.3 | 7.9 | 8.1 |
| fall:post_impact_inactivity | 30 | 3 | 0.1 | 1557.8 | 1557.9 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 6}

### `category`, planted anomalies (seeds [0, 1, 2]), ideal classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| fall:classifier_notified | 30 | 30 | 1.0 | 0.0 | 0.0 |
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.9 |
| impact_burst:repeated_impacts | 30 | 29 | 0.967 | 100.3 | 140.2 |
| fall:post_impact_silence | 30 | 30 | 1.0 | 7.9 | 11.1 |
| fall:post_impact_inactivity | 30 | 30 | 1.0 | 124.8 | 127.9 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 6}

### `category`, planted anomalies (seeds [0, 1, 2]), weaker classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| fall:classifier_notified | 30 | 24 | 0.8 | 2.0 | 2.0 |
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.9 |
| impact_burst:repeated_impacts | 30 | 29 | 0.967 | 100.3 | 140.2 |
| fall:post_impact_silence | 30 | 27 | 0.9 | 7.9 | 10.7 |
| fall:post_impact_inactivity | 30 | 24 | 0.8 | 124.8 | 127.7 |

Stray alerts per day: 0.0 ({"by_kind": {}, "confident": 0}); side alerts caused by planted events: {"post_impact_inactivity": 6}; 0 of 48 false triggers became alerts

### `category`, planted anomalies (seeds [0, 1, 2]), real classifier

| planted:finding | planted | detected | detection_rate | latency_s_median | latency_s_max |
|---|---|---|---|---|---|
| fall:classifier_notified | 30 | 28 | 0.933 | 2.0 | 2.8 |
| prolonged_water:prolonged_water | 30 | 30 | 1.0 | 119.6 | 119.9 |
| impact_burst:repeated_impacts | 30 | 29 | 0.967 | 100.3 | 140.2 |
| fall:post_impact_silence | 30 | 28 | 0.933 | 7.9 | 11.1 |
| fall:post_impact_inactivity | 30 | 28 | 0.933 | 124.8 | 127.9 |

Stray alerts per day: 1.47 ({"by_kind": {"post_impact_inactivity": 15}, "confident": 15}); side alerts caused by planted events: {"post_impact_inactivity": 7}; 15 of 2865 false triggers became alerts

### Real fall classifier on the untouched test days (threshold 0.66)

190 false triggers (55.91/day), 5 became alerts. With the classifier connected: 1.47 alerts/day ({"post_impact_inactivity": 5}).

| local_time | score | top_class | categories | level_dbfs | became_alert |
|---|---|---|---|---|---|
| Thu 00:12:30 | 0.786 | Inside, small room | [] | -41.9 | False |
| Thu 03:19:56 | 0.896 | Inside, small room | [] | -33.2 | False |
| Thu 07:57:46 | 0.973 | Speech | [] | -21.3 | False |
| Thu 07:59:43 | 0.727 | Percussion | [] | -33.4 | False |
| Thu 08:06:08 | 0.733 | Inside, small room | [] | -38.5 | False |
| Thu 08:15:50 | 0.919 | Inside, small room | [] | -23.2 | False |
| Thu 08:23:37 | 0.987 | Thump, thud | ['impact'] | -33.2 | False |
| Thu 08:25:06 | 0.899 | Inside, small room | [] | -20.5 | False |
| Thu 09:23:54 | 0.999 | Thunderstorm | ['water'] | -46.3 | False |
| Thu 09:24:25 | 0.933 | Thunderstorm | [] | -41.5 | False |
| Thu 09:30:27 | 0.705 | Explosion | [] | -40.1 | False |
| Thu 09:47:40 | 0.989 | Thunderstorm | [] | -42.9 | False |
| Thu 09:47:52 | 0.983 | Thunderstorm | [] | -38.5 | False |
| Thu 09:58:53 | 0.917 | Heart sounds, heartbeat | [] | -40.3 | False |
| Thu 09:59:45 | 0.828 | Thunderstorm | [] | -48.3 | False |
| Thu 10:11:41 | 0.851 | Thunderstorm | [] | -44.7 | False |
| Thu 11:17:00 | 0.712 | Vehicle | [] | -32.4 | False |
| Thu 13:46:41 | 0.989 | Thunderstorm | [] | -43.1 | False |
| Thu 13:48:32 | 0.998 | Thunderstorm | [] | -46.4 | False |
| Thu 13:48:53 | 0.952 | Thunderstorm | ['water'] | -47.8 | False |
| Thu 13:49:07 | 0.907 | Thunderstorm | [] | -44.7 | False |
| Thu 14:00:56 | 0.922 | Arrow | ['impact'] | -43.3 | False |
| Thu 14:02:25 | 0.751 | Rumble | [] | -44.2 | False |
| Thu 14:02:33 | 0.994 | Music | [] | -39.1 | False |
| Thu 14:03:02 | 0.718 | Thunderstorm | ['water'] | -44.4 | False |
| Thu 14:03:17 | 0.929 | Thunderstorm | [] | -41.2 | False |
| Thu 14:03:24 | 0.863 | Thunderstorm | ['water'] | -42.1 | False |
| Thu 14:03:37 | 0.975 | Outside, rural or natural | [] | -43.7 | False |
| Thu 14:06:27 | 0.981 | Rumble | [] | -40.9 | False |
| Thu 14:08:29 | 1.0 | Thunderstorm | [] | -37.5 | False |
| Thu 14:08:45 | 0.99 | Thunderstorm | [] | -35.0 | False |
| Thu 14:15:10 | 0.969 | Rustle | [] | -42.8 | False |
| Thu 14:15:52 | 0.978 | Thunderstorm | [] | -35.5 | False |
| Thu 14:44:04 | 1.0 | Thunderstorm | [] | -43.1 | False |
| Thu 14:44:22 | 0.971 | Thunderstorm | [] | -40.0 | False |
| Thu 14:59:36 | 0.943 | Inside, small room | [] | -19.5 | False |
| Thu 15:05:13 | 0.946 | Hiss | [] | -20.5 | False |
| Thu 17:31:10 | 0.884 | Door | ['household'] | -18.6 | False |
| Thu 17:36:39 | 0.704 | Percussion | [] | -36.8 | False |
| Thu 17:36:57 | 0.684 | Inside, small room | [] | -20.3 | False |
| Thu 17:37:19 | 0.864 | Inside, small room | [] | -22.0 | False |
| Thu 17:39:29 | 0.861 | Breaking | ['impact'] | -15.2 | False |
| Thu 17:41:54 | 0.872 | Dishes, pots, and pans | ['household'] | -28.0 | False |
| Thu 17:46:57 | 0.956 | Percussion | [] | -27.8 | False |
| Thu 17:47:38 | 0.765 | Hammer | [] | -27.1 | False |
| Thu 17:53:23 | 0.955 | Inside, small room | [] | -35.7 | False |
| Thu 17:59:59 | 0.856 | Sliding door | [] | -19.6 | False |
| Thu 18:09:04 | 0.982 | Rail transport | ['water'] | -25.1 | False |
| Thu 18:10:53 | 0.976 | Vehicle | [] | -21.6 | False |
| Thu 18:15:14 | 0.69 | Music | ['media'] | -28.2 | False |
| Thu 18:28:11 | 0.782 | Dishes, pots, and pans | ['household'] | -31.2 | False |
| Thu 18:28:50 | 0.854 | Inside, large room or hall | [] | -26.4 | False |
| Thu 18:30:41 | 0.839 | Percussion | [] | -28.0 | False |
| Thu 18:34:46 | 0.769 | Dishes, pots, and pans | ['household'] | -20.5 | False |
| Thu 18:37:10 | 0.81 | Sliding door | ['impact', 'household'] | -34.0 | False |
| Thu 18:43:06 | 0.854 | Inside, small room | [] | -35.0 | False |
| Thu 18:43:44 | 0.773 | Hiss | [] | -39.4 | False |
| Thu 18:59:29 | 0.852 | Speech | [] | -33.7 | False |
| Thu 21:57:40 | 0.712 | Inside, small room | [] | -29.5 | False |
| Thu 21:57:50 | 0.683 | Percussion | [] | -38.5 | True |
| Thu 22:04:05 | 0.88 | Knock | ['impact', 'household'] | -20.6 | False |
| Thu 22:08:04 | 0.958 | Rail transport | [] | -23.2 | False |
| Thu 22:14:32 | 0.82 | Inside, small room | [] | -37.5 | False |
| Thu 22:31:32 | 0.696 | Ping | [] | -31.9 | False |
| Thu 22:33:48 | 0.694 | Sliding door | ['household'] | -34.4 | False |
| Thu 22:33:55 | 0.883 | Percussion | [] | -21.7 | False |
| Thu 22:36:26 | 0.925 | Music | ['media'] | -30.2 | False |
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
