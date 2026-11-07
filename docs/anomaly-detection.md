# Acoustic and behavioural anomaly detection

Owner: Ihsan Aydin (backup: Matthew Ross), proposal Table 1.4. Code: `edge/src/e3_edge/anomaly/`
(runs on the Pi) and `ml/src/anomaly/` (offline evaluation). Design decision: [ADR 0002](decisions/0002-selective-mahalanobis-baseline.md).

## 1. What this module does, and what it doesn't

It learns the home's normal sounds and routine from YAMNet frames, then reports for each new frame
how unusual it is and which behavioural patterns just completed.

| Requirement | Covered by |
|---|---|
| FR4 baseline learning: common sounds, typical durations, frequency per observation period | `calibration.py`, `temporal.py` (`TemporalStats`), `baseline.py` |
| FR5 anomalies when patterns, durations or frequencies exceed thresholds from the baseline | `gaussian.py` (patterns), `temporal.py` (durations, frequencies) |
| 1.6.1 the three temporal anomalies: prolonged running water, frequent impacts in a window, fall-like impact then reduced activity | `ProlongedSound`, `ImpactRate`, `PostImpactMonitor` |
| 1.7.3.3 static baseline from a calibration period; recalibration as future work | static by default; gated adaptation behind `adaptation.enabled` |
| NF1 behavioural anomalies flagged within 5 min of crossing a threshold | checked every frame (≤ 1 hop after crossing) |
| NF1 fall budget 10 s: "short check for silence after the impact" | `post_impact_silence` evidence 5 s after the fall sound ends (planted SAFE falls on SINS: 9.9 s median, 11.8 s max from the start of the fall) |
| NF4 7 days without restart; NF5 ≤ 1 GB RAM, models < 20 MB | flat memory (nothing stored per frame), crash-safe checkpoints, ~5 MB baseline |
| NF10 privacy | input is embeddings, never audio; baseline stays on the device; events carry ≤ 10 small fields |

**Not in this module** (owned by others, see section 7): audio capture and mute button (Matthew),
preprocessing and YAMNet inference (Fareed), the fall classifier (Yannis), and the Event Manager's
final risk decision and alert delivery (FR7/FR8: event handling, Samy). The module never creates or
sends events itself. It hands `Finding`s to whoever runs the Event Manager.

## 2. Where it sits

```
mic ─► capture ─► preprocessing ─► YAMNet ──► Frame(t, embedding[1024], scores[521], level_dbfs)
                                     │                 │
                                     │                 ▼
                                     │        AnomalyService.process(frame)  (one per microphone)
                                     │                 │   calibrating: returns None
                                     ▼                 ▼   detecting:   AnomalyOutput
                               fall classifier ──► Event Manager ──► make_event() ─► outbox ─► backend
                                     │  notify_fall_candidate(t, conf) ▲
                                     └─────────────────────────────────┘
```

## 3. How it works

### 3.1 Embedding baseline: "how unusual is this sound here?"

* **Mahalanobis distance** of each embedding from Gaussians of the home's embeddings (proposal 1.7.3.2).
  Mean and covariance are accumulated in streaming batches (Chan et al.), in float64.
* **OAS shrinkage** keeps the 1024×1024 covariance invertible even when YAMNet dimensions are
  always zero. It needs only the covariance and frame count, so it works on streamed statistics.
  It matches scikit-learn's `OAS` exactly (tested).
* **Selective, category-conditioned** (`embedding.contexts: category`, default): one Gaussian for
  all frames, plus one per sound category fitted on the frames where YAMNet says that category is
  present. The score is the smallest distance. This is what makes a familiar but rare sound (the
  cane thud of 1.7.4) score low while a sound unlike anything in the home still scores high. Why
  this replaced the single Gaussian: ADR 0002.
* **Probabilistic PCA** for categories with few frames: top `min(32, n/20)` directions plus one
  residual variance, so the distance still reacts to sounds outside those directions.
* **Threshold and level**: calibration ends with a day of frames the model hasn't seen; their
  distances give the threshold (`frame_quantile`, default 0.999). They also give `level`, the share of
  normal frames closer to normal than this one (0..1), which the Event Manager can combine with the
  fall confidence.

### 3.2 Behavioural detectors: "is today's routine normal?"

YAMNet's class scores are grouped into categories (`categories.py`; class names verified against
the official `yamnet_class_map.csv`). Episodes use hysteresis (`on`/`off`) and tolerate short dropouts.

| Detector | Learned during calibration | Limit | Finding (event_type, role) |
|---|---|---|---|
| Prolonged running water | durations of water episodes | 99th percentile × 1.5, clipped to 2–60 min (15 min with < 5 episodes) | `prolonged_water` (anomaly, alert) at the limit, again at 2× |
| Repeated impacts | per 5-min window, most impacts seen, per time-of-day period, plus the impact rate | more than ever seen in that period, and at least the count a Poisson model of the learned rate expects ≤ 0.05×/day; minimum 4 | `repeated_impacts` (anomaly, alert) |
| Impact then silence | how long quiet lasts after ordinary impacts | 5 s check, counted from the first quiet frame after the trigger; ends if the room isn't quiet within 10 s | `post_impact_silence` (inactivity, **evidence**) |
| Impact then inactivity | (same) | 95th percentile × 1.5 (1–60 min); after a *strong* trigger (fall candidate or unfamiliar impact) at most 2 min | `post_impact_inactivity` (inactivity, alert) |
| Sustained unusual sound | uses the embedding threshold | ≥ 50% of a 30 s window anomalous, ≥ 10 frames | `sustained_unusual_sound` (anomaly, alert) |

"Activity" for the silence checks is "louder than the learned noise floor + 6 dB, or any category
present". Following 1.7.4, silence is only ever evidence. Confidence carries the trigger's
strength: 0.3 for a familiar thud, the anomaly level for an unfamiliar one, the classifier's
confidence for a fall candidate.

### 3.3 Calibration lifecycle (`AnomalyService`)

1. **stats** (default 3 days of audio): accumulate Gaussians and temporal statistics.
2. **threshold** (default 1 day): score unseen frames, keep a 50,000-value sample. This phase
   must be a whole day: a threshold set on evenings only flags far more than 0.1% of mornings.
   The CLI, the harness and adaptation therefore always use the last whole day when they have
   two days or more (`split_calibration`).
3. Save `baseline.npz` atomically (no pickles) and switch to detection.

Only observed time counts, so mute periods don't. A checkpoint is written every hour and resumed
after a crash or power cut. The baseline then stays **static** (1.7.3.3). With
`adaptation.enabled`, a shadow calibration runs over each `interval_seconds` (use at least 2 days,
so its threshold phase is a whole day). Its learning phase skips frames that scored unusual
(level ≥ 0.99), and nothing is used for 10 min after an alert. Its threshold phase keeps every
frame, because cutting off the tail set thresholds about 10% too low (caught by a test). The result
blends into the baseline (weight 0.2); sounds absent from that round keep their old model, and the
previous file is kept as `.prev.npz`.

**Calibration needs enough of every rare sound.** A category with fewer than 50 frames in the
learning phase gets no Gaussian of its own, so its ordinary sounds look unusual. In simulation,
confident false alerts stopped only once the learning phase was 2 days long (section 6). Check
`python -m e3_edge.anomaly info baseline.npz`: every common category should be listed.

### 3.4 Gaps and mute

A gap longer than 5 s between frames, or `pause()` on mute (FR10), ends all episodes and
watches. The mute is never mistaken for silence after an impact.

## 4. Output

`AnomalyOutput` per frame: `distance`, `level`, `is_anomalous`, `component` (which Gaussian
matched), `active`, `categories`, `findings`. A finding converts straight into a schema-valid event:

```python
make_event(cfg.device_id, cfg.room, **finding.to_event_fields())
# {"event_type": "anomaly", "confidence": 0.5, "details": {"kind": "prolonged_water", "role": "alert",
#   "value": 1113.6, "threshold": 1113.6, "unit": "s", "started_at": "2026-11-02T10:12:00Z", "stage": 1}}
```

## 5. Integration (for the main loop and the Event Manager)

```python
from e3_edge.anomaly import AnomalyConfig, AnomalyService, Frame, frame_level_dbfs

anomaly = AnomalyService(AnomalyConfig.from_dict(cfg.anomaly), mic_id="near")   # one per mic

# for every YAMNet frame (0.96 s window) of that mic:
frame = Frame(t=window_start_unix, embedding=emb, scores=yamnet_scores, level_dbfs=frame_level_dbfs(wave))
out = anomaly.process(frame)          # None while calibrating (the first ~4 days)
if fall_confidence >= FALL_CANDIDATE:  # from the fall classifier
    anomaly.notify_fall_candidate(frame.t, fall_confidence)
if out:
    for f in out.findings:             # Event Manager: combine with fall confidence (FR7), then
        ...                            # outbox.put(make_event(cfg.device_id, cfg.room, **f.to_event_fields()))
# mute pressed:  anomaly.pause()
```

`process()` takes about 0.2 ms per frame on a laptop (1024 dims, category mode), so it can run
in the inference thread.

## 6. Plan and status

| # | Task | Status |
|---|---|---|
| 1 | Input contract (`Frame`, frame-stream `.npz`), category map from YAMNet's class map | done |
| 2 | Streaming moments, OAS, Mahalanobis/PPCA, selective components, blending | done, tested against numpy/sklearn |
| 3 | Temporal detectors with learned thresholds and defaults | done, unit-tested |
| 4 | Two-phase calibration, checkpoint/resume, atomic baseline file, service lifecycle | done, tested |
| 5 | Detector: mute/gaps, fall-candidate hook, runtime re-tuning, optional adaptation | done, tested |
| 6 | CLI `calibrate` / `replay` / `info` / `bench` (also runs on the Pi) | done, tested |
| 7 | Offline harness: 8 methods, AUC/pAUC, planted anomalies, false alerts/day, calibration length, adaptation | done, tested on synthetic data |
| 8 | SINS replay with real YAMNet frames, SAFE falls; choose `contexts` and tune thresholds | **in progress**: first run on SINS Node 6 + SAFE done (below); `contexts` and thresholds not chosen yet |
| 9 | Measure on the Pi 5: `bench`, 7-day soak, combined with YAMNet + fall model (NF5) | **blocked**: needs hardware and the other models |
| 10 | Wire into the main loop and the Event Manager | **blocked**: needs the frame source and the Event Manager |

### First results on real frames (SINS Node 6 + SAFE)

Full report: [ml/reports/anomaly-sins-node6/report.md](../ml/reports/anomaly-sins-node6/report.md).
Calibrated on 31 Jan – 2 Feb 2017 (72 h), tested on 3–5 Feb (57.6 h, 2.4 days; 5 Feb ends at
09:33), category mode, SAFE's 475 fall clips as the planted falls.

| Check | Result |
|---|---|
| False alerts on untouched test days | 0.42/day (1 `repeated_impacts`) |
| Planted 45-min running tap / 8-impact burst | 10/10 at ~120 s / 8/8 at ~80 s |
| Planted fall, with fall-classifier notification | silence evidence 10/10 (9.9 s median, 11.8 s max); inactivity alert 10/10 at ~125 s |
| Planted fall, impact sound only (no classifier) | 1/10: YAMNet rarely puts SAFE falls in the impact category, so the watch never starts |
| Stray alerts on the days with planted events | 1.67/day, mostly `post_impact_inactivity` |
| Embedding-only fall separation (AUC) | global 64-dim PCA 0.909, period 0.859, kNN 0.853, global 0.834, category 0.831, k-means 0.832, Euclidean 0.765, Isolation Forest 0.372 |
| Falls caught by the embedding threshold alone | category 2.6%, global 5.7%, with ~12–13 normal frames flagged per hour |
| Calibration length | 6 h: 6.25 alerts/day; 12–72 h: 0.42–0.83/day |
| Static vs adapting baseline | not run: needs at least three test days |

What it changed: the post-impact watch used to start while the fall clip was still playing, so the
fall's own sound counted as activity and ended every watch at the 5 s check (1/10 planted falls
caught, even with the classifier). Quiet is now measured from the first quiet frame.

Caveats: SAFE was recorded with other microphones in other rooms, so part of the AUC may be the
recording setup rather than the fall. The SINS clock timezone is assumed to be Europe/Brussels;
the hourly profile looks like a normal home, but a ±1 h offset can't be ruled out.

### Results on synthetic frames (laptop; not accuracy claims)

Measured on an Apple-silicon laptop with Python 3.11; the Pi 5 will be slower and must be
re-measured (task 9). Simulated frames show that the code works as designed, not that it
detects real falls.

| Check | Result | Target |
|---|---|---|
| `process()` per frame, 1024 dims, category mode | 0.20 ms median, 0.21 ms p99, 0.42 ms max | model step < 200 ms (NF1) |
| 7 simulated days (1.26 M frames) through one detector | no crash; Python heap 46.453 → 46.462 MB (+0.009 MB) | 7 days without restart (NF4) |
| Baseline file, 1024 dims, global + 6 categories | 5.6 MB | models < 20 MB (NF5) |
| Memory of the module itself | detector: ~2–3 MB above the loaded baseline; calibrator: ~50 MB with 4 categories active, ~100 MB with all 8 | ≤ 1 GB for the whole pipeline (NF5) |
| Silence evidence after a planted fall | 10/10, 8.3 s median, 8.5 s max | fall event within 10 s (NF1) |
| Inactivity alert after a fall, with fall-classifier notification | 10/10 at ~122 s | — |
| Same, impact sound only (no classifier) | 6/10, after the 1 h learned limit | — |
| Planted 45-min running tap / 8-impact burst | 10/10 at the learned 823 s / 10/10 at 60–101 s | within 5 min of the threshold (NF1) |
| False alerts on 3 normal simulated days | 1.33/day, none with confidence ≥ 0.5 | — |
| Calibration length | 6 h: 42% of frames flagged; 24–48 h: 4–5 confident false alerts/day; 72 h: 0 | — |
| Embedding-only fall separation (AUC) | category 0.983, global 0.999, k-means 0.998, kNN 0.984, Isolation Forest 0.684 | — |
| Falls caught by the embedding threshold alone | category 17.5%, global 69% | trade-off, see ADR 0002 |

### Evaluation protocol (task 8)

Calibrate on the first SINS days of one room, replayed in time order, and test on the later
ones. Node 6 (living room, next to the kitchen sink) has five full days, 31 Jan – 4 Feb 2017, plus
parts of 30 Jan and 5 Feb, so the first run calibrates on 31 Jan – 2 Feb and tests on 3–5 Feb.
Run `python -m src.anomaly.evaluate --timezone Europe/Brussels` (see ml/README.md), which reports:
1. Each method's separation of SAFE falls from SINS frames (AUC, pAUC at FPR ≤ 0.1), and falls
   caught/missed and normal frames flagged per hour at the calibrated threshold.
2. Planted falls, running water and impact bursts in the test days: detection rate and latency,
   with and without the fall classifier's notification.
3. False alerts per day on untouched test days (SINS is assumed normal).
4. Results against calibration length (6 h to 5 days), and static vs adapting baseline.

## 7. What I need from teammates

| From | What | Why |
|---|---|---|
| Fareed (preprocessing/YAMNet) | **Done:** YAMNet frames for SINS and SAFE from `ml/src/export_frames.py` (0.96 s windows, 0.48 s hop, scores and levels); SINS Node 6 and SAFE exported, see ml/data/README.md | — |
| Fareed / anyone | The SINS recording clock's timezone (assumed Europe/Brussels, not confirmed) | Time-of-day periods and hourly statistics use local time |
| Yannis (fall classifier) | A fall-candidate confidence per frame and the threshold at which it counts | `notify_fall_candidate()` starts the post-fall silence watch; strong triggers use the 2 min limit |
| Matthew (edge integration) | A frame source in the main loop (thread layout), mute button events, one `mic_id` per microphone | `AnomalyService.process()` per frame; `pause()` on mute |
| Samy / whoever owns the Event Manager | Who owns the FR7 fusion (Table 1.4 assigns "event-handling" to Samy; the Gantt lists "Event Manager" without an owner). The risk formula and which findings alert | The module returns `level`, findings with `role` and `confidence`; it does not decide alerts |
| Samy (backend/dashboard) | Show `details.kind` and `details.started_at`; delayed alerts already carry `detected_at` | Findings use only existing schema fields (`anomaly`, `inactivity`) |
| Team | Agree whether `.github/workflows/ml.yml` may run the harness tests (added in this branch) | CI ownership is members 1 and 5 |

## 8. Limitations

* Real-audio numbers come from one SINS room (one person, one home, one week, 1.6.1) with SAFE
  falls planted into it, and only 2.4 test days. They are a first check, not a measure of accuracy
  in other homes.
* Without the fall classifier, few falls start the post-impact watch, because YAMNet seldom calls a
  fall an impact (1/10 on SAFE).
* Category detection inherits YAMNet's mistakes; the on/off thresholds are untuned starting points.
* Visitors and changed routines show up as anomalies (1.6.3); adaptation is off by default.
* A fall with the TV on isn't followed by silence, so only the fall classifier and the embedding
  score can catch it.
