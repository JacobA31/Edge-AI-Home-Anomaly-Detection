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
| NF1 fall budget 10 s: "short check for silence after the impact" | `post_impact_silence` evidence 3 s after the fall sound ends (planted SAFE falls on SINS: 7.9 s median, 9.4 s max from the start of the fall) |
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
| Repeated impacts | per 5-min window, most impacts seen, per time-of-day period, plus the impact rate | more than ever seen in that period, and at least the count a Poisson model of the learned rate expects ≤ 0.05×/day; minimum 6 | `repeated_impacts` (anomaly, alert) |
| Impact then silence | how long quiet lasts after ordinary impacts | 3 s check, counted from the first quiet frame after the trigger; ends if the room isn't quiet within 10 s | `post_impact_silence` (inactivity, **evidence**) |
| Impact then inactivity | (same) | 95th percentile × 1.5 (1–60 min); after a *strong* trigger (fall candidate or unfamiliar impact) at most 2 min | `post_impact_inactivity` (inactivity, alert) |
| Sustained unusual sound | uses the embedding threshold | ≥ 50% of a 30 s window anomalous, ≥ 10 frames | `sustained_unusual_sound` (anomaly, alert) |

"Activity" for the silence checks is "louder than the learned noise floor + 6 dB, or any category
present" (calibration uses the floor measured so far, so both phases agree). A post-impact watch
ends when more than 20% of it, or more than half of the last 30 s, was active. Following 1.7.4, silence is only ever evidence. Confidence carries the trigger's
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
| 8 | SINS replay with real YAMNet frames, SAFE falls; choose `contexts` and tune thresholds | done: `contexts: category` kept (ADR 0002 accepted); `post_impact.short_seconds` 5 → 3, `impacts.min_count` 4 → 6 (below) |
| 9 | Measure on the Pi 5: `bench`, 7-day soak, combined with YAMNet + fall model (NF5). Re-run the harness with the real fall classifier instead of the stand-ins: its candidate threshold, delay, misses and false triggers per day; re-check NF1 and `post_impact.strong_max_seconds` with them | **blocked**: needs hardware, the fall classifier and its outputs on SAFE/SINS |
| 10 | Wire into the main loop and the Event Manager | **blocked**: needs the frame source and the Event Manager |

### Results on real frames (SINS Node 6 + SAFE, task 8)

Reports: [main split](../ml/reports/anomaly-sins-node6/report.md) (calibrate 31 Jan – 2 Feb 2017, 72 h;
test 3–5 Feb, 2.4 days), [second split](../ml/reports/anomaly-sins-node6-split2/report.md) (calibrate
30 Jan 04:39 – 1 Feb, 66.8 h; test 2–5 Feb, 3.4 days) and [threshold sweep](../ml/reports/anomaly-sins-node6-tuning/report.md).
Category mode, final defaults, anomalies planted with 3 seeds (30 falls, 30 running taps, 24–30 bursts).
The fall classifier isn't built yet, so a stand-in notifies the planted falls (section "Evaluation protocol").

| Check | Main split | Second split |
|---|---|---|
| False alerts on untouched test days | 0/day (was 0.42 before tuning) | 0/day (was 0.59) |
| Planted 45-min running tap | 30/30 at 120 s | 30/30 at 120 s |
| Planted 8-impact burst | 24/24 at ~100 s | 29/30 at ~100 s |
| Planted fall, ideal classifier: silence evidence | 30/30, 7.9 s median, **9.4 s max** (NF1 ≤ 10 s) | 30/30, 7.8 s median, 8.1 s max |
| ...and inactivity alert | 30/30 at ~125 s | 30/30 at ~125 s |
| Planted fall, weaker classifier (80% recall, 2 s late, 5 false triggers/day) | evidence 26/30, alert 24/30; 1 of 34 false triggers became an alert | 25/30, 24/30; 0 of 48 |
| Planted fall, no classifier (impact sound only) | evidence 4/30 | 6/30 |
| Alerts caused by planted events but of another kind | 11 `post_impact_inactivity` after bursts planted at night, when the home really is quiet | 6 |
| Static vs adapting baseline (frames flagged per test day, target 0.1%) | needs 3 test days | static 0.10 → 0.19%, adapting 0.10 → 0.17% after its one update; no alerts either way |
| Calibration length | 6 h: 5.42 alerts/day; 12–72 h: 0–0.42/day | — |

**`contexts`: category stays** ([ADR 0002](decisions/0002-selective-mahalanobis-baseline.md), now accepted).

| Embedding context | Fall AUC vs SINS | Non-fall AUC vs SINS | Falls vs non-falls | Home's impact / water / household frames flagged | Behavioural alerts/day |
|---|---|---|---|---|---|
| category | 0.831 | 0.784 | 0.551 | 1.5% / 0% / 0.13% | 0 |
| global | 0.834 | 0.773 | 0.579 | 1.5% / 2.6% / 1.3% | 0 |
| global, 64 PCA dims | 0.909 | 0.882 | 0.554 | 6.0% / 5.5% / 3.3% | 0 |
| period | 0.859 | 0.808 | 0.565 | 3.0% / 4.8% / 4.0% | 0 |

SAFE's non-falls, recorded the same way, separate from SINS almost as well as its falls, and no
method tells falls from non-falls (AUC 0.52–0.58). The higher AUCs of PCA and period mode come
from the recording setup, not the fall. What does differ is how often the home's own sounds are
flagged, the question ADR 0002 was opened for, and there category mode is best. The behavioural
detectors gave the same alerts with every context, so the choice costs nothing there.

**Thresholds.** Selection rules, set before the sweep: (1) NF1, silence evidence within 10 s of
the start of every planted fall with the ideal classifier; (2) untouched-day alerts must not rise;
(3) planted detection must not drop; (4) otherwise keep the default, since the tuning days are also
the reported days. Each variant was recalibrated ([sweep](../ml/reports/anomaly-sins-node6-tuning/report.md),
main split, before the changes):

| Variant | Result | Decision |
|---|---|---|
| `post_impact.short_seconds` 5 / 4 / **3** | evidence max 11.8 / 10.3 / 9.4 s; alerts unchanged | **3** (rule 1) |
| `impacts.min_count` 4 / **6** | 0.42 → 0 alerts/day; bursts still all caught, ~20 s later. Second split, not used for tuning: 0.59 → 0 | **6** |
| `impacts.target_alerts_per_day` 0.01 | no change (the learned maximum sets the limit) | keep 0.05 |
| `embedding.frame_quantile` 0.995 / 0.9995 | no behavioural change (no `sustained_unusual_sound` either way) | keep 0.999 |
| `post_impact.strong_max_seconds` 60 / 300 | only moves the inactivity alert to ~65 / ~305 s | keep 120 until the real classifier's false triggers are known (task 9) |
| impact category `on` 0.1 / 0.3 | 6.25 alerts/day / 44% of bursts missed | keep 0.2 |
| `activity.margin_db` 3 / 10 | 1.67 alerts/day / no change | keep 6 |
| `timezone` UTC instead of Europe/Brussels | identical results | the unconfirmed SINS clock doesn't affect them |

**What the fall classifier needs to deliver (from the stand-in sweep, main split):** a notification
within about 5 s of the fall's start keeps every silence check inside NF1 (5 s late: 9.1 s
median, 9.4 s max). Every missed fall is one only the impact path can catch, and it catches 13–20%.
False triggers on the home's own door and impact sounds rarely became alerts (1 of 248 across
all runs), because someone is usually moving afterwards. That will be different in a home where
the person is alone and still, so it has to be measured again with the real classifier (task 9).

Caveats: one room, one home, 5.4 days. The two splits share 3–5 Feb, so the second split is only
partly independent. SAFE was recorded with other microphones in other rooms; the non-fall control
shows this matters, so frame-level AUCs are not fall-detection results. The SINS clock timezone is
unconfirmed, but UTC gave the same results.

### Results on synthetic frames (laptop; not accuracy claims)

Measured on an Apple-silicon laptop with Python 3.11, before the SINS tuning (5 s silence check,
impact minimum 4); the Pi 5 will be slower and must be re-measured (task 9). Simulated frames show that the code works as designed, not that it
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
parts of 30 Jan and 5 Feb. Commands: ml/README.md. `python -m src.anomaly.evaluate` reports:
1. Each method's separation of SAFE falls from SINS frames (AUC, pAUC at FPR ≤ 0.1), falls
   caught and normal frames flagged per hour at the calibrated threshold, the same for SAFE
   non-falls (`--non-falls`, a control for the recording setup), and how many of the home's own
   impact, water and household frames each method flags.
2. Per embedding context (`--contexts`): planted falls, running water and impact bursts in the test
   days, once per seed (`--plant-seeds`), with detection rate and latency. Falls are replayed with
   no classifier, an ideal stand-in (every fall, on time) and a weaker one (80%, 2 s late, 5 false
   triggers/day on the home's own impact and household sounds). Alerts are split into detections,
   side alerts (caused by a planted event, of another kind) and stray alerts (false).
3. False alerts per day on untouched test days (SINS is assumed normal).
4. Results against calibration length (6 h to 72 h), and static vs adapting baseline (needs
   ≥ 3 test days, so it runs on the second split).
5. `--parts classifier`: the stand-in's delay, recall and false triggers, one at a time.
6. `--parts tuning`: threshold variants (`TUNING_VARIANTS`), each recalibrated.

## 7. What I need from teammates

| From | What | Why |
|---|---|---|
| Fareed (preprocessing/YAMNet) | **Done:** YAMNet frames for SINS and SAFE from `ml/src/export_frames.py` (0.96 s windows, 0.48 s hop, scores and levels); SINS Node 6 and SAFE exported, see ml/data/README.md | — |
| Fareed / anyone | The SINS recording clock's timezone (assumed Europe/Brussels, not confirmed) | Time-of-day periods use local time; UTC gave identical SINS results, so this is low priority |
| Yannis (fall classifier) | A fall-candidate confidence per frame and the threshold at which it counts; notification within ~5 s of the fall's start; its recall and false triggers per day on SAFE/SINS | `notify_fall_candidate()` starts the post-fall silence watch; strong triggers use the 2 min limit. The numbers replace the stand-ins in task 9 |
| Matthew (edge integration) | A frame source in the main loop (thread layout), mute button events, one `mic_id` per microphone | `AnomalyService.process()` per frame; `pause()` on mute |
| Samy / whoever owns the Event Manager | Who owns the FR7 fusion (Table 1.4 assigns "event-handling" to Samy; the Gantt lists "Event Manager" without an owner). The risk formula and which findings alert | The module returns `level`, findings with `role` and `confidence`; it does not decide alerts |
| Samy (backend/dashboard) | Show `details.kind` and `details.started_at`; delayed alerts already carry `detected_at` | Findings use only existing schema fields (`anomaly`, `inactivity`) |
| Team | Agree whether `.github/workflows/ml.yml` may run the harness tests (added in this branch) | CI ownership is members 1 and 5 |

## 8. Limitations

* Real-audio numbers come from one SINS room (one person, one home, one week, 1.6.1) with SAFE
  falls planted into it, and 2.4–3.4 test days. Thresholds were tuned on the same days. They are a
  first check, not a measure of accuracy in other homes.
* Without the fall classifier, few falls start the post-impact watch, because YAMNet seldom calls a
  fall an impact (13–20% of planted SAFE falls).
* Fall results use a stand-in for the classifier, so they show what the module does with a
  classifier of that quality, not what the system does (task 9).
* Category detection inherits YAMNet's mistakes. The impact on/off levels were checked on SINS;
  the other categories' levels were not tuned.
* Visitors and changed routines show up as anomalies (1.6.3); adaptation is off by default.
* A fall with the TV on isn't followed by silence, so only the fall classifier and the embedding
  score can catch it.
