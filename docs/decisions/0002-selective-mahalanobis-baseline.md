# 0002 – Category-conditioned "selective" Mahalanobis baseline

- **Status:** accepted (confirmed on SINS Node 6, 2026-11-07)
- **Date:** 2026-10-30

## Context
Proposal 1.7.3.2 scores each YAMNet embedding by its Mahalanobis distance from *one* Gaussian
(mean + covariance) of household embeddings. Section 1.7.4 also wants a familiar sound, like a
cane's thud, to get a low anomaly score.

Those two conflict. A single Gaussian is dominated by the commonest frames (quiet, TV). Any sound
that is rare in time is far from its mean, even if the home makes it every day. On simulated
days, 26 of 49 ordinary impact frames scored as anomalous. Each one then looked like an
"unfamiliar impact", and three ordinary thuds followed by quiet turned into false inactivity
alerts in a single day.

## Decision
Keep Mahalanobis scoring, but make it *selective* (as in the DCASE Task 2 baseline the proposal
cites). The baseline holds the global Gaussian plus one Gaussian per household sound category
(water, impact, footsteps, speech, household, media, alarm). Each category's Gaussian is fitted
on the calibration frames where YAMNet's own class scores say that category is present. A frame's
score is its smallest distance to any of them.

- Rare categories have few frames, so their Gaussians keep only `min(32, n/20)` principal
  directions plus one residual variance (probabilistic PCA). Below 50 frames, a category gets no
  Gaussian of its own.
- `contexts: global` restores the proposal's single Gaussian; `period` uses time-of-day periods
  instead of categories. The evaluation harness compares all three (plus k-means contexts,
  Isolation Forest, kNN and a Euclidean baseline).

## Consequences
- On the same simulated days, ordinary impacts flagged dropped from 26/49 to 1/49 and the
  false inactivity alerts disappeared.
- The price, also on simulated data: falls that look a bit like the home's thuds score closer to
  normal. At the calibrated threshold the embedding alone flagged 17.5% of simulated falls in
  category mode vs 69% with one global Gaussian (AUC 0.983 vs 0.999). With the fall classifier's
  notification, every planted fall still raised its silence evidence and inactivity alert, so in
  this design the classifier detects falls and the embedding score keeps routine sounds quiet.
- The model is still small (about 5 MB at 1024 dims, NF5) and scoring stays far below the 200 ms
  per-frame budget.
- Detection now uses YAMNet's class scores as well as its embeddings, so the YAMNet stage must
  pass both on. Without scores it falls back to the single global Gaussian.
- Whether this also holds on real audio is the first question for the SINS evaluation; if it
  doesn't, switch `contexts` back in config, with no code change.

## First SINS result (2026-11-07)
On SINS Node 6 (72 h calibration, 2.4 test days) with SAFE falls, category mode separated falls
from household frames no better than one global Gaussian (AUC 0.831 vs 0.834). A global Gaussian
on 64 PCA dimensions did best (0.909), and time-of-day periods came next (0.859). Category mode had
0.42 false alerts per day on the untouched test days, but the harness only runs the behavioural
detectors in category mode, so it hasn't been compared with global there yet. That comparison
(ordinary impacts flagged, false inactivity alerts) decides this ADR; the status stays proposed
until it is run. Report: `ml/reports/anomaly-sins-node6/report.md`.

## Decision on SINS (2026-11-07)
The harness now runs the behavioural detectors for every context and adds two checks: SAFE's
non-fall clips as a control, and how often each context flags the home's own sounds.

- With the same recordings, SAFE non-falls separate from SINS almost as well as falls (category
  0.784 vs 0.831; global PCA-64 0.882 vs 0.909), and no context separates falls from non-falls
  (AUC 0.55–0.58). The PCA and period modes' higher fall AUCs are the recording setup, not falls.
- Share of the home's own frames flagged (impact / water / household): category 1.5% / 0% / 0.13%,
  global 1.5% / 2.6% / 1.3%, global PCA-64 6.0% / 5.5% / 3.3%, period 3.0% / 4.8% / 4.0%.
- Behavioural alerts on the untouched test days and planted-event detection were identical for all
  four contexts.

So category mode is kept: it flags the home's familiar sounds least, which is what this ADR is for,
and it loses nothing measurable on falls. Report: `ml/reports/anomaly-sins-node6/report.md`;
details in docs/anomaly-detection.md, section 6.
