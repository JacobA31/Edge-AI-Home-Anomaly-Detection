# data/ (contents git-ignored)

Put datasets here locally. Record where each one comes from so everyone can rebuild it.

| Folder | Source | Notes |
|--------|--------|-------|
| `safe/` | SAFE dataset, Garcia & Huang, Smart Health 2025 (doi:10.1016/j.smhl.2024.100539). Download: [kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset](https://www.kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset) | 950 WAVs (3 s, 48 kHz mono): `-01.wav` = fall (475), `-02.wav` = non-fall (475) |
| `staged/` | Our staged falls (dummy + mats) | Shared drive: TODO link |
| `sins/` | SINS database, Dekkers et al., DCASE 2017 (github.com/KULeuvenADVISE/SINS_database) | One week, one home; baseline learning (proposal 1.6.1) |

## Frame streams (what the anomaly evaluation reads)

The anomaly module works on YAMNet frames, not audio, so its evaluation needs datasets turned
into *frame stream* `.npz` files: `timestamps`, `embeddings`, optional `scores` +
`score_classes` and `level_dbfs`. The exact format is in
`edge/src/e3_edge/anomaly/frames.py`. Wanted for the evaluation:

| File(s) | Content |
|---------|---------|
| `sins/<room>/day1.npz` ... `day7.npz` | One room, one file per day, real timestamps, in time order |
| `safe/falls.npz` | Frames of the SAFE fall clips, clips separated by gaps of more than 5 s |
| `safe/non_falls.npz` | Same for the non-fall clips (optional, for an "other impacts" check) |

Embeddings and scores must come from the same YAMNet model and preprocessing as the Pi.
