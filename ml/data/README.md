# data/ (contents git-ignored)

Put datasets here locally. Record where each one comes from so everyone can rebuild it.

| Folder | Source | Notes |
|--------|--------|-------|
| `safe/` | SAFE dataset, Garcia & Huang, Smart Health 2025 (doi:10.1016/j.smhl.2024.100539). Download: [kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset](https://www.kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset) | 950 WAVs (3 s, 48 kHz mono): `-01.wav` = fall (475), `-02.wav` = non-fall (475) |
| `staged/` | Our staged falls (dummy + mats) | Shared drive: TODO link |
| `sins/` | SINS database, Dekkers et al., DCASE 2017 (github.com/KULeuvenADVISE/SINS_database). Node 6 (living room, by the kitchen sink): [Zenodo record 2547313](https://zenodo.org/records/2547313), `Node6_audio_01`–`07.zip` (~35 GB; 08–10 hold only sync files) | One week, one home; baseline learning (proposal 1.6.1). 4-channel 16 kHz WAVs of ~61.5 s |

## Frame streams (what the anomaly evaluation reads)

The anomaly module works on YAMNet frames, not audio, so its evaluation needs datasets turned
into *frame stream* `.npz` files: `timestamps`, `embeddings`, optional `scores` +
`score_classes` and `level_dbfs`. The exact format is in
`edge/src/e3_edge/anomaly/frames.py`. Wanted for the evaluation:

| File(s) | Content |
|---------|---------|
| `processed/sins/<room>/YYYY-MM-DD.npz` | One room, one file per local day, real timestamps, in time order |
| `processed/safe/falls.npz` | Frames of the SAFE fall clips, clips separated by gaps of more than 5 s |
| `processed/safe/non_falls.npz` | Same for the non-fall clips (optional, for an "other impacts" check) |

`python -m src.export_frames` writes these (run from `ml/`, see its docstring). Two things about
SINS it handles: each channel sits at about half full scale (a DC offset that `prepare_audio`
removes), and each file repeats the last ~0.92 s of the one before (trimmed before YAMNet).

Exported so far (2026-11-07): SINS Node 6, `processed/sins/node6/2017-01-30.npz` …
`2017-02-05.npz` (6.4 GB; full days 31 Jan – 4 Feb; 30 Jan starts at 04:39, 5 Feb ends at 09:33;
clock assumed Europe/Brussels), and SAFE `falls.npz` / `non_falls.npz` (2,849 / 2,850 frames).
Node 6 doesn't fit unzipped on most laptops (~100 GB), so it was streamed one zip at a time
through the same `prepare_audio`, `trim_overlap` and `make_stream` functions.

Embeddings and scores must come from the same YAMNet model and preprocessing as the Pi.
