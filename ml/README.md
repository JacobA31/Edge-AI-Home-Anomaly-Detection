# ml/

Training and experiments. **Never deployed**; CI only lints it.

```
ml/
├── notebooks/       exploration (outputs stripped before commit, see below)
├── src/             reusable code: feature extraction, training, export to .tflite
├── data/            git-ignored; download datasets here (SAFE, our staged falls)
├── models/          git-ignored; trained models land here
└── requirements.txt
```

## Rules

- **No audio or datasets in Git.** `data/` and audio files are git-ignored. Keep them in the
  team's shared drive and document where each dataset comes from in `data/README.md`.
- **Strip notebook outputs** before committing (CI checks this):
  `pip install nbstripout && nbstripout --install` once in your clone does it automatically.
- **Features must match the Pi.** Anything the model needs at inference time (resampling,
  framing, normalization) goes in `src/` so `edge/` can reuse the same logic.
- **Shipping a model to the Pi:** export to `.tflite`, then attach it to a GitHub Release
  (`gh release create model-v1 models/fall_classifier.tflite --notes "SAFE + staged falls, F1=…"`).
  The Pi downloads the release asset. Log the run in MLflow so the report can cite the numbers.

## Setup

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Anomaly detection evaluation

`src/anomaly/` evaluates the edge anomaly module (it runs the edge code itself, installed by
`-e ../edge` in requirements.txt). Design and protocol: [docs/anomaly-detection.md](../docs/anomaly-detection.md).

```bash
# frame streams first (see data/README.md): SAFE in a minute, SINS one room per CSV
python -m src.export_frames safe data/safe
# SINS replay: calibrate on early days, test on later ones, SAFE falls as anomalies
S=data/processed/sins/node6
python -m src.anomaly.evaluate --calibration $S/2017-01-31.npz $S/2017-02-0{1,2}.npz \
    --test $S/2017-02-0{3,4,5}.npz --falls data/processed/safe/falls.npz \
    --timezone Europe/Brussels --out reports/anomaly-sins-node6
# smoke test of the harness on simulated frames (numbers mean nothing)
python -m src.anomaly.evaluate --synthetic --out reports/anomaly-synthetic
pytest -q                      # harness tests (CI runs them without TensorFlow)
```
