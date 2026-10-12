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
