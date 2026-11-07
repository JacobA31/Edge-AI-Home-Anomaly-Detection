# edge/

Code that runs on the Raspberry Pi 5: audio capture, fall model, baseline/anomaly detection,
and the offline event queue that sends events to the backend.

CI (`.github/workflows/edge.yml`) only tests this folder. The Pi is **not** deployed by GitHub;
it pulls `main` itself, so nothing on campus Wi-Fi needs to accept incoming connections.

```
edge/
├── src/e3_edge/
│   ├── __main__.py     main loop (python -m e3_edge)
│   ├── config.py       reads config.yaml
│   ├── events.py       builds events that match shared/event.schema.json
│   ├── outbox.py       SQLite queue for outages (events only, never audio)
│   ├── sender.py       HTTPS POST to the backend with retries
│   └── anomaly/        household baseline + acoustic/behavioural anomaly detection
│                       (docs/anomaly-detection.md; CLI: python -m e3_edge.anomaly)
├── tests/
├── deploy/
│   ├── e3-edge.service systemd unit (auto-start, auto-restart)
│   └── update.sh       pull latest main and restart
└── config.example.yaml copy to config.yaml on the Pi
```

## Develop on your laptop

```bash
cd edge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,pi to also get audio + LiteRT
pytest
```

### YAMNet runner tests

Run from `edge/` with its virtual environment activated. Unit tests skip the real-model check:

```bash
python -m pytest tests/test_audio_yamnet.py -q
```

After exporting YAMNet in `ml/`, include the real-model check:

```bash
E3_YAMNET_MODEL=../ml/models/yamnet.tflite python -m pytest tests/test_audio_yamnet.py -q
```

## First-time Pi setup

```bash
sudo apt install -y git python3-venv libportaudio2
sudo useradd -r -m e3 && sudo mkdir -p /opt/e3 && sudo chown e3 /opt/e3
sudo -u e3 git clone https://github.com/<org>/<repo>.git /opt/e3   # private repo: use a deploy key
cd /opt/e3/edge && cp config.example.yaml config.yaml && nano config.yaml
sudo cp deploy/e3-edge.service /etc/systemd/system/
/opt/e3/edge/deploy/update.sh
```

After that, `/opt/e3/edge/deploy/update.sh` installs whatever is on `main`.
Models (`.tflite`) come from the GitHub Release made by the ML team, not from Git history.

## Anomaly module on the Pi

The household baseline is learned on the Pi itself (3 days + 1 day of audio by default) and
saved to `/var/lib/e3/baseline.npz`; it never leaves the device. To check the module's speed,
memory and model size on the Pi:

```bash
.venv/bin/python -m e3_edge.anomaly bench                 # per-frame latency, RSS, baseline size
.venv/bin/python -m e3_edge.anomaly bench --soak-days 7   # also a simulated week (NF4)
.venv/bin/python -m e3_edge.anomaly info /var/lib/e3/baseline.npz
```
