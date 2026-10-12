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
│   └── sender.py       HTTPS POST to the backend with retries
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
