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
│   ├── audio/          microphone capture, overlapping windows, and LiteRT YAMNet frames
│   ├── fall_classifier/ LiteRT fall scoring and model/settings validation
│   └── anomaly/        household baseline + acoustic/behavioural anomaly detection
│                       (docs/anomaly-detection.md; CLI: python -m e3_edge.anomaly)
├── tests/
├── deploy/
│   ├── e3-edge.service systemd unit (auto-start, auto-restart)
│   ├── download_models.py download and validate YAMNet and the fall model/settings
│   └── update.sh       pull latest main and restart
└── config.example.yaml copy to config.yaml on the Pi
```

## Develop on your laptop

```bash
cd edge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # includes LiteRT; add ,pi for microphone capture
pytest
```

### Audio pipeline

`AudioPipeline` removes DC offset, converts to mono, runs YAMNet, and returns a `Frame`
containing an embedding, 521 class scores, the window's start time, and its audio level.
Supply normalized floating-point audio at 16 kHz, shaped `(samples,)` or `(samples, channels)`.
Full windows contain 15,600 samples per channel and start every 0.48 seconds.

Integration example once the model is installed and acquisition supplies `samples` and
`window_start` (UTC Unix seconds):

```python
from e3_edge import config
from e3_edge.audio.pipeline import AudioPipeline

cfg = config.load()
audio = AudioPipeline(cfg.yamnet.model_path)  # create once and reuse sequentially
frame = audio.process_window(samples, start_time=window_start)
```

Use `pad_end=True` only for an incomplete final window. Keep microphone streams separate.

### Continuous capture and windowing

`MicrophoneCapture` provides 16 kHz mono float32 blocks with ADC-based UTC timestamps.
`WindowAssembler` consumes those blocks and emits complete 15,600-sample windows with a
7,680-sample hop. Each waveform spans 0.975 seconds; YAMNet's spectrogram patch spans 0.96 seconds.
Neither component stores audio on disk or transmits it. Wiring them into the application loop
and separating inference from network sending remain integration work.

For example, with the model installed (`cfg` and `audio` as above):

```python
import time

from e3_edge.audio.capture import MicrophoneCapture
from e3_edge.audio.windowing import WindowAssembler

windows = WindowAssembler()  # one per microphone
try:
    with MicrophoneCapture(device="ReSpeaker") as microphone:
        while True:
            block = microphone.read()
            if block is None:
                time.sleep(0.005)
                continue
            for window in windows.push(block):
                # Interrupt downstream temporal detection on window.discontinuity.
                frame = audio.process_window(window.samples, start_time=window.start_time)
finally:
    windows.reset()  # discard partial audio on shutdown or capture failure
```

The assembler retains one fixed 62,400-byte sample buffer; returned windows own their arrays.
Consume or release returned windows promptly instead of accumulating them. Capture's default
1,024-sample blocks yield at most one window per push, while larger blocks can yield several.
No partial window is emitted or padded. Call `reset()` when monitoring is muted or stopped.

Each continuous segment starts at its first block's UTC timestamp. Window timestamps advance
using integer sample offsets, so callback chunk sizes and consumer delays do not affect the hop.
Capture discontinuities, unexpected sample indices, and ADC/UTC drift over 10 ms discard old
overlap and start a fresh segment. The first complete window in that segment has
`discontinuity=True`; call `AnomalyService.pause()` before processing it once detection is wired in.
Within a segment timestamps increase by 0.48 seconds; clock corrections on restart may change
UTC ordering across segments. Absolute timestamp accuracy still requires validation on the Pi.

### Focused tests

Run from `edge/` with its environment activated. These checks require no downloads;
real-model checks are skipped unless a model path is supplied:

```bash
python -m pytest tests/test_audio*.py tests/test_config.py tests/test_model_download.py -q
```

After exporting YAMNet in `ml/`, include the real-model check:

```bash
E3_YAMNET_MODEL=../ml/models/yamnet.tflite python -m pytest \
    tests/test_audio*.py tests/test_config.py tests/test_model_download.py -q
```

Also set `E3_FALL_CLASSIFIER_MODEL=../ml/models/fall_classifier.tflite` to include
the real classifier/settings and YAMNet compatibility check. The JSON must be beside the model.

## First-time Pi setup

Publish the configured GitHub Releases before running `update.sh`:

- `yamnet-v1`: `yamnet.tflite`.
- `fall-classifier-v1`: both `fall_classifier.tflite` and its matching `fall_classifier.json`.

Release tags and local model paths come from `config.yaml`. Classifier settings are saved
beside the model; the threshold comes from JSON, not YAML. The downloader checks all three
assets and their compatibility before installation. Failures stop the update before service restart.

```bash
sudo apt install -y git gh python3-venv libportaudio2
sudo useradd -r -m e3 && sudo mkdir -p /opt/e3 && sudo chown e3 /opt/e3
sudo -u e3 git clone https://github.com/<org>/<repo>.git /opt/e3   # private repo: use a deploy key
cd /opt/e3/edge && cp config.example.yaml config.yaml && nano config.yaml
sudo cp deploy/e3-edge.service /etc/systemd/system/
gh auth login
/opt/e3/edge/deploy/update.sh
```

After that, `/opt/e3/edge/deploy/update.sh` installs whatever is on `main`.
Models and classifier settings come from GitHub Releases made by the ML team, not Git history.

## Anomaly module on the Pi

The household baseline is learned on the Pi itself (3 days + 1 day of audio by default) and
saved to `/var/lib/e3/baseline.npz`; it never leaves the device. To check the module's speed,
memory and model size on the Pi:

```bash
.venv/bin/python -m e3_edge.anomaly bench                 # per-frame latency, RSS, baseline size
.venv/bin/python -m e3_edge.anomaly bench --soak-days 7   # also a simulated week (NF4)
.venv/bin/python -m e3_edge.anomaly info /var/lib/e3/baseline.npz
```
