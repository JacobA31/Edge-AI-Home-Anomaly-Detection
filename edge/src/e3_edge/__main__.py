"""Main loop: python -m e3_edge

TODO(member1): audio capture thread (sounddevice → ring buffer)
TODO(member3): fall model (YAMNet embeddings + classifier, LiteRT)
TODO(member4): baseline + anomaly detection
"""

import logging
import time

from . import config
from .events import make_event
from .outbox import Outbox
from .sender import flush

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    cfg = config.load()
    outbox = Outbox(cfg.queue_db)
    last_heartbeat = 0.0
    while True:
        now = time.monotonic()
        if now - last_heartbeat >= cfg.heartbeat_seconds:
            outbox.put(make_event(cfg.device_id, cfg.room, "heartbeat"))
            last_heartbeat = now
        flush(outbox, cfg)
        time.sleep(1)


if __name__ == "__main__":
    main()
