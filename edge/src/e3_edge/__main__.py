"""Run continuous microphone capture and local YAMNet inference: python -m e3_edge."""

import logging
import signal
from threading import Event

from . import config
from .runtime import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config.load()
    stop = Event()

    def request_stop(signum, frame):
        stop.set()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, request_stop)
        run(cfg, stop_event=stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
