"""Live frame source with independent event delivery and bounded audio buffering.

Downstream code can supply on_frame=service.process and on_gap=service.pause.
This module does not choose the Event Manager's fall/anomaly fusion policy.
"""

import logging
import time
from collections.abc import Callable
from threading import Event

from .anomaly.frames import Frame
from .audio.capture import CaptureError, MicrophoneCapture
from .audio.pipeline import AudioPipeline
from .audio.windowing import WindowAssembler
from .config import Config
from .events import make_event
from .outbox import Outbox
from .sender import SenderWorker

log = logging.getLogger(__name__)
CAPTURE_STALL_SECONDS = 5.0
IDLE_WAIT_SECONDS = 0.005


def run(
    cfg: Config, *, stop_event: Event | None = None,
    on_frame: Callable[[Frame], object] | None = None,
    on_gap: Callable[[], object] | None = None,
) -> int:
    """Capture and process audio until stopped; return the processed frame count.

    Models and callbacks run sequentially on the calling thread. Callbacks must
    finish promptly and must not retain unbounded frame history. on_gap runs
    before the first frame of every continuous segment. No detector or alert
    decision is enabled by default; only metadata heartbeats are queued.

    Capture, model, callback, or sender failure propagates after cleanup so
    systemd can restart the application. GPIO mute/indicator control is separate
    work; stop_event stops the entire runtime, rather than muting it temporarily.
    """
    stop = stop_event if stop_event is not None else Event()
    if stop.is_set():
        return 0
    audio = AudioPipeline(cfg.yamnet.model_path)
    if stop.is_set():
        return 0
    windows = WindowAssembler()
    processed = 0
    try:
        with Outbox(cfg.queue_db) as outbox:
            sender = SenderWorker(cfg)
            sender.start()
            try:
                if stop.is_set():
                    return 0
                with MicrophoneCapture(
                    device=cfg.audio_device, block_size=cfg.audio_block_size,
                    buffer_blocks=cfg.audio_buffer_blocks,
                ) as microphone:
                    next_heartbeat = last_block = time.monotonic()
                    while not stop.is_set():
                        sender.check()
                        now = time.monotonic()
                        block = microphone.read()
                        if stop.is_set():
                            break
                        if block is not None:
                            last_block = now
                        elif now - last_block >= CAPTURE_STALL_SECONDS:
                            raise CaptureError("No microphone audio received for 5 seconds")
                        if now >= next_heartbeat:
                            outbox.put(make_event(cfg.device_id, cfg.room, "heartbeat", details={
                                "audio_active": True, "frames_processed": processed,
                                "dropped_samples": microphone.dropped_samples,
                                "input_overflows": microphone.input_overflows,
                            }))
                            next_heartbeat = now + cfg.heartbeat_seconds
                        if block is None:
                            stop.wait(IDLE_WAIT_SECONDS)
                            continue
                        for window in windows.push(block):
                            if stop.is_set():
                                break
                            if window.discontinuity and on_gap is not None:
                                on_gap()
                            if stop.is_set():
                                break
                            frame = audio.process_window(window.samples, start_time=window.start_time)
                            if stop.is_set():
                                break
                            if on_frame is not None:
                                on_frame(frame)
                            processed += 1
            finally:
                sender.stop()
    finally:
        windows.reset()
    log.info("Audio processing stopped after %d frames", processed)
    return processed
