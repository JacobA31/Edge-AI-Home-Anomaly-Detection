"""Shared feature extraction. Keep in sync with what edge/ computes at inference time.

TODO(member2): resample to 16 kHz mono, split into 0.96 s frames for YAMNet.
"""

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.96
