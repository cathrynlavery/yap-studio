"""Share the exact person coverage of an already identified layer ink mask."""

from __future__ import annotations

import numpy as np

PERSON_ALPHA_THRESHOLD = 128  # Match the person mask used by compositing and final checks.


def covered_ink_pixels(ink: np.ndarray, alpha: np.ndarray, x: int, y: int) -> int:
    """Count only ink inside the frame where the person mask is solid enough."""
    height, width = ink.shape
    frame_height, frame_width = alpha.shape
    left, top = max(0, x), max(0, y)
    right, bottom = min(frame_width, x + width), min(frame_height, y + height)
    if right <= left or bottom <= top:
        return 0
    return int(np.count_nonzero(ink[top - y:bottom - y, left - x:right - x] & (
        alpha[top:bottom, left:right] >= PERSON_ALPHA_THRESHOLD
    )))
