"""Module 2 - Dense optical flow (Farneback) and motion summaries."""
from typing import Optional, Tuple
import cv2
import numpy as np

_DIRECTIONS = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]


class FlowEstimator:
    """Estimate dense optical flow between consecutive frames.

    Uses the Farneback polynomial-expansion method. Frames are optionally
    down-scaled for speed. Flow vectors are (dx, dy) per pixel, in pixels
    of the *scaled* image.
    """

    def __init__(self, scale: float = 0.5):
        self.scale = scale
        self._prev_gray: Optional[np.ndarray] = None

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        return gray

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return flow field (h, w, 2) vs previous frame, None on frame 1."""
        gray = self._prepare(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, pyr_scale=0.5, levels=3,
            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        self._prev_gray = gray
        return flow


def flow_to_color(flow: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Colour-code flow: hue = direction, brightness = speed (BGR image).

    `size` is (width, height) of the output image.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 25, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)


def summarize_flow(flow: np.ndarray, mask: Optional[np.ndarray] = None,
                   min_mag: float = 0.5) -> Tuple[float, str]:
    """Return (mean speed, compass direction) over moving pixels.

    If `mask` (same size as flow) is given, only masked pixels count.
    Returns (0.0, "none") when nothing moves faster than `min_mag`.
    """
    mag = np.hypot(flow[..., 0], flow[..., 1])
    valid = mag > min_mag
    if mask is not None:
        valid &= mask > 0
    if not valid.any():
        return 0.0, "none"
    dx = float(flow[..., 0][valid].mean())
    dy = float(flow[..., 1][valid].mean())
    angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360  # 0 = East, 90 = South
    direction = _DIRECTIONS[int(((angle + 22.5) % 360) // 45)]
    return float(mag[valid].mean()), direction
