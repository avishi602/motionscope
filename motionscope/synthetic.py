"""Generate a synthetic test video so the project runs with no downloads.

Three coloured shapes move across a textured background after a short
empty-background period (used by the detector to learn the scene).
"""
import numpy as np
import cv2
from .video_io import create_writer


def _background(width: int, height: int) -> np.ndarray:
    """Static grey gradient with fixed texture."""
    rng = np.random.default_rng(7)
    gradient = np.tile(np.linspace(90, 150, width, dtype=np.uint8),
                       (height, 1))
    texture = rng.integers(0, 12, (height, width), dtype=np.uint8)
    gray = cv2.add(gradient, texture)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def generate_demo_video(path: str, frames: int = 150, width: int = 480,
                        height: int = 360, fps: float = 25.0) -> str:
    """Write the demo video to `path` and return the path."""
    writer = create_writer(path, fps, (width, height))
    bg = _background(width, height)
    rng = np.random.default_rng(1)
    empty = 15  # frames with no objects
    try:
        for i in range(frames):
            frame = bg.copy()
            t = i - empty
            if t >= 0:
                # 1: rectangle moving left -> right
                x = int(3.0 * t) - 60
                cv2.rectangle(frame, (x, 20), (x + 50, 50), (30, 30, 220), -1)
                # 2: circle moving diagonally up-right
                cx, cy = int(40 + 2.4 * t), int(200 - 0.8 * t)
                cv2.circle(frame, (cx, cy), 22, (40, 200, 40), -1)
                # 3: ellipse moving right -> left on a wavy path
                ex = int(width - 10 - 2.6 * t)
                ey = int(270 + 35 * np.sin(t / 12.0))
                cv2.ellipse(frame, (ex, ey), (34, 18), 0, 0, 360,
                            (220, 120, 20), -1)
            noise = rng.normal(0, 2, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255)
            writer.write(frame.astype(np.uint8))
    finally:
        writer.release()
    return path
