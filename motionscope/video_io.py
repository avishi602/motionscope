"""Video reading and writing helpers with clear error messages."""
from pathlib import Path
import cv2


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or written."""


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file for reading, or raise VideoError."""
    if not Path(path).is_file():
        raise VideoError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"OpenCV could not open the video: {path}")
    return cap


def video_info(cap: cv2.VideoCapture) -> dict:
    """Return width, height, fps and frame count of an open video."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps if fps and fps > 0 else 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def create_writer(path: str, fps: float, size: tuple) -> cv2.VideoWriter:
    """Create an MP4 writer (size = (width, height)); raise if it fails."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, size)
    if not writer.isOpened():
        raise VideoError(f"Could not create output video: {path}")
    return writer
