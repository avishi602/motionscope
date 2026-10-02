"""Module 1 - Motion detection using MOG2 background subtraction."""
from typing import List, Tuple
import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


class MotionDetector:
    """Detect moving objects by modelling the background of a static camera.

    Pipeline: MOG2 foreground mask -> remove shadows (threshold) ->
    morphological opening (remove noise) -> dilation (fill holes) ->
    contour extraction -> area filter -> bounding boxes.
    """

    def __init__(self, history: int = 100, var_threshold: float = 25.0,
                 min_area: int = 300):
        self.min_area = min_area
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (7, 7))

    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Box]]:
        """Return (clean foreground mask, list of bounding boxes)."""
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a BGR image (H x W x 3)")
        raw = self._subtractor.apply(frame)
        # MOG2 marks shadows as 127; keep only certain foreground (255).
        _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        mask = cv2.dilate(mask, self._dilate_kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours
                 if cv2.contourArea(c) >= self.min_area]
        return mask, boxes
