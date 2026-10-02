import unittest
import numpy as np
from motionscope.detector import MotionDetector
from motionscope.optical_flow import FlowEstimator, summarize_flow


def blank():
    return np.full((120, 160, 3), 100, dtype=np.uint8)


class DetectorTests(unittest.TestCase):
    def test_static_scene_has_no_boxes(self):
        det = MotionDetector(history=50, min_area=100)
        for _ in range(20):
            _, boxes = det.apply(blank())
        self.assertEqual(boxes, [])

    def test_moving_square_is_detected(self):
        det = MotionDetector(history=50, min_area=100)
        for _ in range(20):
            det.apply(blank())
        frame = blank()
        frame[40:80, 60:100] = 250
        _, boxes = det.apply(frame)
        self.assertEqual(len(boxes), 1)

    def test_invalid_frame_raises(self):
        with self.assertRaises(ValueError):
            MotionDetector().apply(None)


class FlowTests(unittest.TestCase):
    def _frame(self, x):
        f = blank()
        f[40:80, x:x + 40] = 220
        f[50:60, x + 5:x + 15] = 30  # texture helps flow estimation
        return f

    def test_first_frame_returns_none(self):
        self.assertIsNone(FlowEstimator(1.0).update(self._frame(20)))

    def test_rightward_motion_is_east(self):
        est = FlowEstimator(1.0)
        est.update(self._frame(20))
        flow = est.update(self._frame(24))
        speed, direction = summarize_flow(flow)
        self.assertGreater(speed, 0.5)
        self.assertEqual(direction, "E")

    def test_no_motion_gives_none(self):
        zero = np.zeros((10, 10, 2), dtype=np.float32)
        self.assertEqual(summarize_flow(zero), (0.0, "none"))


if __name__ == "__main__":
    unittest.main()
