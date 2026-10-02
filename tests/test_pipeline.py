import json
import os
import tempfile
import unittest
from motionscope.config import Config
from motionscope.pipeline import run_pipeline
from motionscope.synthetic import generate_demo_video
from motionscope.video_io import VideoError, open_video


class PipelineTests(unittest.TestCase):
    def test_missing_video_raises(self):
        with self.assertRaises(VideoError):
            open_video("does_not_exist.mp4")

    def test_end_to_end_finds_three_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = generate_demo_video(os.path.join(tmp, "v.mp4"))
            out = os.path.join(tmp, "out")
            result = run_pipeline(video, out, Config(save_flow_video=False))
            self.assertEqual(result["summary"]["unique_objects"], 3)
            for name in ("tracks.csv", "summary.json", "heatmap.png",
                         "timeline.png", "annotated.mp4"):
                self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
            with open(os.path.join(out, "summary.json")) as fh:
                self.assertEqual(json.load(fh)["frames_processed"], 150)

    def test_max_frames_limits_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = generate_demo_video(os.path.join(tmp, "v.mp4"))
            result = run_pipeline(video, os.path.join(tmp, "o"),
                                  Config(max_frames=30, save_flow_video=False))
            self.assertEqual(result["summary"]["frames_processed"], 30)


if __name__ == "__main__":
    unittest.main()
