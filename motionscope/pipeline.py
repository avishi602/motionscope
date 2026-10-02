"""Pipeline: ties detection, optical flow, tracking and analytics together."""
import time
from pathlib import Path
import cv2
import numpy as np

from .analytics import Analytics
from .config import Config
from .detector import MotionDetector
from .logger import get_logger
from .optical_flow import FlowEstimator, flow_to_color, summarize_flow
from .tracker import CentroidTracker
from .video_io import create_writer, open_video, video_info


def _color_for(tid: int) -> tuple:
    """Stable, visible colour per track ID (BGR)."""
    palette = [(0, 0, 255), (0, 170, 0), (255, 100, 0), (0, 200, 255),
               (200, 0, 200), (255, 255, 0)]
    return palette[tid % len(palette)]


def _annotate(frame, tracks, trails, frame_idx, speed, direction):
    """Draw boxes, IDs, trails and a status line on a copy of the frame."""
    out = frame.copy()
    for tid, (x, y, w, h) in tracks.items():
        col = _color_for(tid)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label_y = y - 6 if y >= 40 else y + h + 16  # keep label visible
        cv2.putText(out, f"ID {tid}", (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        pts = np.array(trails.get(tid, [])[-30:], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(out, [pts], False, col, 2)
    status = (f"frame {frame_idx}  objects {len(tracks)}  "
              f"speed {speed:.1f}  dir {direction}")
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return out


def run_pipeline(input_path: str, out_dir: str, config: Config = None,
                 log_file: str = None, verbose: bool = False) -> dict:
    """Process a video and write all outputs into `out_dir`.

    Returns a dict with output file paths, the summary and elapsed time.
    Raises VideoError / ConfigError for bad input (handled by the CLI).
    """
    config = (config or Config()).validate()
    log = get_logger(log_file=log_file, verbose=verbose)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_video(input_path)
    info = video_info(cap)
    w, h, fps = info["width"], info["height"], info["fps"]
    log.info("Input: %s (%dx%d, %.1f fps, %d frames)",
             input_path, w, h, fps, info["frames"])

    detector = MotionDetector(config.history, config.var_threshold,
                              config.min_area)
    flow_est = FlowEstimator(config.flow_scale)
    tracker = CentroidTracker(config.max_distance, config.max_disappeared,
                             config.min_hits)
    analytics = Analytics(w, h, fps)

    annotated_path = str(out_dir / "annotated.mp4")
    writer = create_writer(annotated_path, fps, (w, h))
    flow_writer = None
    flow_path = str(out_dir / "optical_flow.mp4")
    if config.save_flow_video:
        flow_writer = create_writer(flow_path, fps, (w, h))

    start, idx = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if config.max_frames and idx >= config.max_frames:
                break

            mask, boxes = detector.apply(frame)
            flow = flow_est.update(frame)
            if idx < config.warmup_frames:
                boxes = []  # background still being learned

            tracks = tracker.update(boxes)
            speed, direction = 0.0, "none"
            if flow is not None:
                small_mask = cv2.resize(mask, (flow.shape[1], flow.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                speed, direction = summarize_flow(flow, small_mask)
                if flow_writer is not None:
                    flow_writer.write(flow_to_color(flow, (w, h)))
            elif flow_writer is not None:
                flow_writer.write(np.zeros((h, w, 3), dtype=np.uint8))

            writer.write(_annotate(frame, tracks, tracker.trails, idx,
                                   speed, direction))
            analytics.add_frame(idx, mask if idx >= config.warmup_frames
                                else np.zeros_like(mask),
                                tracks, speed, direction)
            idx += 1
            if idx % 50 == 0:
                log.debug("processed %d frames", idx)
    finally:
        cap.release()
        writer.release()
        if flow_writer is not None:
            flow_writer.release()

    if idx == 0:
        raise RuntimeError("No frames could be read from the video.")

    result = analytics.save_all(str(out_dir))
    result["files"]["annotated_video"] = annotated_path
    if config.save_flow_video:
        result["files"]["flow_video"] = flow_path
    result["seconds"] = round(time.time() - start, 2)
    log.info("Done: %d frames in %.2fs, %d unique objects",
             idx, result["seconds"], result["summary"]["unique_objects"])
    return result
