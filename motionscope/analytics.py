"""Module 4 - Analytics and reporting: heatmap, CSV, JSON summary, charts."""
import csv
import json
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt


class Analytics:
    """Collect per-frame statistics and write the final reports."""

    def __init__(self, width: int, height: int, fps: float):
        self.width, self.height, self.fps = width, height, fps
        self.heat = np.zeros((height, width), dtype=np.float32)
        self.rows: List[dict] = []          # one row per (frame, track)
        self.frame_stats: List[dict] = []   # one row per frame

    def add_frame(self, idx: int, mask: np.ndarray, tracks: Dict[int, tuple],
                  speed: float, direction: str) -> None:
        """Record statistics for one processed frame."""
        self.heat += (mask > 0).astype(np.float32)
        for tid, (x, y, w, h) in tracks.items():
            self.rows.append({"frame": idx, "track_id": tid,
                              "x": x, "y": y, "w": w, "h": h,
                              "cx": round(x + w / 2, 1),
                              "cy": round(y + h / 2, 1)})
        self.frame_stats.append({
            "frame": idx, "objects": len(tracks),
            "motion_pixels": int((mask > 0).sum()),
            "mean_speed": round(speed, 3), "direction": direction})

    # ---- derived numbers -------------------------------------------------
    def track_summary(self) -> List[dict]:
        """Per-track duration, path length and average speed (px/s)."""
        by_id: Dict[int, List[dict]] = {}
        for r in self.rows:
            by_id.setdefault(r["track_id"], []).append(r)
        out = []
        for tid, pts in sorted(by_id.items()):
            path = sum(np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
                       for a, b in zip(pts, pts[1:]))
            frames = pts[-1]["frame"] - pts[0]["frame"] + 1
            seconds = frames / self.fps
            out.append({"track_id": tid, "first_frame": pts[0]["frame"],
                        "last_frame": pts[-1]["frame"],
                        "path_length_px": round(float(path), 1),
                        "avg_speed_px_per_s": round(float(path) / seconds, 1)
                        if seconds > 0 else 0.0})
        return out

    # ---- writers -----------------------------------------------------------
    def save_all(self, out_dir: str) -> dict:
        """Write tracks.csv, summary.json, heatmap.png, timeline.png."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {
            "tracks_csv": str(out / "tracks.csv"),
            "summary_json": str(out / "summary.json"),
            "heatmap_png": str(out / "heatmap.png"),
            "timeline_png": str(out / "timeline.png"),
        }
        self._write_csv(files["tracks_csv"])
        self._write_heatmap(files["heatmap_png"])
        self._write_timeline(files["timeline_png"])
        summary = {
            "frames_processed": len(self.frame_stats),
            "unique_objects": len({r["track_id"] for r in self.rows}),
            "peak_simultaneous_objects": max(
                (f["objects"] for f in self.frame_stats), default=0),
            "tracks": self.track_summary(),
        }
        Path(files["summary_json"]).write_text(json.dumps(summary, indent=2))
        return {"files": files, "summary": summary}

    def _write_csv(self, path: str) -> None:
        fields = ["frame", "track_id", "x", "y", "w", "h", "cx", "cy"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_heatmap(self, path: str) -> None:
        peak = self.heat.max()
        norm = (self.heat / peak * 255).astype(np.uint8) if peak > 0 \
            else self.heat.astype(np.uint8)
        cv2.imwrite(path, cv2.applyColorMap(norm, cv2.COLORMAP_JET))

    def _write_timeline(self, path: str) -> None:
        frames = [f["frame"] for f in self.frame_stats]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        a1.plot(frames, [f["objects"] for f in self.frame_stats],
                color="black", linewidth=1.5)
        a1.set_ylabel("Tracked objects")
        a1.grid(True, alpha=0.4)
        a2.plot(frames, [f["mean_speed"] for f in self.frame_stats],
                color="black", linestyle="--", linewidth=1.5)
        a2.set_ylabel("Mean flow speed")
        a2.set_xlabel("Frame")
        a2.grid(True, alpha=0.4)
        fig.suptitle("Activity timeline")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
