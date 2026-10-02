"""Module 3 - Multi-object centroid tracker with persistent IDs."""
from typing import Dict, List, Tuple
import numpy as np

Box = Tuple[int, int, int, int]


class CentroidTracker:
    """Track objects across frames by matching nearest centroids.

    Each frame, detections are matched to existing tracks using the smallest
    Euclidean distances (greedy, one-to-one) within `max_distance`.
    Unmatched detections start new tracks; tracks unseen for more than
    `max_disappeared` frames are removed. A track is only *reported* once
    it has been matched in `min_hits` frames, which suppresses one-frame
    noise detections.
    """

    def __init__(self, max_distance: float = 80.0, max_disappeared: int = 10,
                 min_hits: int = 1):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.hits: Dict[int, int] = {}
        self.hits_at_death: Dict[int, int] = {}
        self._next_id = 0
        self.centroids: Dict[int, Tuple[float, float]] = {}
        self.boxes: Dict[int, Box] = {}
        self.missing: Dict[int, int] = {}
        self.trails: Dict[int, List[Tuple[int, int]]] = {}

    @staticmethod
    def _centroid(box: Box) -> Tuple[float, float]:
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def _register(self, box: Box) -> None:
        tid = self._next_id
        self._next_id += 1
        c = self._centroid(box)
        self.centroids[tid] = c
        self.boxes[tid] = box
        self.missing[tid] = 0
        self.hits[tid] = 1
        self.trails[tid] = [(int(c[0]), int(c[1]))]

    def _deregister(self, tid: int) -> None:
        self.hits_at_death[tid] = self.hits.get(tid, 0)
        for store in (self.centroids, self.boxes, self.missing, self.hits):
            store.pop(tid, None)
        if self.trails.get(tid) is not None and \
                self.hits_at_death.get(tid, 0) < self.min_hits:
            self.trails.pop(tid, None)  # never confirmed: discard

    def update(self, boxes: List[Box]) -> Dict[int, Box]:
        """Update tracks with this frame's boxes; return {id: box} of the
        tracks that were matched or created in this frame."""
        if not boxes:
            for tid in list(self.missing):
                self.missing[tid] += 1
                if self.missing[tid] > self.max_disappeared:
                    self._deregister(tid)
            return {}

        if not self.centroids:
            for box in boxes:
                self._register(box)
            return self._confirmed(set(self.boxes))

        ids = list(self.centroids.keys())
        old = np.array([self.centroids[i] for i in ids])
        new = np.array([self._centroid(b) for b in boxes])
        dist = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)

        used_rows, used_cols = set(), set()
        # visit pairs from smallest distance to largest
        for flat in np.argsort(dist, axis=None):
            r, c = divmod(int(flat), dist.shape[1])
            if r in used_rows or c in used_cols:
                continue
            if dist[r, c] > self.max_distance:
                break  # all remaining pairs are even farther
            tid = ids[r]
            cen = self._centroid(boxes[c])
            self.centroids[tid] = cen
            self.boxes[tid] = boxes[c]
            self.missing[tid] = 0
            self.hits[tid] += 1
            self.trails[tid].append((int(cen[0]), int(cen[1])))
            used_rows.add(r)
            used_cols.add(c)

        matched = {ids[r] for r in used_rows}
        for r, tid in enumerate(ids):
            if r not in used_rows:
                self.missing[tid] += 1
                if self.missing[tid] > self.max_disappeared:
                    self._deregister(tid)
        for c, box in enumerate(boxes):
            if c not in used_cols:
                self._register(box)
                matched.add(self._next_id - 1)

        return self._confirmed(matched)

    def _confirmed(self, ids) -> Dict[int, Box]:
        """Boxes of the given tracks that have enough hits to be reported."""
        return {t: self.boxes[t] for t in ids
                if t in self.boxes and self.hits.get(t, 0) >= self.min_hits}
