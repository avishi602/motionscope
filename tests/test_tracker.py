import unittest
from motionscope.tracker import CentroidTracker


class TrackerTests(unittest.TestCase):
    def test_new_objects_get_unique_ids(self):
        t = CentroidTracker()
        tracks = t.update([(0, 0, 10, 10), (200, 200, 10, 10)])
        self.assertEqual(set(tracks), {0, 1})

    def test_id_is_kept_while_object_moves(self):
        t = CentroidTracker(max_distance=50)
        t.update([(0, 0, 10, 10)])
        tracks = t.update([(8, 0, 10, 10)])
        self.assertEqual(list(tracks), [0])

    def test_far_jump_creates_new_id(self):
        t = CentroidTracker(max_distance=20)
        t.update([(0, 0, 10, 10)])
        tracks = t.update([(300, 300, 10, 10)])
        self.assertEqual(list(tracks), [1])

    def test_track_removed_after_disappearing(self):
        t = CentroidTracker(max_disappeared=2)
        t.update([(0, 0, 10, 10)])
        for _ in range(3):
            t.update([])
        self.assertNotIn(0, t.centroids)

    def test_track_survives_short_gap(self):
        t = CentroidTracker(max_disappeared=5)
        t.update([(0, 0, 10, 10)])
        t.update([])
        tracks = t.update([(4, 0, 10, 10)])
        self.assertEqual(list(tracks), [0])

    def test_min_hits_hides_unconfirmed_tracks(self):
        t = CentroidTracker(min_hits=3)
        self.assertEqual(t.update([(0, 0, 10, 10)]), {})
        self.assertEqual(t.update([(2, 0, 10, 10)]), {})
        self.assertEqual(list(t.update([(4, 0, 10, 10)])), [0])

    def test_two_objects_do_not_swap_ids(self):
        t = CentroidTracker(max_distance=60)
        t.update([(0, 0, 10, 10), (100, 0, 10, 10)])
        tracks = t.update([(5, 0, 10, 10), (105, 0, 10, 10)])
        self.assertEqual(tracks[0][0], 5)
        self.assertEqual(tracks[1][0], 105)


if __name__ == "__main__":
    unittest.main()
