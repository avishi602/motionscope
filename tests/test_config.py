import unittest
from motionscope.config import Config, ConfigError


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertIsInstance(Config().validate(), Config)

    def test_bad_values_raise(self):
        bad = [dict(history=0), dict(var_threshold=-1), dict(min_area=0),
               dict(warmup_frames=-1), dict(max_distance=0),
               dict(max_disappeared=-1), dict(min_hits=0),
               dict(flow_scale=0.0), dict(flow_scale=2.0),
               dict(max_frames=-5)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ConfigError):
                    Config(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
