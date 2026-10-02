"""Configuration object with validation for the MotionScope pipeline."""
from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass
class Config:
    """All tunable parameters of the pipeline.

    Attributes:
        history: number of frames the background model remembers (MOG2).
        var_threshold: pixel variance threshold for foreground (MOG2).
        min_area: smallest contour area (pixels) accepted as an object.
        warmup_frames: initial frames used only to learn the background.
        max_distance: max centroid jump (pixels) to keep the same track ID.
        max_disappeared: frames a track may be missing before it is dropped.
        min_hits: matches needed before a track is reported (noise filter).
        flow_scale: resize factor applied before optical flow (speed-up).
        save_flow_video: also write a colour-coded optical-flow video.
        max_frames: stop after this many frames (0 = process all).
    """

    history: int = 100
    var_threshold: float = 25.0
    min_area: int = 500
    warmup_frames: int = 10
    max_distance: float = 80.0
    max_disappeared: int = 10
    min_hits: int = 3
    flow_scale: float = 0.5
    save_flow_video: bool = True
    max_frames: int = 0

    def validate(self) -> "Config":
        """Check every value; raise ConfigError on the first problem."""
        if self.history < 1:
            raise ConfigError("history must be >= 1")
        if self.var_threshold <= 0:
            raise ConfigError("var_threshold must be > 0")
        if self.min_area < 1:
            raise ConfigError("min_area must be >= 1")
        if self.warmup_frames < 0:
            raise ConfigError("warmup_frames must be >= 0")
        if self.max_distance <= 0:
            raise ConfigError("max_distance must be > 0")
        if self.max_disappeared < 0:
            raise ConfigError("max_disappeared must be >= 0")
        if self.min_hits < 1:
            raise ConfigError("min_hits must be >= 1")
        if not 0.1 <= self.flow_scale <= 1.0:
            raise ConfigError("flow_scale must be between 0.1 and 1.0")
        if self.max_frames < 0:
            raise ConfigError("max_frames must be >= 0")
        return self
