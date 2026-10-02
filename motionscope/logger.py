"""Central logging setup (console + optional log file)."""
import logging
from pathlib import Path


def get_logger(name: str = "motionscope", log_file: str = None,
               verbose: bool = False) -> logging.Logger:
    """Return a configured logger. Safe to call many times."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if logger.handlers:  # already configured
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
