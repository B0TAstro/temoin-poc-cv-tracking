"""Configuration loading and saving.

Its own module because several entry points need it: every app reads a config,
and calibrate_hsv writes one back.

The path is always explicit. A shared core must not default to one POC's config,
which is why there is no implicit fallback here; each app names the file it
wants through config_path().
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"

HsvRange = tuple[np.ndarray, np.ndarray]


def config_path(name: str) -> Path:
    """Resolve a config file name, such as "ball.yaml", inside configs/."""
    return CONFIG_DIR / name


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Write a config back to YAML.

    Comments are not preserved: the parameters are documented in README.md.
    """
    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    return config_path


def parse_hsv_ranges(raw_ranges: list[dict[str, list[int]]]) -> list[HsvRange]:
    """Turn a config's list of {lower, upper} into numpy bounds for inRange.

    A list of ranges (rather than a single one) lets a hue that wraps around
    H=0, such as red, be expressed as two bands without changing the code.
    """
    return [
        (
            np.array(entry["lower"], dtype=np.uint8),
            np.array(entry["upper"], dtype=np.uint8),
        )
        for entry in raw_ranges
    ]
