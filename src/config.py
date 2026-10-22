"""Configuration loading and saving.

Lives in its own module because two entry points need it: main.py reads the
config, calibrate.py reads it and writes it back.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

HsvRange = tuple[np.ndarray, np.ndarray]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config file into a plain dict."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write the config back to YAML.

    Comments are not preserved: the parameters are documented in README.md.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    return config_path


def parse_hsv_ranges(raw_ranges: list[dict[str, list[int]]]) -> list[HsvRange]:
    """Turn the config's list of {lower, upper} into numpy bounds for inRange.

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
