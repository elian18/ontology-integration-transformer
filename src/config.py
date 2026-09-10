from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Read and cache config/config.yaml as a plain dict."""
    with open(_CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)