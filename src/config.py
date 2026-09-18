"""Central access to config/config.yaml and the .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = ROOT / "config" / "config.yaml"

# Load .env once, as soon as this module is imported. Because src.config is
# imported early by the other modules, this runs before any os.getenv() call
# (e.g. EMBEDDING_MODEL in embeddings.py, LLM_MODEL in llm_client.py).
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Read and cache config/config.yaml as a plain dict."""
    with open(_CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable (from the process or the loaded .env)."""
    return os.getenv(name, default)