"""Shared fixtures for the test suite."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Isolate ChromaDB: the whole suite uses a throwaway directory, so the real
# data/chroma_db is never touched. This must run before vector_store is imported
# anywhere, which is why it lives at the top of conftest (imported first).
_CHROMA_TEST_DIR = tempfile.mkdtemp(prefix="chroma_test_")
os.environ["CHROMA_PATH"] = _CHROMA_TEST_DIR
atexit.register(lambda: shutil.rmtree(_CHROMA_TEST_DIR, ignore_errors=True))


@pytest.fixture(scope="session")
def dpv_report():
    """Load the DPV once per test session (it is large and slow to parse)."""
    from src.ingest.dpv_loader import load_dpv
    dpv_path = ROOT / "vocab/dpv.ttl"
    if not dpv_path.exists():
        pytest.skip("DPV no descargado en vocab/ (S1-T05)")
    return load_dpv(dpv_path)