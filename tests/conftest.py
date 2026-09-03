"""Shared fixtures for the test suite."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dpv_report():
    """Load the DPV once per test session (it is large and slow to parse)."""
    from src.ingest.dpv_loader import load_dpv
    dpv_path = ROOT / "vocab/dpv.ttl"
    if not dpv_path.exists():
        pytest.skip("DPV no descargado en vocab/ (S1-T05)")
    return load_dpv(dpv_path)