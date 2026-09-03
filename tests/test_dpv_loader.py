from pathlib import Path
import pytest
from src.ingest.dpv_loader import load_dpv

ROOT = Path(__file__).resolve().parents[1]
DPV = ROOT / "vocab/dpv.ttl"


@pytest.mark.skipif(not DPV.exists(), reason="DPV no descargado en vocab/ (S1-T05)")
def test_dpv_loads_in_memory():
    d = load_dpv(DPV)
    assert d.loaded
    assert d.n_concepts > 500        # DPV 2.3 tiene ~1123 conceptos
    assert d.n_triples > 5000


def test_missing_dpv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dpv(tmp_path / "no_existe.ttl")