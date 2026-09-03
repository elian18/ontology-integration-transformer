from pathlib import Path
import pytest
from src.ingest.text_loader import load_legal_text

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data/input/lopdp.pdf"


def test_txt_loads(tmp_path):
    f = tmp_path / "mini.txt"
    f.write_text("Art. 1.-Objeto.\nArt. 2.-Ámbito.\n", encoding="utf-8")
    r = load_legal_text(f)
    assert r.loaded
    assert r.source == "txt"
    assert r.candidate_articles == 2


def test_unsupported_source_is_rejected(tmp_path):
    f = tmp_path / "law.docx"
    f.write_bytes(b"fake")
    with pytest.raises(ValueError, match="no soportada"):
        load_legal_text(f)


@pytest.mark.skipif(not PDF.exists(), reason="LOPDP PDF no está en data/input/")
def test_pdf_extracts_text():
    r = load_legal_text(PDF)
    assert r.source == "pdf"
    assert r.n_chars > 100_000          # PDF digital (no escaneado)
    assert r.candidate_articles == 77   # artículos propios de la LOPDP