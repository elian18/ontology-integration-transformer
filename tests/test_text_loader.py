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


def test_tentative_count_reads_every_heading_style(tmp_path):
    text = ("Article 1\nSubject-matter\nText.\nArticle 2\nMaterial scope\nText.\n"
            "Article 3 - Territorial scope\nText.\n")
    f = tmp_path / "gdpr.txt"
    f.write_bytes(text.encode("utf-8"))
    assert load_legal_text(f).candidate_articles == 3            # S7-T07: English headings
    windows = tmp_path / "gdpr-windows.txt"
    windows.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    report = load_legal_text(windows)
    assert "\r" not in report.text and report.candidate_articles == 3
    old_mac = tmp_path / "gdpr-mac.txt"
    old_mac.write_bytes(text.replace("\n", "\r").encode("latin-1"))
    assert load_legal_text(old_mac).text == text


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