"""S7-T07: cleaning the text of an uploaded law (page furniture, EUR-Lex marks, ligatures)."""
from pathlib import Path

import pytest

from src.ingest import law_text as lt
from src.ingest.law_text import clean_law_file, clean_pages, furniture, load_law_pages

ROOT = Path(__file__).resolve().parents[1]
LAWS = {"lopdp.pdf": (ROOT / "data/input/lopdp.pdf", 77),
        "gdpr-en.pdf": (ROOT / "data/input/laws/gdpr-en.pdf", 99),
        "ley-1581-2012-colombia.pdf": (ROOT / "data/input/laws/ley-1581-2012-colombia.pdf", 30)}


def _page(n, body, header="Ley 1581 de 2012 {n} EVA - Gestor Normativo"):
    return f"Departamento Administrativo\n{header.format(n=n)}\n{body}\nPágina {n} de 4"


def test_repeated_page_edges_are_furniture():
    pages = [_page(n, f"Artículo {n}. Texto {n}.") for n in range(1, 5)]
    found = furniture(pages)
    assert found == {"Departamento Administrativo", "Ley # de # # EVA - Gestor Normativo",
                     "Página # de #"}
    text, report = clean_pages(pages)
    assert "Gestor" not in text and "Página" not in text and "Departamento" not in text
    assert all(f"Artículo {n}. Texto {n}." in text for n in range(1, 5))
    assert report["removed_lines"] == 12 and report["pages"] == 4


def test_an_article_heading_is_never_furniture():
    pages = [f"Article {n}\nTitle {n}\nBody of article {n}.\nArticle {n + 10}" for n in range(4)]
    assert not any("Article" in f for f in furniture(pages))
    assert clean_pages(pages)[0].count("Article") == 8


def test_few_pages_or_lines_in_the_middle_are_kept():
    assert furniture(["a\nb\nc", "a\nb\nc"]) == set()                 # fewer than 3 pages
    pages = [f"Encabezado\nlínea {n}\nEncabezado\nmás {n}\notra {n}\nfin {n}" for n in range(4)]
    text, _ = clean_pages(pages)
    assert text.count("Encabezado") == 4                 # only the copies at the top go


def test_eurlex_marks_spaced_headings_and_ligatures():
    page = ("▼B\nArticle 1\nThe ►C1 controller ◄ shall act.\nS e c t i o n 1\n"
            "T r a n s p a r e n c y a n d m o d a l i t i e s\nDeﬁniciones y ﬂujo")
    text, report = clean_pages([page])
    assert text == "Article 1\nThe controller shall act.\nDefiniciones y flujo"
    assert (report["markers"], report["spaced_headings"], report["ligatures"]) == (2, 2, 2)


def test_a_txt_law_is_one_page(tmp_path):
    f = tmp_path / "ley.txt"
    f.write_bytes("Art. 1.- Objeto.\r\nArt. 2.- Ámbito.\r\n".encode("utf-8"))   # Windows
    assert load_law_pages(f) == ["Art. 1.- Objeto.\nArt. 2.- Ámbito.\n"]
    text, report = clean_law_file(f)
    assert text.startswith("Art. 1.- Objeto.") and report["furniture"] == []


def test_settings():
    assert (lt.EDGE_LINES, lt.MIN_PAGES, lt.MIN_SHARE) == (2, 3, 0.5)


@pytest.mark.parametrize("name", list(LAWS))
def test_real_laws_lose_their_furniture_but_no_article(name):
    from src.ingest.legal_segmenter import segment_articles
    path, expected = LAWS[name]
    if not path.exists():
        pytest.skip(f"Falta {path}")
    text, report = clean_law_file(path)
    assert report["furniture"] and report["removed_lines"] >= report["pages"]
    summary = segment_articles(text)
    assert summary["n_articles"] == expected and summary["missing"] == []
    for leftover in ("Gestor Normativo", "FINDER LOYAL", " — EN — ", "▼", "ﬁ"):
        assert leftover not in text