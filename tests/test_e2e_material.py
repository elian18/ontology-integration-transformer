from __future__ import annotations

import subprocess
from pathlib import Path

from src.e2e import material as mt

ROOT = Path(__file__).resolve().parents[1]
PAGE = ("<html><head><meta charset='windows-1252'><title>Ley</title></head><body><nav>Menú"
        "</nav><p>ARTICULO 1° — (Objeto).</p><p>La presente ley&nbsp;protege.</p>"
        "<script>x()</script><div>ARTICULO 2° — (Definiciones).<br>Datos.</div>"
        "</body></html>").encode("cp1252")


def test_a_page_keeps_only_its_text():
    assert mt.page_text(PAGE) == ("ARTICULO 1° — (Objeto).\nLa presente ley protege.\n"
                                  "ARTICULO 2° — (Definiciones).\nDatos.\n")
    assert mt.page_text("<p>Artículo 1.</p>".encode("utf-8"), "utf-8") == "Artículo 1.\n"


def test_the_line_breaks_of_the_html_source_are_spaces():
    # The Mexican law (ordenjuridico.gob.mx) breaks its paragraphs in the source (S11-T11).
    page = ("<p class=x>Artículo\r\n1. La presente Ley es\nde orden público.</p>\n"
            "<p>Artículo\n2. Para los efectos<br>de esta Ley:</p>").encode("utf-8")
    assert mt.page_text(page, "utf-8") == ("Artículo 1. La presente Ley es de orden público.\n"
                                           "Artículo 2. Para los efectos\nde esta Ley:\n")


def test_the_material_is_downloaded_and_a_failure_is_said(tmp_path):
    asked = []

    def get(url):
        asked.append(url)
        if "ordenjuridico" in url:
            raise OSError("sin conexión")
        if "interior.go.ke" in url:
            return b"<!doctype html><title>Ministry</title>", None     # its home page
        if url.endswith(".pdf"):
            return b"%PDF-1.7 datos de " + url.encode(), None
        return (PAGE, None) if "argentina" in url or "te.gob" in url else (b"x " + url.encode(),
                                                                           None)
    lines = mt.download(tmp_path, get=get)
    assert len(asked) == sum(len(s.urls) for s in mt.SOURCES) - 1     # kenya: the 2nd answers
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(s.name for s in mt.SOURCES)
    for law in ("argentina-ley-25326.txt", "mexico-lfpdppp-2025.txt"):
        assert (tmp_path / law).read_text(encoding="utf-8").startswith("ARTICULO 1° — (Objeto).")
    assert (tmp_path / "kenya-dpa-2019.pdf").read_bytes().startswith(b"%PDF-1.7 datos de http")
    assert lines[4].endswith(" — LFPDPPP de México (2025), texto oficial (www.te.gob.mx)")
    assert lines[-1].endswith(" — Data Protection Act 2019 de Kenia (kenyalaw.org)")
    assert lines[0].startswith("  gconsent.ttl: 0 KB, sha256 ")


def test_a_web_page_is_not_taken_for_a_pdf(tmp_path):
    lines = mt.download(tmp_path, get=lambda url: (b"<!doctype html>", None),
                        only={"kenya-dpa-2019.pdf"})
    assert lines[0].count("respondió una página web, no un PDF") == 3
    assert not (tmp_path / "kenya-dpa-2019.pdf").exists()


def test_an_empty_answer_from_every_source_is_a_failure(tmp_path):
    lines = mt.download(tmp_path, get=lambda url: (b"", None), only={"kenya-dpa-2019.pdf"})
    assert lines == ["  kenya-dpa-2019.pdf: no se pudo descargar (interior.go.ke: respuesta "
                     "vacía; kenyalaw.org: respuesta vacía; www.odpc.go.ke: respuesta vacía)"]
    assert not (tmp_path / "kenya-dpa-2019.pdf").exists()


def test_only_some_files_can_be_downloaded(tmp_path):
    mt.download(tmp_path, get=lambda url: (b"x", None), only={"pizza.owl"})
    assert [p.name for p in tmp_path.iterdir()] == ["pizza.owl"]


def test_the_material_matches_the_cases_and_stays_out_of_git():
    from src.e2e import runner as rn
    cases = rn.load_cases(ROOT / "data/reference/e2e-s11.yaml", ROOT)
    wanted = {p.name for c in cases for p in c.files.values()
              if p.parent == ROOT / mt.FOLDER}
    assert wanted == {s.name for s in mt.SOURCES}
    ignored = subprocess.run(["git", "check-ignore", "-q", f"{mt.FOLDER}/pizza.owl"], cwd=ROOT)
    assert ignored.returncode == 0
