from __future__ import annotations

import shutil
import subprocess
import urllib.error
from pathlib import Path

import pytest
from rdflib import BNode, Graph, Literal
from rdflib.namespace import RDF, XSD

from src.publish import base_doc as bd
from src.publish import oops
from src.publish import widoco as wd
from tests.test_base_doc import LOPDP, _modules

JAR = wd.settings().jar
needs_widoco = pytest.mark.skipif(not JAR.exists() or not shutil.which("java"),
                                  reason="Falta WIDOCO en tools/widoco o Java")


def answer(*pitfalls) -> bytes:
    """What OOPS! answers: RDF/XML with one oops:pitfall per problem found."""
    g = Graph()
    for code, name, level, cases, elements in pitfalls:
        node = BNode()
        g.add((node, RDF.type, oops.OOPS.pitfall))
        g.add((node, oops.OOPS.hasCode, Literal(code)))
        g.add((node, oops.OOPS.hasName, Literal(name)))
        g.add((node, oops.OOPS.hasImportanceLevel, Literal(level)))
        if cases is not None:
            g.add((node, oops.OOPS.hasNumberAffectedElements, Literal(cases, datatype=XSD.int)))
        for e in elements:
            g.add((node, oops.OOPS.hasAffectedElement, Literal(e)))
    return g.serialize(format="xml").encode("utf-8")


ONTOPRIV = answer(("P08", "Missing annotations", "Minor", 3, ["a", "b", "c"]),
                  ("P19", "Defining multiple domains or ranges in properties", "Critical", 2,
                   ["p", "q"]),
                  ("P10", "Missing disjointness", "Important", None, []),
                  ("P41", "No license declared", "Important", None, []))
BASE = answer(("P08", "Missing annotations", "Minor", 5, ["a", "b", "c", "d", "e"]),
              ("P10", "Missing disjointness", "Important", None, []),
              ("P04", "Creating unconnected ontology elements", "Minor", 1, ["z"]))


class FakeOOPS:
    def __init__(self, fail=False):
        self.bodies, self.fail = [], fail

    def __call__(self, body, timeout):
        if self.fail:
            raise urllib.error.URLError("sin red")
        self.bodies.append(body)
        return ONTOPRIV if len(self.bodies) == 1 else BASE


def fake_widoco(cmd, cwd, **kw):
    """Writes what WIDOCO writes: with an IRI that ends in "/", in the folder doc."""
    page = Path(cmd[cmd.index("-outFolder") + 1]) / "doc"
    (page / "resources").mkdir(parents=True)
    ontology = Path(cmd[cmd.index("-ontFile") + 1]).read_text(encoding="utf-8")
    for name in ("index-es.html", "index-en.html"):
        (page / name).write_text(f"{name}\n{ontology}", encoding="utf-8")
    (page / "readme.md").write_text("de WIDOCO", encoding="utf-8")
    (page / "provenance").mkdir()
    (page / "provenance" / "provenance-es.html").write_text(r"C:\Users\x\Temp", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, "Documentation generated successfully", "")


@pytest.fixture
def site(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for name, g in zip(bd.MODULES, _modules()):
        g.serialize(output / f"{name}.ttl", format="turtle")
    ontopriv = tmp_path / "ontopriv.rdf"
    ontopriv.write_text("<rdf:RDF>OntoPriv original</rdf:RDF>", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CHANGELOG.md").write_text("historial", encoding="utf-8")
    return wd.Settings(jar=JAR if JAR.exists() else tmp_path / "widoco.jar", java="java",
                       docs=docs, ontopriv=ontopriv, output=output, oops_timeout=5,
                       widoco_timeout=600)


def test_the_request_is_the_one_widoco_sends():
    assert oops.request_body("<rdf:RDF/>") == (
        '<?xml version="1.0" encoding="UTF-8"?><OOPSRequest><OntologyUrl></OntologyUrl>'
        "<OntologyContent><![CDATA[ <rdf:RDF/> ]]></OntologyContent><Pitfalls></Pitfalls>"
        "<OutputFormat>RDF/XML</OutputFormat></OOPSRequest>")


def test_the_answer_is_read_most_important_first():
    found = oops.parse(ONTOPRIV)
    assert [(p.code, p.level, p.cases) for p in found] == [
        ("P19", "critical", 2), ("P10", "important", None), ("P41", "important", None),
        ("P08", "minor", 3)]
    assert found[0].elements == ("p", "q")


def test_the_comparison_says_what_changed_in_both_languages(tmp_path):
    result = oops.save(tmp_path, {"ontopriv": ONTOPRIV, "base": BASE}, "2026-11-23")
    assert oops.load(tmp_path) == result
    assert (tmp_path / "base-oops.rdf").read_bytes() == BASE          # what OOPS! answered
    es = oops.table(result, "es").split("\n")
    assert es[0] == "### Evaluación con OOPS!" and "el 2026-11-23" in es[2]
    assert es[4:6] == ["| Código | Problema | Importancia | OntoPriv | Base | Cambio |",
                       "|---|---|---|---|---|---|"]
    assert es[6:] == [
        "| P19 | Defining multiple domains or ranges in properties | crítico | 2 | 0 | corregido |",
        "| P10 | Missing disjointness | importante | toda la ontología | toda la ontología | "
        "igual |",
        "| P41 | No license declared | importante | toda la ontología | 0 | corregido |",
        "| P04 | Creating unconnected ontology elements | menor | 0 | 1 | nuevo |",
        "| P08 | Missing annotations | menor | 3 | 5 | más casos |"]
    assert "| P08 | Missing annotations | minor | 3 | 5 | more cases |" in oops.table(result, "en")
    assert oops.change(oops.Pitfall("P08", "", "minor", 5), oops.Pitfall("P08", "", "minor", 2)) \
        == "fewer"


def test_oops_sees_ontopriv_as_published_and_the_base_without_the_page_sentences(site):
    fake = FakeOOPS()
    wd.evaluate(site, _modules(), post=fake, today="2026-11-23")
    ontopriv, base = fake.bodies
    assert "<![CDATA[ <rdf:RDF>OntoPriv original</rdf:RDF> ]]>" in ontopriv
    assert "Elian Alexander Gavilanes Miranda" in base                  # with its header
    assert "Correspondencias con el DPV" not in base and "Data Owner" not in base
    assert oops.load(site.docs / "oops")["date"] == "2026-11-23"


def test_the_page_goes_to_docs_with_the_comparison(site):
    (site.docs / "provenance").mkdir()                       # from an earlier page
    lines = wd.build(site, ask_oops=True, post=FakeOOPS(), run=fake_widoco, today="2026-11-23")
    assert lines[1] == ("  OOPS!: evaluación del 2026-11-23 (OntoPriv: 4 problemas; base: "
                        "3 problemas)")
    docs = site.docs
    assert (docs / "index.html").read_text(encoding="utf-8") == (
        docs / "index-es.html").read_text(encoding="utf-8")
    assert "### Evaluación con OOPS!" in (docs / "index.html").read_text(encoding="utf-8")
    assert (docs / ".nojekyll").exists() and (docs / "resources").is_dir()
    assert not (docs / "readme.md").exists()                            # only the page
    assert not (docs / "provenance").exists()            # it showed a folder of this computer
    assert "-excludeProvenance" in wd.widoco_command(site, Path("a.ttl"), Path("out"))
    assert (docs / "CHANGELOG.md").read_text(encoding="utf-8") == "historial"


def test_without_internet_the_saved_evaluation_is_used(site):
    oops.save(site.docs / "oops", {"ontopriv": ONTOPRIV, "base": BASE}, "2026-11-01")
    lines = wd.build(site, ask_oops=True, post=FakeOOPS(fail=True), run=fake_widoco)
    assert lines[1].startswith("  OOPS! no respondió (") and "se usa la evaluación guardada" \
        in lines[1]
    assert lines[2].startswith("  OOPS!: evaluación del 2026-11-01")


def test_without_widoco_it_says_where_it_goes(site, tmp_path):
    site = wd.Settings(**{**site.__dict__, "jar": tmp_path / "no-esta.jar"})
    with pytest.raises(FileNotFoundError, match="falta WIDOCO en .*no-esta.jar"):
        wd.build(site, ask_oops=False, run=fake_widoco)


def test_a_failed_widoco_shows_its_last_lines(site):
    def broken(cmd, cwd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "linea 1\nError: Java 17 o mayor")
    with pytest.raises(RuntimeError, match="WIDOCO no generó la página:\nlinea 1\nError: Java"):
        wd.run_widoco(site, site.output / "x.ttl", site.output, run=broken)


@pytest.mark.slow
@needs_widoco
def test_the_real_widoco_links_the_table_to_each_entry(site):
    wd.build(site, ask_oops=True, post=FakeOOPS(), today="2026-11-23")
    page = (site.docs / "index.html").read_text(encoding="utf-8")
    owner = f"{LOPDP}Data_Owner"
    assert f'id="{owner}"' in page and f"(#{owner})" in page            # the link has its entry
    assert "Evaluación con OOPS!" in page and "Correspondencias de la base" in page
    assert (site.docs / "index-en.html").exists() and (site.docs / "ontology.ttl").exists()
    assert "Provenance" not in page and not (site.docs / "provenance").exists()
