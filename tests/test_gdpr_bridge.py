"""Tests for the bridge of the base with the GDPR (S9-T09).

Toy graphs check how the pairs are chained and written; the real base and the alignment of
GDPRtEXT fixed in data/reference check the 34 pairs, that rebuilding gives the same ones, that
Forma and Lógica find nothing new and that a fake AI explains each pair once (cache)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.alignment import gdpr_bridge as gb
from src.validation import ai_auditor as au
from src.validation import shape_inspector as si
from tests.test_questions import HAS_JENA

ROOT = Path(__file__).resolve().parents[1]
BASE_FILES = [ROOT / "data" / "output" / name for name in
              ("ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf")]
needs_base = pytest.mark.skipif(
    not all(p.exists() for p in [*BASE_FILES, *gb.paths().values()]),
    reason="Falta la base en data/output o GDPRtEXT")
ENGINE = "jena" if HAS_JENA else "owlrl"
DPV = Namespace(gb.DPV)
ONTO = Namespace("http://www.semanticweb.org/ley-organica-proteccion-datos-personales#")
GDPR = Namespace(gb.GDPRTEXT_IRI + "#")


class ReviewLLM:
    """Says every pair is coherent, or what ``verdicts`` says by the name of the OntoPriv side."""

    def __init__(self, verdicts=None, model="modelo-de-prueba"):
        self.verdicts, self.model, self.calls, self.contexts = verdicts or {}, model, 0, []
        self.prompts = []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        self.prompts.append(prompt)
        out = []
        for block in context.split("=== ")[1:]:
            cid, text = block.split(" ===\n", 1)
            name = text.split("\n", 1)[0].split(": ", 1)[1].split(" (")[0]
            out.append({"concept": cid, "verdict": self.verdicts.get(name, "coherente"),
                        "relation": "closeMatch", "explanation": f"Explicación de {name}."})
        return json.dumps({"reviews": out})


def _graph(*triples) -> Graph:
    g = Graph()
    for t in triples:
        g.add(t)
    return g


# ------------------------------------------------------------------ the pairs

def test_the_pairs_reach_the_same_dpv_term_on_both_sides():
    ours = _graph((ONTO.Consent, SKOS.exactMatch, DPV.Consent),
                  (ONTO.Rights, SKOS.closeMatch, DPV.DataSubjectRight),
                  (ONTO.Data, SKOS.broadMatch, DPV.Data),                 # broad: out
                  (ONTO.Seals, SKOS.closeMatch, DPV.CertificationSeal),
                  (ONTO.Seals, SKOS.exactMatch, DPV.Seal),
                  (ONTO.Other, SKOS.exactMatch, URIRef("http://example.org/x")),
                  (BNode(), SKOS.exactMatch, DPV.Consent))
    theirs = _graph((GDPR.GivenConsent, SKOS.exactMatch, DPV.Consent),
                    (GDPR.Consent, SKOS.exactMatch, DPV.Consent),
                    (GDPR.Rights, SKOS.relatedMatch, DPV.DataSubjectRight),  # related: out
                    (GDPR.Data, SKOS.exactMatch, DPV.Data),
                    (GDPR.Seals, SKOS.closeMatch, DPV.CertificationSeal),
                    (GDPR.Seals, SKOS.exactMatch, DPV.Seal),
                    (GDPR.Other, SKOS.exactMatch, URIRef("http://example.org/x")))  # not the DPV
    pairs = gb.chain(ours, theirs)
    assert [(gb.local(p.ontopriv), gb.local(p.gdprtext)) for p in pairs] == [
        ("Consent", "Consent"), ("Consent", "GivenConsent"), ("Seals", "Seals")]
    assert pairs[2].terms == ((str(DPV.CertificationSeal), "closeMatch", "closeMatch"),
                              (str(DPV.Seal), "exactMatch", "exactMatch"))
    loose = gb.chain(ours, theirs, gb.LOOSE)
    assert len(loose) == 5                          # Data and Rights come in with any relation
    assert pairs[0].key == f"{ONTO.Consent} {GDPR.Consent}"


def test_the_module_says_the_term_each_pair_goes_through():
    pair = gb.Pair(str(ONTO.Seals), str(GDPR.Seals),
                   ((str(DPV.CertificationSeal), "closeMatch", "closeMatch"),
                    (str(DPV.Seal), "exactMatch", "exactMatch")))
    g = gb.build_graph([pair])
    onto = URIRef(gb.BRIDGE_IRI)
    assert (onto, RDF.type, OWL.Ontology) in g
    assert (onto, OWL.imports, URIRef("http://www.semanticweb.org/profiles/"
                                      "ecuador-lopdp-dpv-alignment")) in g
    assert set(g.objects(onto, DCTERMS.references)) == {URIRef(gb.GDPRTEXT_IRI),
                                                         URIRef("https://w3id.org/dpv")}
    assert (ONTO.Seals, SKOS.closeMatch, GDPR.Seals) in g
    assert len(list(g.subject_objects(SKOS.closeMatch))) == 1
    (axiom,) = g.subjects(RDF.type, OWL.Axiom)
    assert axiom == gb._axiom(pair) and str(axiom).startswith("pair")
    assert set(g.objects(axiom, gb.BRIDGE.chainedBy)) == {DPV.CertificationSeal, DPV.Seal}
    assert (g.value(axiom, OWL.annotatedSource), g.value(axiom, OWL.annotatedProperty),
            g.value(axiom, OWL.annotatedTarget)) == (ONTO.Seals, SKOS.closeMatch, GDPR.Seals)
    assert str(g.value(axiom, RDFS.comment)) == (
        "Encadenado por el DPV: OntoPriv Seals closeMatch dpv:CertificationSeal y GDPRtEXT "
        "Seals closeMatch dpv:CertificationSeal; OntoPriv Seals exactMatch dpv:Seal y GDPRtEXT "
        "Seals exactMatch dpv:Seal.")
    assert g.value(axiom, OWL.versionInfo) == Literal("1")
    assert (gb.BRIDGE.chainedBy, RDF.type, OWL.AnnotationProperty) in g
    assert all(o.language == "es" for o in g.objects(None, RDFS.comment))


def test_the_four_formats_hold_the_same_triples(tmp_path):
    from src.transform.convert import load_graph
    pair = gb.Pair(str(ONTO.Consent), str(GDPR.GivenConsent),
                   ((str(DPV.Consent), "exactMatch", "exactMatch"),))
    g = gb.build_graph([pair])
    files = gb.write_module(g, tmp_path)
    assert [f.name for f in files] == [f"{gb.MODULE}{e}" for e in (".rdf", ".owl", ".ttl",
                                                                    ".jsonld")]
    for f in files:
        assert isomorphic(load_graph(f), g), f.name


def test_what_each_side_says():
    g = _graph((ONTO.Consent, RDF.type, OWL.Class),
               (ONTO.Consent, RDFS.label, Literal("Consent", lang="en")),
               (ONTO.Consent, RDFS.label, Literal("Consentimiento", lang="es")),
               (ONTO.Consent, RDFS.comment, Literal("comentario")),
               (ONTO.Consent, SKOS.definition, Literal("Manifestación de voluntad.", lang="es")),
               (ONTO.Consent, DCTERMS.source, Literal("LOPDP, art. 8", lang="es")),
               (ONTO.hasPart, RDF.type, OWL.ObjectProperty),
               (GDPR.GivenConsent, RDFS.label, Literal("Given Consent")),
               (GDPR.GivenConsent, RDFS.comment, Literal("Consent given by the data subject.")),
               (GDPR.GivenConsent, RDFS.isDefinedBy, URIRef(
                   "http://purl.org/adaptcentre/openscience/resources/GDPRtEXT#recital32")),
               (GDPR.GivenConsent, RDFS.isDefinedBy, URIRef(
                   "http://purl.org/adaptcentre/openscience/resources/GDPRtEXT#article6-1-a")))
    ours = gb.describe(g, str(ONTO.Consent))
    assert (ours.kind, ours.label, ours.definition, ours.sources) == (
        "clase", "Consentimiento", "Manifestación de voluntad.", ["LOPDP, art. 8"])
    assert gb.describe(g, str(ONTO.hasPart)).kind == "propiedad"
    assert gb.describe(g, str(ONTO.Nothing)).label is None
    theirs = gb.describe(g, str(GDPR.GivenConsent), gdpr=True)
    assert (theirs.label, theirs.definition, theirs.sources) == (
        "Given Consent", "Consent given by the data subject.", ["article6-1-a", "recital32"])
    item = gb.BridgeItem("k", "Consent ↔ GivenConsent", ours, theirs,
                         [("Consent", "The agreement of the data subject.", "exactMatch",
                           "exactMatch")])
    assert gb.render_item(item, 10).splitlines() == [
        "OntoPriv: Consent (clase)", "Etiqueta: Consentimiento", "Definicion: Manifestac",
        "Articulo de la LOPDP: LOPDP, art. 8", "GDPRtEXT: GivenConsent (clase)",
        "Etiqueta: Given Consent", "Descripcion: Consent gi",
        "Definido en el GDPR: article6-1-a, recital32",
        "Termino DPV: Consent (OntoPriv exactMatch, GDPRtEXT exactMatch)",
        "Definicion DPV: The agreem"]


def test_the_prompt_reviews_pairs():
    text = gb.PROMPT.text
    assert gb.PROMPT.name == "bridge" and gb.PROMPT.version == 1
    assert '"reviews"' in text and "coherente" in text and "GDPRtEXT" in text
    assert "DEL concepto de OntoPriv HACIA el de GDPRtEXT" in text


def test_the_reviewing_ai_takes_another_prompt_and_text(tmp_path):
    item = au.AuditItem("k1", "Consent", ("class",), None, None, "skos:exactMatch",
                        str(DPV.Consent), "Consent", None)
    llm = ReviewLLM()
    cache = tmp_path / "c.jsonl"
    report = au.ask_audit([item], llm=llm, cache_path=cache, prompt=gb.PROMPT,
                          render=lambda i, chars: f"OntoPriv: {i.name} (clase)")
    assert report.answers["k1"]["verdict"] == "coherente" and llm.calls == 1
    assert llm.contexts[0] == "=== C1 ===\nOntoPriv: Consent (clase)"
    saved = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    assert saved["prompt"] == gb.PROMPT.manifest()
    again = au.ask_audit([item], llm=llm, cache_path=cache,         # audit.v1: another key
                         render=lambda i, chars: f"OntoPriv: {i.name} (clase)")
    assert llm.calls == 2 and again.cached == 0
    assert llm.prompts == [gb.PROMPT.text, au.PROMPT.text]


# ------------------------------------------------------------------ the real bridge

@needs_base
def test_the_bridge_of_the_base(tmp_path):
    llm = ReviewLLM({"ConfidentialityAgreement": "dudosa"})
    out = gb.build(llm=llm, use_llm=True, engine=ENGINE, out_dir=tmp_path)
    assert (out["pairs"], out["terms"], out["loose_pairs"]) == (34, 22, 256)
    assert llm.calls == 4                                               # 10 per call
    assert set(llm.prompts) == {gb.PROMPT.text}
    assert out["validation"]["shapes"]["new_violations"] == 0
    assert out["validation"]["shapes"]["new_warnings"] == 0
    assert out["validation"]["logic"]["new_errors"] == 0
    assert out["validation"]["logic"]["new_warnings"] == 0
    assert gb.clean(out["validation"])
    assert out["summary"] == {"coherente": 32, "dudosa": 2, "incoherente": 0, "inválida": 0,
                              "pending": 0}
    rows = {(r["ontopriv"], r["gdprtext"]): r for r in out["rows"]}
    given = rows[("Consent", "GivenConsent")]
    assert (given["dpv_terms"], given["ontopriv_relation"], given["gdprtext_relation"]) == (
        "dpv:Consent", "exactMatch", "exactMatch")
    assert "article6-1-a" in given["gdpr_articles"]
    assert given["explanation"] == "Explicación de Consent."
    with open(tmp_path / gb.CSV_FILE, encoding="utf-8-sig", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 34 and list(table[0]) == gb.CSV_COLUMNS
    assert all(r["explanation"] for r in table)
    module = Graph().parse(tmp_path / f"{gb.MODULE}.ttl")
    assert len(list(module.subject_objects(SKOS.closeMatch))) == 34
    assert len(list(module.subjects(RDF.type, OWL.Axiom))) == 34
    assert "Explicación" not in (tmp_path / f"{gb.MODULE}.ttl").read_text(encoding="utf-8")
    saved = json.loads((tmp_path / gb.RESULT_FILE).read_text(encoding="utf-8"))
    assert saved["pairs"] == 34 and saved["prompt"] == gb.PROMPT.manifest()
    assert saved["gdprtext_alignment"]["file"] == "data/reference/gdprtext-dpv-alignment.ttl"
    text = gb.render(out)
    assert "  Sin hallazgos nuevos: el puente no rompe ninguna regla ni agrega contradicciones." \
        in text
    assert "    [coherente] Consent ↔ GivenConsent · por dpv:Consent (exactMatch / exactMatch)" \
        in text
    assert "         Explicación de Consent." in text
    assert " · 32 coherentes, 2 dudosas, 0 incoherentes; solo explica" in text
    one = {**out, "summary": {**out["summary"], "coherente": 1, "dudosa": 1, "incoherente": 1}}
    assert " · 1 coherente, 1 dudosa, 1 incoherente; solo explica" in gb.render(one)


@needs_base
def test_rebuilding_gives_the_same_pairs_and_asks_nothing(tmp_path):
    first = gb.build(llm=ReviewLLM(), use_llm=True, check=False, out_dir=tmp_path)
    ttl = (tmp_path / f"{gb.MODULE}.ttl").read_bytes()
    module = Graph().parse(tmp_path / f"{gb.MODULE}.rdf")
    again = ReviewLLM()
    second = gb.build(llm=again, use_llm=True, check=False, out_dir=tmp_path)
    assert again.calls == 0 and second["ai"]["cached"] == 34
    assert [(r["ontopriv"], r["gdprtext"]) for r in second["rows"]] == \
        [(r["ontopriv"], r["gdprtext"]) for r in first["rows"]]
    assert (tmp_path / f"{gb.MODULE}.ttl").read_bytes() == ttl
    assert isomorphic(Graph().parse(tmp_path / f"{gb.MODULE}.rdf"), module)
    assert second["validation"] is None
    assert "  Forma y Lógica: no se revisaron" in gb.render(second)


@needs_base
def test_without_the_ai_the_pairs_wait(tmp_path, monkeypatch):
    import src.ai.llm_client as client

    class NoKey:
        def __init__(self, *a, **k):
            raise KeyError("LLM_API_KEY")
    monkeypatch.setattr(client, "LLMClient", NoKey)
    out = gb.build(use_llm=True, check=False, out_dir=tmp_path)
    assert out["ai"]["available"] is False
    assert out["ai"]["error"] == "Falta la variable 'LLM_API_KEY' en .env"
    assert out["summary"]["pending"] == 34 and out["pairs"] == 34
    assert ", 34 sin revisar; solo explica" in gb.render(out)
    assert "    [sin revisar] " in gb.render(out)
    off = gb.build(use_llm=False, check=False, out_dir=tmp_path / "otra")
    assert off["ai"]["error"] == "se pidió --sin-ia"


@needs_base
def test_a_bridge_that_breaks_a_rule_is_seen(tmp_path):
    parts, _ = si.base_parts()
    stray = URIRef("http://example.org/no-declarado")
    g = gb.build_graph([gb.Pair(str(stray), str(GDPR.Consent),
                                ((str(DPV.Consent), "exactMatch", "exactMatch"),))])
    path = tmp_path / "puente.ttl"
    g.serialize(path, format="turtle")
    bridge = si.Part("bridge", "puente.ttl", si.NEW, g, "x", path)
    v = gb.validate(parts, bridge, None, ENGINE)
    assert v["shapes"]["new_violations"] == 1                  # declared nowhere
    assert v["shapes"]["new"][0]["rule"] == "AlignedConceptShape"
    assert v["shapes"]["inherited_violations"] >= 1           # the base already had one
    assert not gb.clean(v)
    assert "  El puente trae hallazgos nuevos: revisa los reportes." in gb.validation_lines(v)


def test_clean_means_nothing_new():
    def v(sv=0, sw=0, le=0, lw=0):
        return {"shapes": {"new_violations": sv, "new_warnings": sw},
                "logic": {"new_errors": le, "new_warnings": lw}}
    assert gb.clean(v()) and not gb.clean(None)
    assert [gb.clean(v(**{k: 1})) for k in ("sv", "sw", "le", "lw")] == [False] * 4


def test_the_cli(monkeypatch, capsys):
    calls = []
    canned = {"prompt": gb.PROMPT.manifest(), "pairs": 1, "terms": 1, "loose_pairs": 2,
              "triples": 9, "files": ["data/output/ontopriv-gdpr-alignment.ttl"],
              "validation": None,
              "gdprtext_alignment": {"file": "data/reference/gdprtext-dpv-alignment.ttl",
                                     "sha256": "a" * 64},
              "ai": {"available": False, "error": "se pidió --sin-ia"},
              "summary": {"coherente": 0, "dudosa": 0, "incoherente": 0, "inválida": 0,
                          "pending": 1},
              "rows": [{"ontopriv": "Consent", "gdprtext": "GivenConsent",
                        "dpv_terms": "dpv:Consent", "ontopriv_relation": "exactMatch",
                        "gdprtext_relation": "exactMatch", "verdict": None,
                        "explanation": None}]}
    monkeypatch.setattr(gb, "build", lambda cfg, **kw: calls.append(kw) or canned)
    assert gb._run(["--sin-ia", "--motor", "owlrl"]) == 0
    assert calls == [{"use_llm": False, "engine": "owlrl"}]
    out = capsys.readouterr().out
    assert out.startswith("Puente de la base con el GDPR (S9-T09)")
    assert "IA revisora (bridge.v1): no usada (se pidió --sin-ia)" in out
    assert "    [sin revisar] Consent ↔ GivenConsent · por dpv:Consent" in out
    assert out.rstrip().endswith(gb.RESULT_FILE)
    monkeypatch.setattr(gb, "build", lambda cfg, **kw: (_ for _ in ()).throw(
        FileNotFoundError("Falta data/reference/gdprtext-dpv-alignment.ttl")))
    assert gb._run([]) == 1
    assert "Error: Falta data/reference/gdprtext-dpv-alignment.ttl" in capsys.readouterr().out
