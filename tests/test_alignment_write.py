"""S5-T06: the aligned graph is written from the human decisions only, imports the profile (not
the DPV), declares the approved new concepts, attaches law articles to OntoPriv duplicates and
records who decided each correspondence."""
import csv
import json
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.core.emit import PROFILE_IRI
from src.alignment.decisions import (read_log, make_decision, append_decision, DECISIONS_FILE,
                                     TARGET_CONCEPT, DECISION_APPROVED, DECISION_REJECTED,
                                     DECISION_DUPLICATE)
from src.alignment.mapping_review import mapping_concepts, approve, mark_no_match
from src.alignment.write import (
    collect, build_graph, write_alignment, render_console, alignment_iri, article_source,
    ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV, CSV_COLUMNS,
    DECIDED_BY_PERSON, DECIDED_BY_ASSISTANT,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = "ontopriv+lopdp"
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
DPV = "https://w3id.org/dpv#"
META = {"embedding_model": "m", "justification": {"llm_model": "gemini", "prompt_version": 2}}
CREATED = "2026-10-07T12:00:00+00:00"


def _rows(key, origin, name, cands, kinds=("class",), family=None, **kw):
    out = []
    for rank, (dpv, rel) in enumerate(cands, start=1):
        r = {"concept_key": key, "origin": origin, "concept_name": name,
             "concept_label": kw.get("label"), "concept_kinds": list(kinds),
             "concept_family": family, "concept_definition": kw.get("definition"),
             "concept_articles": kw.get("articles", []), "duplicate_status": "new",
             "duplicate_of": kw.get("duplicate_of"), "duplicate_of_name": None,
             "duplicate_score": None, "duplicate_reason": None, "rank": rank,
             "dpv_iri": DPV + dpv, "dpv_name": dpv, "dpv_kind": "class", "score": 0.5,
             "lexical": 0.5, "semantic": 0.5, "proposed_relation": rel,
             "justification": "j", "evidence_article": 8}
        out.append(r)
    return out


def _data():
    rows = []
    rows += _rows(ONTO + "Consent", "ontology", "Consent",
                  [("Consent", "skos:exactMatch"), ("ConsentRecord", "skos:relatedMatch")],
                  family="Principles")
    rows += _rows(ONTO + "Turnover", "ontology", "Turnover", [("Fee", "none")],
                  family="Terminology")
    rows += _rows("ai:ImpactAssessment", "ai", "ImpactAssessment",
                  [("DPIA", "skos:broadMatch")], label="evaluación de impacto",
                  definition="Análisis de riesgos.", articles=[42])
    rows += _rows("ai:BusinessVolume", "ai", "BusinessVolume", [("Fee", "none")],
                  label="volumen de negocio", articles=[71, 72], duplicate_of=ONTO + "Fee_x")
    rows += _rows("ai:Noise", "ai", "Noise", [("Fee", "none")])
    return {"metadata": META, "rows": rows}


def _concept(log, key, decision, **extra):
    append_decision(make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key=key,
                                  decision=decision, reviewer="Elian", **extra), log)


def _decide_all(tmp_path, *, leave_pending=False):
    """Every concept decided: Consent aligned twice (one by the assistant), Turnover no match,
    ImpactAssessment new + aligned, BusinessVolume duplicate of Turnover, Noise rejected."""
    log = tmp_path / "review" / DECISIONS_FILE
    data = _data()
    _concept(log, "ai:ImpactAssessment", DECISION_APPROVED, entity_kind="class")
    _concept(log, "ai:BusinessVolume", DECISION_DUPLICATE, same_as=ONTO + "Turnover")
    _concept(log, "ai:Noise", DECISION_REJECTED)
    by = {c.key: c for c in mapping_concepts(data, read_log(log), SRC)}
    kw = dict(reviewer="Elian", source_id=SRC, log_path=log, metadata=META)
    approve(by[ONTO + "Consent"], DPV + "Consent", "skos:exactMatch", **kw)
    approve(by[ONTO + "Consent"], DPV + "ConsentRecord", "skos:relatedMatch",
            extra_snapshot={"proposed_by": "assistant"}, note="propuesta del asistente confirmada",
            **kw)
    approve(by["ai:ImpactAssessment"], DPV + "DPIA", "skos:exactMatch", **kw)
    if not leave_pending:
        mark_no_match(by[ONTO + "Turnover"], **kw)
    return data, log


def _content(tmp_path, **kw):
    data, log = _decide_all(tmp_path, **kw)
    return collect(data, read_log(log), SRC), log


# ---------------------------------------------------------------- gathering
def test_refuses_while_something_is_pending(tmp_path):
    data, log = _decide_all(tmp_path, leave_pending=True)
    with pytest.raises(ValueError, match="Faltan 1 de 3 conceptos por validar"):
        collect(data, read_log(log), SRC)
    log2 = tmp_path / "other.jsonl"                       # AI concepts never decided
    with pytest.raises(ValueError, match="Faltan 3 conceptos de la IA"):
        collect(data, read_log(log2), SRC)


def test_collect_reads_mappings_and_who_decided(tmp_path):
    content, _ = _content(tmp_path)
    got = {(m["concept_name"], m["dpv_name"], m["relation"], m["decided_by"])
           for m in content.mappings}
    assert got == {("Consent", "Consent", "skos:exactMatch", DECIDED_BY_PERSON),
                   ("Consent", "ConsentRecord", "skos:relatedMatch", DECIDED_BY_ASSISTANT),
                   ("ImpactAssessment", "DPIA", "skos:exactMatch", DECIDED_BY_PERSON)}
    assert [c["name"] for c in content.new_concepts] == ["ImpactAssessment"]
    assert content.duplicates[0]["duplicate_of"] == ONTO + "Turnover"   # the person's choice
    assert content.progress["no_match"] == 1 and content.progress["pending"] == 0
    assert content.agreement["same_type_as_ai"] == 2                 # Consent x2, not DPIA


# ---------------------------------------------------------------- the graph
def test_module_imports_the_profile_and_not_the_dpv(tmp_path):
    content, _ = _content(tmp_path)
    g = build_graph(content, created=CREATED)
    onto = URIRef(alignment_iri(PROFILE_IRI))
    assert (onto, RDF.type, OWL.Ontology) in g
    assert list(g.objects(onto, OWL.imports)) == [URIRef(PROFILE_IRI)]
    assert not [o for o in g.objects(None, OWL.imports) if str(o).startswith("https://w3id.org")]
    assert not list(g.triples((None, OWL.equivalentClass, None)))     # SKOS only, no OWL axioms
    assert str(onto) == "http://www.semanticweb.org/profiles/ecuador-lopdp-dpv-alignment"


def test_one_skos_triple_per_approved_mapping(tmp_path):
    content, _ = _content(tmp_path)
    g = build_graph(content)
    consent, ia = URIRef(ONTO + "Consent"), URIRef(PROFILE_IRI + "#ImpactAssessment")
    assert (consent, SKOS.exactMatch, URIRef(DPV + "Consent")) in g
    assert (consent, SKOS.relatedMatch, URIRef(DPV + "ConsentRecord")) in g
    assert (ia, SKOS.exactMatch, URIRef(DPV + "DPIA")) in g
    assert not list(g.triples((URIRef(ONTO + "Turnover"), SKOS.closeMatch, None)))  # no match


def test_new_concepts_and_duplicate_articles(tmp_path):
    content, _ = _content(tmp_path)
    g = build_graph(content)
    ia = URIRef(PROFILE_IRI + "#ImpactAssessment")
    assert (ia, RDF.type, OWL.Class) in g
    assert (ia, RDFS.label, Literal("evaluación de impacto", lang="es")) in g
    assert (ia, SKOS.definition, Literal("Análisis de riesgos.", lang="es")) in g
    assert (ia, DCTERMS.source, Literal("LOPDP, art. 42", lang="es")) in g
    turnover = URIRef(ONTO + "Turnover")                              # gets the law articles
    assert set(g.objects(turnover, DCTERMS.source)) == {Literal("LOPDP, art. 71", lang="es"),
                                                         Literal("LOPDP, art. 72", lang="es")}
    assert not list(g.triples((URIRef(ONTO + "Fee_x"), None, None)))  # not the detected one
    assert not list(g.triples((URIRef(PROFILE_IRI + "#Noise"), None, None)))   # rejected


def test_profile_iri_and_law_are_parameters(tmp_path):
    data, log = _decide_all(tmp_path)
    other = "http://example.org/profiles/peru-ley29733"
    content = collect(data, read_log(log), SRC, profile_iri=other)
    assert content.new_concepts[0]["iri"] == other + "#ImpactAssessment"
    g = build_graph(content, profile_iri=other, profile_name="Perú", law_name="Ley 29733")
    onto = URIRef(other + "-dpv-alignment")
    assert list(g.objects(onto, OWL.imports)) == [URIRef(other)]
    assert (URIRef(other + "#ImpactAssessment"), SKOS.exactMatch, URIRef(DPV + "DPIA")) in g
    assert article_source("Ley 29733", 5) == "Ley 29733, art. 5"
    assert (URIRef(ONTO + "Turnover"), DCTERMS.source,
            Literal("Ley 29733, art. 71", lang="es")) in g


# ---------------------------------------------------------------- files
def test_write_the_three_files(tmp_path):
    content, log = _content(tmp_path)
    out = tmp_path / "out"
    result = write_alignment(content, out, source_id=SRC, log_path=log, created=CREATED)
    assert {p.name for p in out.iterdir()} == {ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV}
    g = Graph().parse(result.rdf_path, format="xml")                  # valid RDF/XML
    assert len(list(g.triples((None, SKOS.exactMatch, None)))) == 2
    m = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert m["imports"] == PROFILE_IRI and m["imports_dpv"] is False
    assert m["concepts"] == {"reviewed": 3, "total": 3, "aligned": 2, "no_match": 1,
                             "pending": 0}
    assert m["mappings"]["total"] == 3 and m["mappings"]["assistant_confirmed"] == 1
    assert m["ai_concepts"]["new_declared"] == 1 and m["ai_concepts"]["rejected"] == 1
    assert m["decisions_log"]["sha256"] and len(m["decisions_log"]["sha256"]) == 64
    assert m["triples"] == len(g) and m["created"] == CREATED
    raw = Path(result.csv_path).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                            # BOM for Excel
    with open(result.csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == CSV_COLUMNS and len(rows) == 3
    assert {r["decided_by"] for r in rows} == {DECIDED_BY_PERSON, DECIDED_BY_ASSISTANT}
    text = render_console(result)
    assert "Correspondencias SKOS: 3" in text and "el DPV no se importa" in text


def test_real_alignment():
    cand = ROOT / "data/output/alignment-candidates.json"
    log = ROOT / "data/review" / DECISIONS_FILE
    if not cand.exists() or not log.exists():
        pytest.skip("Faltan los candidatos (Sprint 4) o el registro de decisiones")
    data = json.loads(cand.read_text(encoding="utf-8"))
    try:
        content = collect(data, read_log(log), SRC)
    except ValueError:
        pytest.skip("La revision todavia no esta completa")
    g = build_graph(content)
    assert content.progress["reviewed"] == content.progress["total"] == 552
    assert len(content.mappings) == content.progress["mappings"]
    assert sum(1 for _ in g.triples((None, SKOS.exactMatch, None))) == \
        content.progress["relations"]["skos:exactMatch"]
    assert len(content.new_concepts) == 25
    assert all(str(m["subject_iri"]).startswith("http") for m in content.mappings)