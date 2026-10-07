"""S5-T09: the whole Sprint 5 flow end to end, through the same services the web uses.

candidates (Sprint 4) -> approve the AI concepts (S5-T03) -> the person aligns some concepts on
the card (S5-T05) -> the assistant proposals for the rest are confirmed in a batch, one of them
changed (S5-T08) -> the graph is written (S5-T06) from the web tab (S5-T07) -> the Protégé zip
resolves alignment -> profile -> core. Every number of the manifest is checked against the graph,
the CSV and the log.
"""
import csv
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from app.services import concept_review as cr
from app.services import mapping_review as mr
from app.services import alignment_graph as ag
from src.core.emit import CORE_IRI, PROFILE_IRI, CORE_FILE, PROFILE_FILE
from src.alignment.decisions import read_log, DECISIONS_FILE
from src.alignment.assistant_review import PROPOSALS_FILE
from src.alignment.write import ALIGNMENT_FILE, ALIGNMENT_CSV, alignment_iri
from src.alignment.bundle import BUNDLE_FILE, CATALOG_FILE

ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
DPV = "https://w3id.org/dpv#"
CFG = {"review": {"source_id": "ontopriv+lopdp", "reviewer": "Elian"}}
CAT = "{urn:oasis:names:tc:entity:xmlns:xml:catalog}"


def _rows(key, origin, name, cands, kinds=("class",), family=None, **kw):
    out = []
    for rank, (dpv, rel) in enumerate(cands, start=1):
        out.append({
            "concept_key": key, "origin": origin, "concept_name": name,
            "concept_label": kw.get("label"), "concept_kinds": list(kinds),
            "concept_family": family, "concept_definition": kw.get("definition"),
            "concept_articles": kw.get("articles", []),
            "duplicate_status": kw.get("status", "new"), "duplicate_of": kw.get("dup"),
            "duplicate_of_name": None, "duplicate_score": None, "duplicate_reason": None,
            "rank": rank, "dpv_iri": DPV + dpv, "dpv_name": dpv,
            "dpv_kind": "property" if dpv.startswith("has") else "class",
            "dpv_definition": f"def {dpv}", "dpv_parents": [], "score": 0.9 - rank / 10,
            "lexical": 0.5, "semantic": 0.5, "proposed_relation": rel,
            "justification": "j", "evidence_article": 8})
    return out


def _candidates():
    rows = []
    rows += _rows(ONTO + "Consent", "ontology", "Consent",
                  [("Consent", "skos:exactMatch"), ("ConsentRecord", "skos:relatedMatch")],
                  family="Principles")
    rows += _rows(ONTO + "Turnover", "ontology", "Turnover", [("Fee", "none")],
                  family="Terminology")
    rows += _rows(ONTO + "Data_Owner", "ontology", "Data_Owner",
                  [("DataSubject", "skos:exactMatch")], family="Terminology")
    rows += _rows(ONTO + "has_purpose", "ontology", "has_purpose",
                  [("hasPurpose", "skos:exactMatch")], kinds=("property",),
                  family="Processing")
    rows += _rows("ai:ImpactAssessment", "ai", "ImpactAssessment", [("DPIA", "skos:broadMatch")],
                  label="evaluación de impacto", definition="Análisis de riesgos.",
                  articles=[42])
    rows += _rows("ai:BusinessVolume", "ai", "BusinessVolume", [("Fee", "none")],
                  label="volumen de negocio", articles=[71])            # detector: new
    rows += _rows("ai:DataSubject", "ai", "DataSubject", [("DataSubject", "skos:exactMatch")],
                  articles=[4], status="possible_duplicate", dup=ONTO + "Data_Owner")
    rows += _rows("ai:Noise", "ai", "Noise", [("Fee", "none")])
    return {"metadata": {"embedding_model": "m",
                         "justification": {"llm_model": "gemini", "prompt_version": 2}},
            "rows": rows}


def _proposals():
    return {"metadata": {"assistant": "asistente de prueba", "created_at": "2026-10-01"},
            "proposals": [
        {"concept_key": ONTO + "Turnover", "concept_name": "Turnover", "family": "Terminology",
         "decision": "approve", "reason": "volumen de negocio",
         "mappings": [{"dpv_iri": DPV + "Turnover", "dpv_name": "Turnover",
                       "dpv_kind": "class", "relation": "skos:exactMatch",
                       "found_by": "search"}]},
        {"concept_key": ONTO + "Data_Owner", "concept_name": "Data_Owner",
         "family": "Terminology", "decision": "approve", "reason": "titular",
         "mappings": [{"dpv_iri": DPV + "DataSubject", "dpv_name": "DataSubject",
                       "relation": "skos:closeMatch", "found_by": "candidates"}]},
        {"concept_key": ONTO + "has_purpose", "concept_name": "has_purpose",
         "family": "Processing", "decision": "approve", "reason": "finalidad",
         "mappings": [{"dpv_iri": DPV + "hasPurpose", "dpv_name": "hasPurpose",
                       "relation": "skos:exactMatch", "found_by": "candidates"}]}]}


def _module(path, iri, imports=None, body=""):
    imp = f'<owl:imports rdf:resource="{imports}"/>' if imports else ""
    path.write_text(f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Ontology rdf:about="{iri}">{imp}</owl:Ontology>
  {body}
</rdf:RDF>""", encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    cand = out / "alignment-candidates.json"
    cand.write_text(json.dumps(_candidates()), encoding="utf-8")
    props = tmp_path / "review" / PROPOSALS_FILE
    props.parent.mkdir()
    props.write_text(json.dumps(_proposals()), encoding="utf-8")
    decl = "".join(f'<owl:Class rdf:about="{ONTO}{n}"/>'
                   for n in ("Consent", "Turnover", "Data_Owner"))
    decl += f'<owl:ObjectProperty rdf:about="{ONTO}has_purpose"/>'
    _module(out / CORE_FILE, CORE_IRI, body=decl)                     # Sprint 3 outputs
    _module(out / PROFILE_FILE, PROFILE_IRI, CORE_IRI)
    return {"candidates": cand, "log": tmp_path / "review" / DECISIONS_FILE,
            "dpv": tmp_path / "dpv.ttl", "proposals": props}


def _sprint5(paths):
    """T03 + T05 + T08 as the person does them in the web."""
    # S5-T03: the AI concepts
    state = cr.load(paths["candidates"], paths["log"], CFG)
    assert cr.apply(state, "ai:ImpactAssessment", "approve", "Elian", entity_kind="class")[0]
    assert cr.apply(state, "ai:BusinessVolume", "duplicate", "Elian",
                    same_as=ONTO + "Turnover")[0]                     # missed by the detector
    assert cr.apply(state, "ai:DataSubject", "duplicate", "Elian")[0]  # detector's suggestion
    assert cr.apply(state, "ai:Noise", "reject", "Elian")[0]
    assert cr.load(paths["candidates"], paths["log"], CFG)["progress"]["pending"] == 0
    # S5-T05: the person aligns two concepts on the card
    state = mr.load(paths["candidates"], paths["log"], CFG)
    assert [c.name for c in state["concepts"]] == ["ImpactAssessment", "Consent", "has_purpose",
                                                   "Data_Owner", "Turnover"]   # new, then by family
    assert mr.apply_approve(state, "ai:ImpactAssessment", DPV + "DPIA", "skos:exactMatch",
                            "Elian")[0]
    assert mr.apply_approve(state, ONTO + "Consent", DPV + "Consent", "skos:exactMatch",
                            "Elian")[0]
    # S5-T08: the assistant proposals for the other three, confirmed by families
    state = mr.load(paths["candidates"], paths["log"], CFG)
    assist = mr.load_assistant(state, paths["proposals"])
    assert [f for f, _ in mr.proposal_families(assist)] == ["Terminology", "Processing"]
    rows = mr.proposal_table(state, assist, "Terminology", prefill=True)
    rows[[r["Concepto"] for r in rows].index("Data_Owner")]["Tipo SKOS"] = \
        "exactMatch · equivalente"                                    # the person changes it
    ok, msg, _ = mr.confirm_table(state, assist, rows, "Elian")
    assert ok and "1 con cambios" in msg
    state = mr.load(paths["candidates"], paths["log"], CFG)
    assist = mr.load_assistant(state, paths["proposals"])
    rows = mr.proposal_table(state, assist, "Processing")
    rows[0]["Acción"] = "Sin correspondencia"                         # rejects the proposal
    assert mr.confirm_table(state, assist, rows, "Elian")[0]
    return mr.load(paths["candidates"], paths["log"], CFG)


def _resolve(folder, start):
    root = ET.parse(folder / CATALOG_FILE).getroot()
    files = {u.get("name"): u.get("uri") for u in root.iter(f"{CAT}uri")}
    g, todo, seen = Graph(), [start], set()
    while todo:
        iri = todo.pop()
        if iri not in seen:
            seen.add(iri)
            part = Graph().parse(folder / files[iri], format="xml")
            g += part
            todo += [str(o) for o in part.objects(URIRef(iri), OWL.imports)]
    return g


def test_sprint5_end_to_end(project, tmp_path):
    state = _sprint5(project)
    assert ag.readiness(state)["ready"]
    assert mr.load_assistant(state, project["proposals"])["open"] == []

    # S5-T07 (web tab) -> S5-T06 (writer)
    ok, msg = ag.write_graph(project, CFG)
    assert ok, msg
    graph = ag.current_graph(project)
    m = graph["manifest"]
    assert m["concepts"] == {"reviewed": 5, "total": 5, "aligned": 4, "no_match": 1,
                             "pending": 0}
    assert m["mappings"]["total"] == 4
    assert m["mappings"]["by_relation"]["skos:exactMatch"] == 4
    assert (m["mappings"]["decided_by_person"], m["mappings"]["assistant_confirmed"]) == (2, 2)
    assert m["mappings"]["found_by_search"] == 1
    assert m["review_provenance"] == {"reviewed_by_person": 2, "confirmed_from_assistant": 3,
                                      "accepted_as_proposed": 1, "changed_by_person": 2}
    assert m["ai_concepts"] == {"proposed": 4, "new_declared": 1, "new_by_kind": {"class": 1},
                                "duplicates": 2, "ontopriv_entities_with_law_articles": 2,
                                "rejected": 1}
    assert m["agreement_with_ai_type"]["same_type_as_ai"] == 2      # Consent and Data_Owner
    log_sha = hashlib.sha256(project["log"].read_bytes()).hexdigest()
    assert m["decisions_log"]["sha256"] == log_sha                  # traceable to the log

    # the graph says exactly what the manifest says
    g = Graph().parse(ag.output_dir(project) / ALIGNMENT_FILE, format="xml")
    assert len(g) == m["triples"]
    skos = [(str(s), str(o)) for p in (SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch,
                                       SKOS.narrowMatch, SKOS.relatedMatch)
            for s, o in g.subject_objects(p)]
    assert sorted(skos) == sorted([(ONTO + "Consent", DPV + "Consent"),
                                   (ONTO + "Data_Owner", DPV + "DataSubject"),
                                   (ONTO + "Turnover", DPV + "Turnover"),
                                   (PROFILE_IRI + "#ImpactAssessment", DPV + "DPIA")])
    ia = URIRef(PROFILE_IRI + "#ImpactAssessment")
    assert (ia, RDF.type, OWL.Class) in g
    assert (ia, RDFS.label, Literal("evaluación de impacto", lang="es")) in g
    assert (URIRef(ONTO + "Turnover"), DCTERMS.source, Literal("LOPDP, art. 71", lang="es")) in g
    assert (URIRef(ONTO + "Data_Owner"), DCTERMS.source, Literal("LOPDP, art. 4", lang="es")) in g
    assert not list(g.triples((URIRef(ONTO + "has_purpose"), None, None)))   # no match
    assert not list(g.triples((URIRef(PROFILE_IRI + "#Noise"), None, None)))  # rejected

    # the CSV has one row per triple, with who decided it
    with open(ag.output_dir(project) / ALIGNMENT_CSV, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(skos)
    assert {r["concept_name"]: r["decided_by"] for r in rows} == {
        "Consent": "person", "ImpactAssessment": "person",
        "Data_Owner": "assistant_confirmed", "Turnover": "assistant_confirmed"}

    # the zip opens the whole chain, and every mapped entity is declared in it
    folder = tmp_path / "protege"
    zipfile.ZipFile(io.BytesIO(graph["files"][BUNDLE_FILE])).extractall(folder)
    assert graph["bundle_missing"] == []
    full = _resolve(folder, alignment_iri(PROFILE_IRI))
    declared = {str(s) for s, p, o in full if p == RDF.type and o != OWL.Ontology}
    assert {s for s, _ in skos} <= declared


def test_the_graph_waits_for_the_review_and_notices_changes(project):
    state = cr.load(project["candidates"], project["log"], CFG)
    cr.apply(state, "ai:ImpactAssessment", "approve", "Elian", entity_kind="class")
    ok, msg = ag.write_graph(project, CFG)                    # 3 AI concepts still pending
    assert not ok and "conceptos de la IA" in msg
    assert ag.current_graph(project) is None

    state = _sprint5_after_first(project)
    assert ag.write_graph(project, CFG)[0]
    manifest = ag.current_graph(project)["manifest"]
    assert not ag.stale(state, manifest)
    assert mr.apply_undo(state, ONTO + "Consent", "Elian", DPV + "Consent")[0]
    state = mr.load(project["candidates"], project["log"], CFG)
    assert ag.stale(state, manifest)                          # the tab warns: write it again
    assert not ag.readiness(state)["ready"]                   # Consent is pending again
    assert len([d for d in read_log(project["log"]) if d.target == "mapping"]) == 6


def _sprint5_after_first(paths):
    """The rest of _sprint5 when ImpactAssessment was already approved."""
    state = cr.load(paths["candidates"], paths["log"], CFG)
    cr.apply(state, "ai:BusinessVolume", "duplicate", "Elian", same_as=ONTO + "Turnover")
    cr.apply(state, "ai:DataSubject", "duplicate", "Elian")
    cr.apply(state, "ai:Noise", "reject", "Elian")
    state = mr.load(paths["candidates"], paths["log"], CFG)
    mr.apply_approve(state, "ai:ImpactAssessment", DPV + "DPIA", "skos:exactMatch", "Elian")
    mr.apply_approve(state, ONTO + "Consent", DPV + "Consent", "skos:exactMatch", "Elian")
    for family in ("Terminology", "Processing"):
        state = mr.load(paths["candidates"], paths["log"], CFG)
        assist = mr.load_assistant(state, paths["proposals"])
        mr.confirm_table(state, assist, mr.proposal_table(state, assist, family, True), "Elian")
    return mr.load(paths["candidates"], paths["log"], CFG)