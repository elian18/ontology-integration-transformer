"""Tests for the PROV-O of the base (S10-T06).

A toy base (the decisions of tests/test_alignment_write.py, a repair of each rule and one pair of
the bridge) checks the records one by one; the real base checks that each of its 190
correspondences says who and when, that "Elian" and "Elian Gavilanes" are one person, that Forma
and Lógica find nothing new and that the four formats hold the same triples. No AI is called."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.alignment import gdpr_bridge as gb
from src.alignment.decisions import (DECISION_APPROVED, DECISION_DUPLICATE, DECISION_REJECTED,
                                     DECISIONS_FILE, TARGET_CONCEPT, append_decision,
                                     make_decision, read_log)
from src.alignment.mapping_review import (approve, mapping_concepts, mark_no_match, reject,
                                          undo)
from src.alignment.write import alignment_iri, build_graph as alignment_graph, collect
from src.core.emit import PROFILE_IRI
from src.provenance import base as pb
from src.provenance import model as pm
from src.provenance.model import FOAF, PROV
from src.validation import shape_inspector as si
from tests.test_alignment_write import DPV, META, ONTO, SRC, _data
from tests.test_questions import HAS_JENA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output"
needs_base = pytest.mark.skipif(
    not all((OUT / n).exists() for n in (
        "ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf",
        "alignment-candidates.json", "core-repairs.json", "ontopriv-gdpr-alignment.rdf",
        gb.RESULT_FILE)),
    reason="Falta la base en data/output")
ENGINE = "jena" if HAS_JENA else "owlrl"
GDPR = gb.GDPRTEXT_IRI + "#"
PEOPLE = {"Elian Gavilanes": ["Elian"]}
PROPOSAL = {"proposed_by": "assistant", "proposal_assistant": "Asistente de prueba",
            "proposal_created_at": "2026-10-01", "assistant_confirmation": "accepted",
            "proposal": {"decision": "approve", "reason": "registro del consentimiento",
                         "mappings": [{"dpv_iri": DPV + "ConsentRecord",
                                       "relation": "skos:relatedMatch",
                                       "found_by": "candidates"}]}}
REPAIRS = {"changes": [
    {"rule": "several_domains", "property": "p", "iri": ONTO + "p", "classes": ["A", "B"],
     "module": "core"},
    {"rule": "double_type", "property": "q", "iri": ONTO + "q", "kept": "propiedad de objeto",
     "removed": "propiedad de datos", "evidence": {"resources": 1, "texts": 0,
                                                   "sub_properties": 0},
     "removed_ranges": ["string"], "module": "profile"}]}
PAIR = {"ontopriv": "Consent", "ontopriv_iri": ONTO + "Consent", "gdprtext": "GivenConsent",
        "gdprtext_iri": GDPR + "GivenConsent", "dpv_terms": "dpv:Consent",
        "ontopriv_relation": "exactMatch", "gdprtext_relation": "exactMatch",
        "ontopriv_source": "LOPDP, art. 8", "gdpr_articles": "article6-1-a",
        "verdict": "coherente", "ai_relation": "exactMatch", "explanation": "Explicación."}
BRIDGE = {"created": "2026-11-08T00:00:00+00:00", "rows": [PAIR],
          "prompt": {"name": "bridge", "version": 1, "sha256": "b" * 64}}
CACHE = [{"concept_key": f"{ONTO}Consent {GDPR}GivenConsent", "prompt": BRIDGE["prompt"],
          "model": "modelo-de-prueba", "created": "2026-11-08T01:00:00Z",
          "answer": {"verdict": "coherente", "relation": "exactMatch",
                     "explanation": "Explicación."}},
         {"concept_key": f"{ONTO}Consent {GDPR}GivenConsent", "prompt": BRIDGE["prompt"],
          "model": "modelo-de-prueba", "created": "2026-11-07T01:00:00Z",
          "answer": {"verdict": "dudosa", "relation": "closeMatch", "explanation": "Antes."}},
         {"concept_key": f"{ONTO}Consent {GDPR}GivenConsent", "prompt": BRIDGE["prompt"],
          "model": "modelo-de-prueba", "created": "2026-11-06T01:00:00Z",
          "answer": {"verdict": "coherente", "relation": "exactMatch",
                     "explanation": "Explicación."}}]


def _log(tmp_path):
    """The toy base decided: three AI concepts (new, duplicate, rejected) and four mapping
    decisions, signed "Elian" and "Elian Gavilanes"."""
    log = tmp_path / "review" / DECISIONS_FILE

    def concept(key, decision, at, reviewer="Elian", **extra):
        append_decision(make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key=key,
                                      decision=decision, reviewer=reviewer, decided_at=at,
                                      **extra), log)
    concept("ai:ImpactAssessment", DECISION_APPROVED, "2026-10-01T10:00:00+00:00",
            entity_kind="class", note="evaluación de la ley")
    concept("ai:BusinessVolume", DECISION_DUPLICATE, "2026-10-01T10:05:00+00:00",
            reviewer="Elian Gavilanes", same_as=ONTO + "Turnover")
    concept("ai:Noise", DECISION_REJECTED, "2026-10-01T10:10:00+00:00")
    data = _data()
    by = {c.key: c for c in mapping_concepts(data, read_log(log), SRC)}
    kw = dict(reviewer="Elian Gavilanes", source_id=SRC, log_path=log, metadata=META)
    approve(by[ONTO + "Consent"], DPV + "Consent", "skos:exactMatch",
            decided_at="2026-10-02T09:00:00+00:00", **kw)
    approve(by[ONTO + "Consent"], DPV + "ConsentRecord", "skos:relatedMatch",
            decided_at="2026-10-02T09:01:00+00:00", extra_snapshot=PROPOSAL,
            note="propuesta del asistente confirmada", **kw)
    approve(by["ai:ImpactAssessment"], DPV + "DPIA", "skos:exactMatch",
            decided_at="2026-10-02T09:02:00+00:00", **kw)
    mark_no_match(by[ONTO + "Turnover"], decided_at="2026-10-02T09:03:00+00:00", **kw)
    return data, log


def _inputs(tmp_path, *, bridge=True, alignment=None) -> pb.Inputs:
    data, log = _log(tmp_path)
    records = read_log(log)
    files = {"log": log}
    for name in ("core", "profile", "alignment", "candidates", "ontopriv", "lopdp",
                 *(("bridge",) if bridge else ())):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        files[name] = path
    if alignment is None:
        alignment = alignment_graph(collect(data, records, SRC))
    return pb.Inputs(data=data, records=records, alignment=alignment, source_id=SRC,
                     files=files, alignment_manifest={"created": "2026-10-07T20:29:30+00:00"},
                     repairs=REPAIRS, proposed={"counts": {"articles_processed": 3,
                                                           "concepts_proposed": 3}},
                     bridge=BRIDGE if bridge else None, bridge_cache=CACHE if bridge else [])


def _settings():
    return pb.Settings(people=PEOPLE, extraction_model="modelo-extractor", extraction_prompt=1)


@pytest.fixture
def toy(tmp_path):
    inp = _inputs(tmp_path)
    g, counts = pb.build_graph(inp, _settings())
    return inp, g, counts


def _node(local):
    return URIRef(pb.PROVENANCE_IRI + "#" + local)


def _record(g, s, p, o):
    found = [r for r in g.subjects(OWL.annotatedSource, URIRef(s))
             if (r, OWL.annotatedProperty, p) in g and (r, OWL.annotatedTarget, URIRef(o)) in g]
    assert len(found) == 1
    return found[0]


def _about(g, term, prefix):
    return [r for r in pm.records_about(g, term) if pb.local(r).startswith(prefix)]


# ------------------------------------------------------------------ the building blocks

def test_digest_slug_and_when():
    assert pm.digest("a", "b") == pm.digest("a", "b") != pm.digest("a b", "c")
    assert len(pm.digest("x")) == 16
    assert pm.slug("Elian Gavilanes") == "elian-gavilanes"
    assert pm.slug("gemini-3.1-flash-lite") == "gemini-3-1-flash-lite"
    assert pm.slug("Ñandú ") == "nandu" and pm.slug("***") == "x"
    assert pm.when("2026-10-02T09:00:00+00:00") == Literal("2026-10-02T09:00:00+00:00",
                                                           datatype=XSD.dateTime)
    assert pm.when("2026-11-08T01:00:00Z").datatype == XSD.dateTime
    assert pm.when("2026-10-01") is None and pm.when("ayer") is None and pm.when(None) is None


def test_the_names_of_one_person_are_one_agent():
    rec = pm.Recorder("http://example.org/p", "p", pm.People(PEOPLE))
    assert rec.person("Elian") == rec.person("Elian Gavilanes") == rec.person(" Elian ")
    assert rec.person("Ana") != rec.person("Elian")
    people = set(rec.g.subjects(RDF.type, PROV.Person))
    assert len(people) == 2
    elian = rec.person("Elian")
    assert set(rec.g.objects(elian, FOAF.nick)) == {Literal("Elian")}
    assert rec.g.value(elian, FOAF.name) == Literal("Elian Gavilanes")
    assert rec.g.value(elian, RDFS.label) == Literal("Elian Gavilanes", lang="es")
    alone = pm.Recorder("http://example.org/p", "p")
    assert alone.person("Elian") != alone.person("Elian Gavilanes")


# ------------------------------------------------------------------ the correspondences

def test_each_correspondence_says_who_and_when(toy):
    inp, g, counts = toy
    assert counts["mappings"] == {"total": 3, "assistant": 1, "person": 2, "replaced": 0}
    assert pb.missing(g, inp.alignment) == []
    elian = _node("person-elian-gavilanes")
    first = _record(g, ONTO + "Consent", SKOS.exactMatch, DPV + "Consent")
    assert (first, RDF.type, PROV.Entity) in g and (first, RDF.type, OWL.Axiom) in g
    assert g.value(first, PROV.wasAttributedTo) == elian
    assert g.value(first, PROV.generatedAtTime) == pm.when("2026-10-02T09:00:00+00:00")
    assert g.value(first, PROV.wasGeneratedBy) == _node("activity-mapping-review")
    assert g.value(first, _node("decidedBy")) == Literal("base", lang="es")
    assert g.value(first, OWL.versionInfo) == Literal("1")
    assert g.value(first, DCTERMS.isPartOf) == _node("module-alignment")
    assert set(g.objects(first, PROV.wasDerivedFrom)) == {_node("file-candidates")}
    text = str(g.value(first, RDFS.comment))
    assert text.startswith("Decidida por Elian Gavilanes el 2026-10-02 en la tarjeta de "
                           "revisión, sin propuesta del asistente. Candidato n.º 1 del DPV")
    assert "La IA (gemini) había propuesto skos:exactMatch con el art. 8 de la LOPDP" in text
    assert str(g.value(first, RDFS.label)) == "Consent skos:exactMatch dpv:Consent"
    assert pb.local(first) == "mapping-" + pm.digest(
        alignment_iri(PROFILE_IRI), ONTO + "Consent", str(SKOS.exactMatch), DPV + "Consent")


def _corrected(tmp_path):
    """The toy base, corrected later as in S10-T16: ImpactAssessment gets a searched term
    (broadMatch), the old one is discarded and the new one changes to closeMatch."""
    inp = _inputs(tmp_path)
    log = inp.files["log"]
    by = {c.key: c for c in mapping_concepts(inp.data, read_log(log), SRC)}
    kw = dict(reviewer="Elian Gavilanes", source_id=SRC, log_path=log, metadata=META)
    found = SimpleNamespace(iri=DPV + "ImpactAssessment", name="ImpactAssessment",
                            kind="class")
    approve(by["ai:ImpactAssessment"], found.iri, "skos:broadMatch", target=found,
            decided_at="2026-11-14T10:00:00+00:00", **kw)
    reject(by["ai:ImpactAssessment"], DPV + "DPIA", decided_at="2026-11-14T10:01:00+00:00",
           **kw)
    approve(by["ai:ImpactAssessment"], found.iri, "skos:closeMatch", target=found,
            decided_at="2026-11-14T10:02:00+00:00", **kw)
    inp.records = read_log(log)
    inp.alignment = alignment_graph(collect(inp.data, inp.records, SRC))
    return inp


def test_the_approvals_of_each_concept_and_their_versions(tmp_path):
    inp = _corrected(tmp_path)
    history = pb.approvals(inp.records, SRC)
    impact = history["ai:ImpactAssessment"]
    assert [(local_name(a.decision.dpv_iri), a.decision.relation, a.version,
             a.ended.decision if a.ended else None) for a in impact] == [
        ("DPIA", "skos:exactMatch", 1, "rejected"),
        ("ImpactAssessment", "skos:broadMatch", 2, "approved"),
        ("ImpactAssessment", "skos:closeMatch", 3, None)]
    assert [b.version for b in pb.revised(impact[2], impact)] == [2]
    assert [b.version for b in pb.revised(impact[1], impact)] == [1]
    assert pb.revised(impact[0], impact) == []
    assert [a.version for a in history[ONTO + "Consent"]] == [1, 1]     # nothing ended
    assert pb.approvals(inp.records, "otra-fuente") == {}


def test_approving_again_changes_nothing_and_undoing_ends_it(tmp_path):
    inp = _inputs(tmp_path)
    log = inp.files["log"]
    by = {c.key: c for c in mapping_concepts(inp.data, read_log(log), SRC)}
    kw = dict(reviewer="Elian", source_id=SRC, log_path=log, metadata=META)
    approve(by[ONTO + "Consent"], DPV + "Consent", "skos:exactMatch",
            decided_at="2026-11-14T09:00:00+00:00", **kw)               # the same again
    undo(by[ONTO + "Consent"], DPV + "ConsentRecord", reviewer="Elian", source_id=SRC,
         log_path=log, decided_at="2026-11-14T09:05:00+00:00")
    inp.records = read_log(log)
    inp.alignment = alignment_graph(collect(inp.data, inp.records, SRC))
    consent = pb.approvals(inp.records, SRC)[ONTO + "Consent"]
    assert [(a.version, a.at, a.ended.decision if a.ended else None) for a in consent] == [
        (1, "2026-10-02T09:00:00+00:00", None), (1, "2026-10-02T09:01:00+00:00", "pending")]
    g, counts = pb.build_graph(inp, _settings())
    assert counts["mappings"]["replaced"] == 1
    [old] = [r for r in g.subjects(RDF.type, RDF.Statement)]
    assert str(g.value(old, RDFS.comment)).endswith(
        "Deshecha por Elian Gavilanes el 2026-11-14.")
    assert g.value(old, PROV.wasRevisionOf) is None and not list(
        g.subjects(PROV.wasRevisionOf, old))                  # nothing replaced it


def local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1]


def test_a_corrected_correspondence_revises_the_one_before(tmp_path):
    inp = _corrected(tmp_path)
    g, counts = pb.build_graph(inp, _settings())
    assert counts["mappings"]["replaced"] == 2 and counts["mappings"]["total"] == 3
    assert pb.missing(g, inp.alignment) == []
    now = _record(g, PROFILE_IRI + "#ImpactAssessment", SKOS.closeMatch,
                  DPV + "ImpactAssessment")
    assert g.value(now, OWL.versionInfo) == Literal("3")
    assert g.value(now, PROV.generatedAtTime) == pm.when("2026-11-14T10:02:00+00:00")
    v2 = g.value(now, PROV.wasRevisionOf)
    v1 = g.value(v2, PROV.wasRevisionOf)
    assert str(g.value(v2, RDFS.label)) == ("ImpactAssessment skos:broadMatch "
                                            "dpv:ImpactAssessment (versión 2, ya no está en "
                                            "la base)")
    assert str(g.value(v2, RDFS.comment)).endswith(
        "Cambiada a skos:closeMatch por Elian Gavilanes el 2026-11-14.")
    assert str(g.value(v1, RDFS.comment)) == (
        "Decidida por Elian Gavilanes el 2026-10-02 en la tarjeta de revisión, sin propuesta "
        "del asistente. Descartada por Elian Gavilanes el 2026-11-14.")
    assert pb.local(v1) == pb.local(pb.mapping_record(
        PROFILE_IRI + "#ImpactAssessment", SKOS.exactMatch, DPV + "DPIA")) + "-v1"
    for old, at in ((v1, "2026-11-14T10:01:00+00:00"), (v2, "2026-11-14T10:02:00+00:00")):
        assert (old, RDF.type, RDF.Statement) in g and (old, RDF.type, OWL.Axiom) not in g
        assert g.value(old, PROV.invalidatedAtTime) == pm.when(at)
        assert g.value(old, PROV.wasInvalidatedBy) == _node("activity-mapping-review")
        assert g.value(old, PROV.wasAttributedTo) == _node("person-elian-gavilanes")
        assert g.value(old, _node("decidedBy")) == Literal("base", lang="es")
    assert (g.value(v1, OWL.versionInfo), g.value(v2, OWL.versionInfo)) == (
        Literal("1"), Literal("2"))
    assert g.value(v1, RDF.object) == URIRef(DPV + "DPIA")
    assert g.value(v1, PROV.wasRevisionOf) is None
    review = _node("activity-mapping-review")
    assert g.value(review, PROV.endedAtTime) == pm.when("2026-11-14T10:02:00+00:00")
    consent = _record(g, ONTO + "Consent", SKOS.exactMatch, DPV + "Consent")
    assert g.value(consent, OWL.versionInfo) == Literal("1")
    assert g.value(consent, PROV.wasRevisionOf) is None
    lines = pb.render_records(g, PROFILE_IRI + "#ImpactAssessment").splitlines()
    heads = [x for x in lines if x.startswith("  · ImpactAssessment skos:")]
    assert heads == [
        "  · ImpactAssessment skos:closeMatch dpv:ImpactAssessment",     # in force first
        "  · ImpactAssessment skos:broadMatch dpv:ImpactAssessment (versión 2, ya no está en "
        "la base)",
        "  · ImpactAssessment skos:exactMatch dpv:DPIA (versión 1, ya no está en la base)"]
    assert ("    Reemplaza a: ImpactAssessment skos:broadMatch dpv:ImpactAssessment (versión 2, "
            "ya no está en la base)") in lines
    assert ("    La reemplazó: ImpactAssessment skos:broadMatch dpv:ImpactAssessment (versión "
            "2, ya no está en la base)") in lines
    assert sum(x.startswith("    Reemplaza a: ") for x in lines) == 2
    assert sum(x.startswith("    La reemplazó: ") for x in lines) == 2


def test_a_confirmed_proposal_comes_from_the_assistant(toy):
    _, g, _ = toy
    rec = _record(g, ONTO + "Consent", SKOS.relatedMatch, DPV + "ConsentRecord")
    assert "confirmando la propuesta del asistente" in str(g.value(rec, RDFS.comment))
    assert "Nota:" not in str(g.value(rec, RDFS.comment))
    proposal = _node("proposal-" + pm.digest(ONTO + "Consent"))
    assert (rec, PROV.wasDerivedFrom, proposal) in g
    assert g.value(proposal, PROV.wasAttributedTo) == _node("agent-assistant")
    assert g.value(proposal, PROV.wasGeneratedBy) == _node("activity-assistant-proposals")
    assert str(g.value(proposal, RDFS.comment)) == (
        "Propuso skos:relatedMatch dpv:ConsentRecord: «registro del consentimiento».")
    assert g.value(proposal, DCTERMS.subject) == URIRef(ONTO + "Consent")
    assistant = _node("agent-assistant")
    assert g.value(assistant, RDFS.label) == Literal("Asistente de prueba", lang="es")
    assert (assistant, PROV.actedOnBehalfOf, _node("person-elian-gavilanes")) in g
    proposing = _node("activity-assistant-proposals")
    assert g.value(proposing, DCTERMS.date) == Literal("2026-10-01", datatype=XSD.date)


def test_a_correspondence_of_a_law_concept_keeps_its_article(toy):
    _, g, _ = toy
    rec = _record(g, PROFILE_IRI + "#ImpactAssessment", SKOS.exactMatch, DPV + "DPIA")
    assert set(g.objects(rec, PROV.hadPrimarySource)) == {_node("lopdp-art-42")}
    decision = _node("concept-decision-" + pm.digest("ai:ImpactAssessment"))
    assert (rec, PROV.wasDerivedFrom, decision) in g
    assert g.value(_node("lopdp-art-42"), RDFS.label) == Literal("LOPDP, art. 42", lang="es")
    assert g.value(_node("lopdp-art-42"), DCTERMS.isPartOf) == _node("file-lopdp")


def test_a_concept_without_counterpart_says_who_and_when(toy):
    _, g, counts = toy
    assert counts["no_match"] == 1
    [rec] = _about(g, ONTO + "Turnover", "no-match-")
    assert g.value(rec, PROV.wasAttributedTo) == _node("person-elian-gavilanes")
    assert g.value(rec, PROV.generatedAtTime) == pm.when("2026-10-02T09:03:00+00:00")
    assert str(g.value(rec, RDFS.comment)) == (
        "Sin correspondencia en el DPV. Decidido por Elian Gavilanes el 2026-10-02 en la "
        "tarjeta de revisión, sin propuesta del asistente.")
    assert set(g.objects(rec, PROV.hadPrimarySource)) == {_node("lopdp-art-71"),
                                                         _node("lopdp-art-72")}


# ------------------------------------------------------------------ the concepts of the law

def test_each_concept_of_the_law_says_its_article_and_who_decided(toy):
    _, g, counts = toy
    assert counts["law_concepts"] == {"total": 3, "approved": 1, "duplicate": 1, "rejected": 1}
    elian = _node("person-elian-gavilanes")
    new = _node("concept-decision-" + pm.digest("ai:ImpactAssessment"))
    assert g.value(new, PROV.wasAttributedTo) == elian                  # signed "Elian"
    assert g.value(new, DCTERMS.subject) == URIRef(PROFILE_IRI + "#ImpactAssessment")
    assert str(g.value(new, RDFS.label)) == "ImpactAssessment: concepto nuevo del perfil"
    assert str(g.value(new, RDFS.comment)).endswith(
        "Decidido por Elian Gavilanes el 2026-10-01. Nota: evaluación de la ley.")
    extracted = _node("law-concept-" + pm.digest("ai:ImpactAssessment"))
    assert (new, PROV.wasDerivedFrom, extracted) in g
    assert g.value(extracted, PROV.wasAttributedTo) == _node("agent-llm-modelo-extractor")
    assert g.value(extracted, PROV.wasGeneratedBy) == _node("activity-law-extraction")
    assert str(g.value(extracted, RDFS.comment)) == (
        "Propuesto por la IA con base en el art. 42 de la LOPDP, como clase: Análisis de "
        "riesgos.")
    dup = _node("concept-decision-" + pm.digest("ai:BusinessVolume"))
    assert g.value(dup, DCTERMS.subject) == URIRef(ONTO + "Turnover")    # the person's choice
    assert "la comprobación automática había encontrado Fee_x" in str(g.value(dup, RDFS.comment))
    assert set(g.objects(dup, PROV.hadPrimarySource)) == {_node("lopdp-art-71"),
                                                         _node("lopdp-art-72")}
    assert "los arts. 71 y 72 de la LOPDP" in str(g.value(
        _node("law-concept-" + pm.digest("ai:BusinessVolume")), RDFS.comment))
    gone = _node("concept-decision-" + pm.digest("ai:Noise"))
    assert g.value(gone, DCTERMS.subject) is None
    assert str(g.value(gone, RDFS.label)) == "Noise: rechazado"
    noise = _node("law-concept-" + pm.digest("ai:Noise"))
    assert g.value(noise, PROV.wasInvalidatedBy) == _node("activity-concept-review")
    assert g.value(noise, PROV.invalidatedAtTime) == pm.when("2026-10-01T10:10:00+00:00")
    assert g.value(noise, DCTERMS.subject) == URIRef(PROFILE_IRI + "#Noise")   # as proposed
    assert g.value(extracted, DCTERMS.subject) == URIRef(PROFILE_IRI + "#ImpactAssessment")
    assert g.value(extracted, PROV.wasInvalidatedBy) is None


def test_the_extraction_followed_its_prompt(toy):
    _, g, _ = toy
    extraction = _node("activity-law-extraction")
    assert str(g.value(extraction, RDFS.comment)) == (
        "La IA lee la ley por lotes de artículos y propone conceptos de protección de datos, "
        "cada uno con el artículo del que sale (S3); solo propone: una persona decide cada uno. "
        "3 artículos procesados, 3 conceptos propuestos.")
    plans = {g.value(a, PROV.hadPlan) for a in g.objects(extraction, PROV.qualifiedAssociation)}
    assert plans == {_node("plan-prompt-extract-v1")}
    from src.ai.prompts import load_prompt
    assert g.value(_node("plan-prompt-extract-v1"), DCTERMS.identifier) == Literal(
        "sha256:" + load_prompt("extract", 1).manifest()["sha256"])
    review = _node("activity-concept-review")
    assert g.value(review, PROV.startedAtTime) == pm.when("2026-10-01T10:00:00+00:00")
    assert g.value(review, PROV.endedAtTime) == pm.when("2026-10-01T10:10:00+00:00")
    assert set(g.objects(review, PROV.wasAssociatedWith)) == {_node("person-elian-gavilanes")}
    candidates = _node("activity-candidates")
    plans = {g.value(a, PROV.hadPlan) for a in g.objects(candidates, PROV.qualifiedAssociation)}
    assert plans == {_node("plan-prompt-justify-v2")}


# ------------------------------------------------------------------ repairs and bridge

def test_the_repairs_r1_and_r2(toy):
    _, g, counts = toy
    assert counts["repairs"] == {"R1": 1, "R2": 1}
    [r1] = _about(g, ONTO + "p", "repair-")
    assert g.value(r1, PROV.wasGeneratedBy) == _node("activity-repair-r1")
    assert g.value(r1, DCTERMS.isPartOf) == _node("module-core")
    assert str(g.value(r1, RDFS.comment)) == (
        "R1 en el núcleo: tenía 2 dominios (A, B); ahora tiene uno, la unión de esas clases "
        "(owl:unionOf), como pide P19 de OOPS!.")
    [r2] = _about(g, ONTO + "q", "repair-")
    assert g.value(r2, PROV.wasGeneratedBy) == _node("activity-repair-r2")
    assert g.value(r2, DCTERMS.isPartOf) == _node("module-profile")
    assert str(g.value(r2, RDFS.label)) == "R2 · q"
    assert str(g.value(r2, RDFS.comment)).endswith(
        "Uso: 1 valor recurso, 0 textos, 0 subpropiedades. Se quitaron los rangos que ya no le "
        "sirven: string.")
    assert (_node("activity-core-build"), PROV.wasInformedBy, _node("activity-repair-r2")) in g


def test_each_pair_of_the_bridge_comes_from_the_base(toy):
    _, g, counts = toy
    assert counts["bridge"] == {"pairs": 1, "summary": None}
    pair = _record(g, ONTO + "Consent", SKOS.closeMatch, GDPR + "GivenConsent")
    base = _record(g, ONTO + "Consent", SKOS.exactMatch, DPV + "Consent")
    assert (pair, PROV.wasDerivedFrom, base) in g
    assert g.value(pair, PROV.wasAttributedTo) == _node("agent-system")
    assert g.value(pair, DCTERMS.isPartOf) == _node("module-bridge")
    assert g.value(pair, _node("decidedBy")) is None                      # not a DPV mapping
    assert "Explicación" not in str(g.value(pair, RDFS.comment))
    [note] = list(g.subjects(DCTERMS.subject, pair))
    assert g.value(note, PROV.wasAttributedTo) == _node("agent-llm-modelo-de-prueba")
    assert g.value(note, PROV.generatedAtTime) == pm.when("2026-11-08T01:00:00Z")  # latest
    assert str(g.value(note, RDFS.comment)) == (
        "Dijo «coherente» (exactMatch): Explicación. Solo explica: no cambia el par ni entra a "
        "la ontología.")
    review = _node("activity-bridge-review")
    assert g.value(review, PROV.startedAtTime) == pm.when("2026-11-06T01:00:00Z")
    assert g.value(review, PROV.endedAtTime) == pm.when("2026-11-08T01:00:00Z")
    plans = {g.value(a, PROV.hadPlan) for a in g.objects(review, PROV.qualifiedAssociation)}
    assert plans == {_node("plan-prompt-bridge-v1")}
    assert g.value(_node("plan-prompt-bridge-v1"), DCTERMS.identifier) == Literal(
        "sha256:" + "b" * 64)


def test_the_fixed_alignment_says_which_upload_it_came_from(tmp_path):
    inp = _inputs(tmp_path)
    fixed = tmp_path / "fija.ttl"
    fixed.write_text("fija", encoding="utf-8")
    inp.files["gdprtext_alignment"] = fixed
    g, _ = pb.build_graph(inp, _settings())
    assert not list(g.objects(_node("file-gdprtext-alignment"), PROV.wasDerivedFrom))
    inp.bridge = {**BRIDGE, "gdprtext_alignment": {"from": {"workspace": "gdprtext-ontology-1",
                                                            "version": 2}}}
    g, _ = pb.build_graph(inp, _settings())
    upload = _node("upload-gdprtext-ontology-1-v2")
    assert list(g.objects(_node("file-gdprtext-alignment"), PROV.wasDerivedFrom)) == [upload]
    assert str(g.value(upload, RDFS.label)) == "Versión 2 de la subida gdprtext-ontology-1"
    assert (upload, RDF.type, PROV.Entity) in g


def test_without_the_bridge_there_are_no_pairs(tmp_path):
    g, counts = pb.build_graph(_inputs(tmp_path, bridge=False), _settings())
    assert counts["bridge"] is None
    assert not any(pb.local(s).startswith("bridge-") for s in g.subjects())


# ------------------------------------------------------------------ the module

def test_the_module_header_and_no_blank_nodes(toy):
    _, g, _ = toy
    onto = URIRef(pb.PROVENANCE_IRI)
    assert (onto, RDF.type, OWL.Ontology) in g
    assert not list(g.objects(onto, OWL.imports))                     # by IRI, not imported
    assert URIRef("http://www.w3.org/ns/prov#") in set(g.objects(onto, DCTERMS.references))
    assert not [t for t in g if any(isinstance(x, BNode) for x in t)]
    assert all(o.language for o in g.objects(None, RDFS.label))
    people = list(g.subjects(RDF.type, PROV.Person))
    assert people == [_node("person-elian-gavilanes")]


def test_rebuilding_gives_the_same_graph(tmp_path):
    inp = _inputs(tmp_path)
    first, _ = pb.build_graph(inp, _settings())
    again, _ = pb.build_graph(inp, _settings())
    assert set(first) == set(again)
    from src.transform.convert import serialize
    assert serialize(first, "turtle") == serialize(again, "turtle")


def test_the_four_formats_hold_the_same_triples(toy, tmp_path):
    _, g, _ = toy
    files = pb.write_module(g, tmp_path / "out")
    assert [f.name for f in files] == ["ontopriv-provenance.rdf", "ontopriv-provenance.owl",
                                       "ontopriv-provenance.ttl", "ontopriv-provenance.jsonld"]
    assert pb.same_in_all_formats(files)
    from src.transform.convert import load_graph
    assert isomorphic(load_graph(files[3]), g)
    broken = load_graph(files[3])
    broken.remove((None, RDFS.comment, None))
    files[3].write_text(broken.serialize(format="json-ld"), encoding="utf-8")
    assert not pb.same_in_all_formats(files)


def test_a_base_out_of_step_with_the_log_is_refused(tmp_path):
    data, log = _log(tmp_path / "x")
    extra = alignment_graph(collect(data, read_log(log), SRC))
    extra.add((URIRef(ONTO + "Turnover"), SKOS.closeMatch, URIRef(DPV + "Fee")))
    with pytest.raises(ValueError, match="1 del módulo no están en el registro y 0 del registro"):
        pb.build_graph(_inputs(tmp_path / "a", alignment=extra), _settings())
    less = alignment_graph(collect(data, read_log(log), SRC))
    less.remove((URIRef(ONTO + "Consent"), SKOS.exactMatch, URIRef(DPV + "Consent")))
    with pytest.raises(ValueError, match="py -m src.alignment.write"):
        pb.build_graph(_inputs(tmp_path / "b", alignment=less), _settings())


def test_missing_finds_a_correspondence_without_who_or_when(toy):
    inp, g, _ = toy
    rec = _record(g, ONTO + "Consent", SKOS.exactMatch, DPV + "Consent")
    g.remove((rec, PROV.generatedAtTime, None))
    assert pb.missing(g, inp.alignment) == [(ONTO + "Consent", str(SKOS.exactMatch),
                                             DPV + "Consent")]
    g.add((rec, PROV.generatedAtTime, pm.when("2026-10-02T09:00:00+00:00")))
    g.remove((rec, PROV.wasAttributedTo, None))
    g.add((rec, PROV.wasAttributedTo, _node("agent-system")))           # not a person
    assert len(pb.missing(g, inp.alignment)) == 1


# ------------------------------------------------------------------ reading it

def test_what_the_provenance_says_of_a_concept(toy):
    _, g, _ = toy
    assert pb.find_term(g, "Consent") == [ONTO + "Consent", DPV + "Consent"]
    assert pb.find_term(g, ONTO + "Turnover") == [ONTO + "Turnover"]
    assert pb.find_term(g, "Nada") == []
    by_term = pm.records_about(g, DPV + "Consent")
    assert [pb.local(r).split("-")[0] for r in by_term] == ["mapping"]
    rec = _record(g, ONTO + "Consent", SKOS.exactMatch, DPV + "Consent")
    d = pm.describe(g, rec)
    assert (d["by"], d["at"], d["activity"]) == (
        "Elian Gavilanes", "2026-10-02T09:00:00+00:00",
        "Revisión de las correspondencias con el DPV")
    assert len(d["derived_from"]) == 1
    assert d["derived_from"][0].startswith("Candidatos del DPV y tipo SKOS propuesto por la IA (")
    assert d["notes"] == [] and d["sources"] == []
    text = pb.render_records(g, ONTO + "Consent")
    lines = text.splitlines()
    assert lines[0] == f"Procedencia de Consent ({ONTO}Consent)"
    assert lines[1].startswith("  · Consent skos:")                     # mappings first
    assert "    Revisión de las correspondencias con el DPV · Elian Gavilanes · " \
           "2026-10-02T09:00:00+00:00" in lines
    assert "  · Consent ↔ GivenConsent (por dpv:Consent)" in lines
    assert "    IA revisora: Consent ↔ GivenConsent, coherente (modelo-de-prueba): Dijo " \
           "«coherente» (exactMatch): Explicación. Solo explica: no cambia el par ni entra a " \
           "la ontología." in lines
    assert lines.index("  · Consent ↔ GivenConsent (por dpv:Consent)") > lines.index(
        "  · Propuesta del asistente para Consent")
    impact = pb.render_records(g, PROFILE_IRI + "#ImpactAssessment").splitlines()
    assert [x for x in impact if x.startswith("  · ")] == [
        "  · evaluación de impacto (ImpactAssessment)",
        "  · ImpactAssessment: concepto nuevo del perfil",
        "  · ImpactAssessment skos:exactMatch dpv:DPIA"]
    assert impact[2:5] == [
        "    Propuesto por la IA con base en el art. 42 de la LOPDP, como clase: Análisis de "
        "riesgos.", "    Extracción de conceptos de la LOPDP con IA · modelo-extractor",
        "    Fuente: LOPDP, art. 42"]
    assert "    Llevó a: ImpactAssessment: concepto nuevo del perfil" in impact
    assert "    Viene de: evaluación de impacto (ImpactAssessment)" in impact
    other = _node("other-1")                                   # a record of another kind
    g.add((other, RDFS.label, Literal("Otro registro", lang="es")))
    g.add((other, DCTERMS.subject, URIRef(ONTO + "Turnover")))
    turnover = pb.render_records(g, ONTO + "Turnover").splitlines()
    assert "    Fuente: LOPDP, art. 71; LOPDP, art. 72" in turnover
    assert [x for x in turnover if x.startswith("  · ")] == [
        "  · BusinessVolume: ya existe en OntoPriv (Turnover)",
        "  · Turnover: sin correspondencia en el DPV", "  · Otro registro"]


# ------------------------------------------------------------------ console

def _out(**kw):
    out = {"files": ["data/output/ontopriv-provenance.rdf"], "triples": 99,
           "same_in_all_formats": True, "missing": [],
           "counts": {"mappings": {"total": 3, "assistant": 1, "person": 2}, "no_match": 1,
                      "law_concepts": {"total": 3, "approved": 1, "duplicate": 1,
                                       "rejected": 1},
                      "repairs": {"R1": 1, "R2": 0}, "bridge": {"pairs": 1}},
           "people": [{"name": "Elian Gavilanes", "also_signed_as": ["Elian"]}],
           "validation": None}
    return {**out, **kw}


def test_render():
    text = pb.render(_out())
    assert text.splitlines() == [
        "Procedencia de la base (PROV-O, S10-T06)",
        "  Módulo: data/output/ontopriv-provenance.rdf (99 tripletas)",
        "  Los cuatro formatos dan las mismas tripletas.",
        "  Correspondencias con el DPV: 3 de 3 con quién y cuándo (1 confirmando una propuesta "
        "del asistente, 2 decididas por la persona sola)",
        "  Conceptos sin correspondencia: 1, con quién y cuándo",
        "  Conceptos que la IA tomó de la LOPDP: 3 (1 nuevo, 1 ya existía en OntoPriv, 1 "
        "rechazado), con su artículo y quién los decidió",
        "  Reparaciones: 1 R1 (varios dominios o rangos → una unión), 0 R2 (un solo tipo)",
        "  Puente con el GDPR: 1 par, cada uno con la explicación de la IA revisora",
        "  Persona: Elian Gavilanes (en el registro firmó también como «Elian»)",
        "  Forma y Lógica: no se revisaron"]
    other = pb.render(_out(same_in_all_formats=False, missing=[[ONTO + "A", str(SKOS.exactMatch),
                                                               DPV + "B"]],
                           people=[{"name": "Ana", "also_signed_as": []}],
                           counts={**_out()["counts"], "bridge": None}))
    assert "  Los cuatro formatos NO dan las mismas tripletas: revisa la escritura." in other
    assert "  Correspondencias con el DPV: 2 de 3 con quién y cuándo" in other
    plural = pb.render(_out(counts={**_out()["counts"], "mappings": {
        "total": 3, "assistant": 2, "person": 1}, "law_concepts": {
        "total": 5, "approved": 2, "duplicate": 2, "rejected": 1}}))
    assert "(2 confirmando una propuesta del asistente, 1 decidida por la persona sola)" in plural
    assert ": 5 (2 nuevos, 2 ya existían en OntoPriv, 1 rechazado)" in plural
    assert "Correcciones" not in plural
    for n, text in ((1, "1 correspondencia reemplazada; la anterior queda como versión previa"),
                    (2, "2 correspondencias reemplazadas; las anteriores quedan como versiones "
                        "previas")):
        fixed = pb.render(_out(counts={**_out()["counts"], "mappings": {
            "total": 3, "assistant": 1, "person": 2, "replaced": n}})).splitlines()
        assert fixed[4] == f"  Correcciones: {text}"
    assert "  Sin quién y cuándo: 1, por ejemplo A exactMatch B" in other
    assert "  Persona: Ana" in other.splitlines()
    assert "  Puente con el GDPR: no está construido (py -m src.alignment.gdpr_bridge)" in other


def test_the_cli(monkeypatch, capsys, tmp_path, toy):
    import src.config as config
    _, g, _ = toy
    calls = []
    monkeypatch.setattr(config, "load_config", lambda: {"outputs": {"dir": str(tmp_path)}})
    monkeypatch.setattr(pb, "build", lambda cfg, **kw: calls.append(kw) or _out())
    assert pb._run(["--motor", "owlrl"]) == 0
    assert calls == [{"engine": "owlrl"}]
    out = capsys.readouterr().out
    assert out.startswith("Procedencia de la base (PROV-O, S10-T06)")
    assert out.rstrip().endswith(pb.MANIFEST_FILE)
    monkeypatch.setattr(pb, "build", lambda cfg, **kw: (_ for _ in ()).throw(
        ValueError("La alineación de la base no coincide")))
    assert pb._run([]) == 1
    assert "Error: La alineación de la base no coincide" in capsys.readouterr().out
    assert pb._run(["--ver", "Consent"]) == 1
    assert "corre antes py -m src.provenance.base" in capsys.readouterr().out
    pb.write_module(g, tmp_path)
    assert pb._run(["--ver", "Turnover"]) == 0
    assert "  · Turnover: sin correspondencia en el DPV" in capsys.readouterr().out
    assert pb._run(["--ver", "Nada"]) == 1
    assert "La procedencia no dice nada de Nada." in capsys.readouterr().out


def test_settings_from_the_config():
    from src.config import load_config
    s = pb.settings(load_config())
    assert s.people == {"Elian Gavilanes": ["Elian"]}
    assert (s.extraction_model, s.extraction_prompt) == ("gemini-3.1-flash-lite", 1)
    assert pb.settings({}) == pb.Settings()


# ------------------------------------------------------------------ the real base

@needs_base
def test_the_provenance_of_the_base(tmp_path):
    out = pb.build(engine=ENGINE, out_dir=tmp_path)
    c = out["counts"]
    assert c["mappings"] == {"total": 190, "assistant": 159, "person": 31, "replaced": 1}
    assert c["no_match"] == 379
    assert c["law_concepts"] == {"total": 93, "approved": 25, "duplicate": 66, "rejected": 2}
    assert c["repairs"] == {"R1": 119, "R2": 34}
    assert c["bridge"]["pairs"] == 31
    assert out["missing"] == [] and out["same_in_all_formats"]
    assert out["people"] == [{"name": "Elian Gavilanes", "also_signed_as": ["Elian"]}]
    v = out["validation"]
    assert (v["shapes"]["new_violations"], v["shapes"]["new_warnings"],
            v["logic"]["new_errors"], v["logic"]["new_warnings"]) == (0, 0, 0, 0)
    assert gb.clean(v)
    saved = json.loads((tmp_path / pb.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert saved["counts"] == c and saved["inputs"]["log"]["file"] == \
        "data/review/alignment-decisions.jsonl"
    g = Graph().parse(tmp_path / f"{pb.MODULE}.ttl")
    alignment = Graph().parse(OUT / "ontopriv-dpv-alignment.rdf")
    triples = [(s, p, o) for p in pb.MAPPINGS for s, o in alignment.subject_objects(p)]
    assert len(triples) == 190 and pb.missing(g, alignment) == []
    signed = {d.reviewer for d in read_log(ROOT / "data" / "review" / DECISIONS_FILE)}
    assert {"Elian", "Elian Gavilanes"} <= signed
    assert list(g.subjects(RDF.type, PROV.Person)) == [_node("person-elian-gavilanes")]
    text = pb.render(out)
    assert "  Correspondencias con el DPV: 190 de 190 con quién y cuándo (159 confirmando una " \
           "propuesta del asistente, 31 decididas por la persona sola)" in text
    assert "  Sin hallazgos nuevos: la procedencia no rompe ninguna regla ni agrega " \
           "contradicciones." in text
    portability = pb.render_records(g, "http://www.semanticweb.org/franc/ontologies/2023/9/"
                                       "OntologiaLOPDP#Right_to_portability")
    assert "    Ya existía en OntoPriv como Right_to_portability; sus artículos de la LOPDP se " \
           "agregan a esa entidad. Decidido por Elian Gavilanes el 2026-10-01." in \
        portability.splitlines()
    registry = pb.render_records(g, PROFILE_IRI + "#NationalDataProtectionRegistry")
    heads = [x for x in registry.splitlines() if x.startswith("  · NationalDataProtection")]
    assert heads[-2:] == [                                     # corrected in S10-T16
        "  · NationalDataProtectionRegistry skos:broadMatch dpv:PublicRegisterOfEntities",
        "  · NationalDataProtectionRegistry skos:broadMatch dpv:ProtectionOfNationalSecurity "
        "(versión 1, ya no está en la base)"]
    assert ("    Reemplaza a: NationalDataProtectionRegistry skos:broadMatch "
            "dpv:ProtectionOfNationalSecurity (versión 1, ya no está en la base)") \
        in registry.splitlines()
    fixed = g.value(_node("file-gdprtext-alignment"), PROV.wasDerivedFrom)
    assert str(g.value(fixed, RDFS.label)) == "Versión 3 de la subida gdprtext-ontology-b863b1da"


@needs_base
def test_a_provenance_that_breaks_a_rule_is_seen(tmp_path):
    from src.config import load_config
    g = Graph()
    g.add((_node("x"), RDFS.label, Literal("sin idioma")))
    path = tmp_path / "procedencia.ttl"
    g.serialize(path, format="turtle")
    inp = pb.load_inputs(load_config())
    v = pb.validate(g, path, None, ENGINE, inp)
    assert v["shapes"]["new_violations"] + v["shapes"]["new_warnings"] >= 1
    assert not gb.clean(v)
    assert "  La procedencia trae hallazgos nuevos: revisa los reportes." in \
        gb.validation_lines(v, "la procedencia")


def test_the_bridge_keeps_its_own_texts():
    v = {"shapes": {"engine": {"version": "0"}, "new_violations": 0, "new_warnings": 0,
                    "inherited_violations": 0, "inherited_warnings": 0},
         "logic": {"engine": {"name": "owlrl", "version": "0", "reason": "prueba"},
                   "new_errors": 0, "new_warnings": 0, "inherited_errors": 0,
                   "inherited_warnings": 0}}
    assert gb.validation_lines(v)[-1] == ("  Sin hallazgos nuevos: el puente no rompe ninguna "
                                          "regla ni agrega contradicciones.")
    assert si.NEW == "nuevo"
