"""Tests for the PROV-O of each upload (S10-T07).

A hand-built upload (version 3: two changes of the person on top of the base and the rule, a
law with its concepts, labels, a definition and the AI reviewer) checks the records one by one;
the small pipeline of the tests checks that every upload writes the module, that it goes in
the .zip and that a change makes version 2 a revision of version 1. No AI is called."""
from __future__ import annotations

import io
import zipfile

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.alignment import decisions as dl
from src.alignment import user_alignment as ua
from src.alignment import user_decisions as ud
from src.provenance import base as pb
from src.provenance import model as pm
from src.provenance import upload as up
from src.provenance.model import FOAF, PROV
from src.transform.convert import load_graph
from tests.test_alignment_propose import env, targets  # noqa: F401
from tests.test_user_decisions import _apply, aligned, console  # noqa: F401

ONTO_IRI = "http://example.org/mi"
ONTO = ONTO_IRI + "#"
AL = ONTO_IRI + "-dpv-alignment"
D = "https://w3id.org/dpv#"
CORE = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
PEOPLE = {"Elian Gavilanes": ["Elian"]}
RULE = {"accept": ["alta"], "min_score": 0.5, "property_relations": [],
        "relations": ["skos:exactMatch", "skos:closeMatch", "skos:broadMatch",
                      "skos:narrowMatch"]}


def _entry(name, status, **kw):
    return {"key": ONTO + name, "name": name, "kinds": ["class"], "status": status,
            "decided_by": "rule", "dpv_iri": None, "relation": None, "score": None,
            "rank": None, "confidence": None, "justification": None, "version": 1, **kw}


def _record(key, decision, at, version, before, after, note, dpv="", relation=None):
    return dl.make_decision(
        source_id="upload:mi", target=dl.TARGET_MAPPING, concept_key=ONTO + key,
        decision=decision, reviewer=ud.REVIEWER, dpv_iri=dpv, relation=relation, note=note,
        decided_at=at, snapshot={"version": version, "before": before, "after": after})


def _was(dpv, relation, by, version):
    return {"dpv_iri": D + dpv, "dpv_name": dpv, "relation": relation, "decided_by": by,
            "version": version}


def _inputs(**kw) -> up.UploadInputs:
    """Version 3 of 'Mi ontología.ttl': in v2 the person turned PersonalData (exactMatch by the
    rule) into closeMatch; in v3 they took out the inherited Consent and aligned Pizza, which
    the rule had left without correspondence."""
    inherited = {"settings": {"duplicate_threshold": 0.75, "weight_lexical": 0.4,
                              "weight_semantic": 0.6},
                 "inherited": [
                     {"key": ONTO + "Consent", "name": "Consent", "method": "same_name",
                      "score": 1.0, "base_iri": CORE + "Consent", "base_name": "Consent",
                      "status": "aligned",
                      "mappings": [{"dpv_iri": D + "Consent", "relation": "skos:exactMatch",
                                    "reviewer": "Elian", "decided_at": "2026-10-07T17:00:00"
                                                                       "+00:00"}]},
                     {"key": ONTO + "Registry", "name": "Registry", "method": "similar",
                      "score": 0.7512, "base_iri": "http://x.org/p#NationalRegistry",
                      "base_name": "NationalRegistry", "status": "aligned",
                      "mappings": [{"dpv_iri": D + "ProtectionOfNationalSecurity",
                                    "relation": "skos:broadMatch",
                                    "reviewer": "Elian Gavilanes",
                                    "decided_at": "2026-10-02T01:41:22+00:00"}]},
                     {"key": ONTO + "Rule", "name": "Rule", "method": "same_iri", "score": 1.0,
                      "base_iri": CORE + "Rule", "base_name": "Rule", "status": "no_match",
                      "mappings": []}],
                 "not_inherited": []}
    decisions = {"rule": RULE, "threshold_without_ai": 0.75,
                 "prompt": {"name": "decide", "version": 1, "sha256": "d" * 64},
                 "summary": {"rule": "confianza alta de la IA y puntaje >= 0.5"},
                 "entries": [
                     _entry("PersonalData", "accepted", dpv_iri=D + "PersonalData",
                            relation="skos:exactMatch", score=0.81, rank=1, confidence="alta",
                            justification="Es el mismo concepto.", model="modelo-x",
                            precedents=["ai:DataSubject", CORE + "Personal_data"]),
                     _entry("Fee", "accepted", dpv_iri=D + "Fee", score=0.8,
                            decided_by="rule_without_ai"),
                     _entry("hasThing", "suggested", kinds=["property"],
                            dpv_iri=D + "hasRecipient", relation="skos:exactMatch", score=0.9,
                            confidence="alta"),
                     _entry("Thing", "suggested", dpv_iri=D + "Thing", score=0.4,
                            relation="skos:relatedMatch", confidence="alta"),
                     _entry("Pizza", "no_match", confidence="alta",
                            justification="Nada parecido.", model="modelo-x"),
                     _entry("Other", "no_proposal", confidence="media", rank=2, score=0.6,
                            justification="Quizá Policy."),
                     _entry("Lost", "undecided")]}
    records = [
        _record("PersonalData", "approved", "2026-11-02T10:00:00+00:00", 2,
                [_was("PersonalData", "skos:exactMatch", "rule", 1)],
                [_was("PersonalData", "skos:closeMatch", "user", 2)],
                "PersonalData closeMatch PersonalData", D + "PersonalData", "skos:closeMatch"),
        _record("Consent", "no_match", "2026-11-03T10:00:00+00:00", 3,
                [_was("Consent", "skos:exactMatch", "base", 1)], [], "Consent quitar"),
        _record("Pizza", "approved", "2026-11-03T10:00:00+00:00", 3, [],
                [_was("Food", "skos:exactMatch", "user", 3)],
                "¿«Pizza» es «Food»? → Sí", D + "Food", "skos:exactMatch")]
    values = dict(
        workspace="mi-1234", display="Mi ontología.ttl", mode="law", onto_iri=ONTO_IRI,
        alignment_iri=AL, version=3, created="2026-11-03T10:00:05+00:00",
        inherited=inherited, decisions=decisions,
        labels={"prompt": {"name": "labels", "version": 1, "sha256": "l" * 64},
                "ai": {"model": "modelo-x"},
                "entries": [{"key": ONTO + "Consent", "name": "Consent",
                             "added": {"es": "Consentimiento", "en": "Consent"}},
                            {"key": ONTO + "Pizza", "name": "Pizza", "added": {}}]},
        labels_iri=ONTO_IRI + "-labels",
        definitions={"prompt": {"name": "define", "version": 1, "sha256": "f" * 64},
                     "ai": {}, "law": {"file": "ley.txt", "language": "es"},
                     "entries": [{"key": ONTO + "Thing", "name": "Thing",
                                  "definition": "Una cosa de la ley.", "article": 3,
                                  "articles": [3], "status": "articles"}]},
        definitions_iri=ONTO_IRI + "-definitions",
        audit={"prompt": {"name": "audit", "version": 1},
               "created": "2026-11-01T00:05:00+00:00",
               "entries": [{"key": ONTO + "PersonalData", "name": "PersonalData",
                            "dpv_iri": D + "PersonalData", "verdict": "coherente",
                            "ai_relation": "exactMatch", "explanation": "Lo mismo."},
                           {"key": ONTO + "Fee", "name": "Fee", "dpv_iri": D + "Fee",
                            "verdict": None}]},                        # not reviewed yet
        records=records,
        law_concepts={"prompt": {"name": "extract", "version": 2, "sha256": "e" * 64},
                      "rule": "cita un artículo de la ley, está bien formado y no se repite",
                      "extraction": {"model": "modelo-x", "articles_processed": 3,
                                     "concepts": 3},
                      "entries": [{"name": "Consent", "label": "consentimiento",
                                   "kind": "class", "articles": [2, 1], "status": "accepted",
                                   "definition": "Manifestación de voluntad.",
                                   "joined": ["ConsentAgain"]},
                                  {"name": "ConsentAgain", "kind": "class", "articles": [5],
                                   "status": "repeated", "repeated_of": "Consent"},
                                  {"name": "Bad", "kind": "property", "articles": [],
                                   "status": "rejected",
                                   "reasons": ["no cita un artículo de la ley"]}]},
        law_profile={"entries": [{"name": "Consent", "status": "reused",
                                  "core_iri": CORE + "Consent", "iri": CORE + "Consent"}]},
        inputs={"ontology": {"original_name": "Perfil de ley.txt", "stored_as": "ley-profile.ttl",
                             "generated_from": "law", "content_sha256": "a" * 64},
                "law": {"original_name": "ley.txt", "sha256": "b" * 64}},
        steps={"extract": {"at": "2026-11-01T00:00:00+00:00"},
               "accept": {"at": "2026-11-01T00:00:01+00:00"},
               "profile": {"at": "2026-11-01T00:00:02+00:00"},
               "inherit": {"at": "2026-11-01T00:00:03+00:00"},
               "propose": {"at": "2026-11-01T00:00:04+00:00"},
               "labels": {"at": "2026-11-01T00:00:05+00:00"},
               "definitions": {"at": "2026-11-01T00:00:06+00:00"}},
        versions=[{"version": 1, "folder": "history/v1", "until": "2026-11-02T10:00:00+00:00"},
                  {"version": 2, "folder": "history/v2", "until": "2026-11-03T10:00:00+00:00"}],
        history={1: "2026-11-01T00:10:00+00:00", 2: "2026-11-02T10:00:05+00:00"},
        embedding_model="modelo-embeddings")
    return up.UploadInputs(**{**values, **kw})


@pytest.fixture
def toy():
    g, counts = up.build_graph(_inputs(), PEOPLE)
    return g, counts


def _node(local):
    return URIRef(up.provenance_iri(ONTO_IRI) + "#" + local)


def _correspondence(name, dpv, relation, by, version):
    return _node("correspondence-" + pm.digest(AL, ONTO + name, str(relation), D + dpv, by,
                                               str(version)))


def _current(g):
    """The triples the records say are in the ontology now."""
    return {(s, g.value(r, OWL.annotatedProperty), g.value(r, OWL.annotatedTarget))
            for r, s in g.subject_objects(OWL.annotatedSource)}


# ------------------------------------------------------------------ correspondences

def test_each_correspondence_in_the_ontology_says_who_decided_it(toy):
    g, counts = toy
    assert counts["correspondences"] == {
        "current": {"base": 1, "regla sin IA": 1, "usuario": 2}, "total": 4, "replaced": 2}
    assert _current(g) == {
        (URIRef(ONTO + "Registry"), SKOS.broadMatch, URIRef(D + "ProtectionOfNationalSecurity")),
        (URIRef(ONTO + "Fee"), RDFS.seeAlso, URIRef(D + "Fee")),
        (URIRef(ONTO + "PersonalData"), SKOS.closeMatch, URIRef(D + "PersonalData")),
        (URIRef(ONTO + "Pizza"), SKOS.exactMatch, URIRef(D + "Food"))}
    alignment = Graph()
    for s, p, o in _current(g):
        alignment.add((s, p, o))
    assert up.missing(g, alignment) == []
    alignment.add((URIRef(ONTO + "Fee"), RDFS.seeAlso, URIRef("http://example.org/doc")))
    assert up.missing(g, alignment) == []                       # not a correspondence
    alignment.add((URIRef(ONTO + "Lost"), SKOS.exactMatch, URIRef(D + "Policy")))
    assert up.missing(g, alignment) == [(ONTO + "Lost", str(SKOS.exactMatch), D + "Policy")]
    registry = _correspondence("Registry", "ProtectionOfNationalSecurity", SKOS.broadMatch,
                               "base", 1)
    g.remove((registry, _node("decidedBy"), None))
    assert len(up.missing(g, alignment)) == 2                   # says no longer who decided
    by = _node("decidedBy")
    fee = _correspondence("Fee", "Fee", RDFS.seeAlso, "rule_without_ai", 1)
    assert g.value(fee, by) == Literal("regla sin IA", lang="es")
    assert str(g.value(fee, RDFS.comment)) == (
        "Aceptada sin IA: el mejor candidato tenía puntaje 0.8 (mínimo 0.75); sin tipo SKOS "
        "(rdfs:seeAlso).")
    assert g.value(fee, PROV.wasAttributedTo) == _node("agent-system")
    assert g.value(fee, PROV.generatedAtTime) == pm.when("2026-11-01T00:00:04+00:00")
    assert g.value(fee, DCTERMS.isPartOf) == _node("version-3")


def test_an_inherited_one_points_to_its_record_in_the_base(toy):
    g, counts = toy
    assert counts["inherited"] == {"mappings": 2, "no_match": 1}
    registry = _correspondence("Registry", "ProtectionOfNationalSecurity", SKOS.broadMatch,
                               "base", 1)
    assert (registry, RDF.type, OWL.Axiom) in g
    assert g.value(registry, _node("decidedBy")) == Literal("base", lang="es")
    assert g.value(registry, PROV.wasDerivedFrom) == pb.mapping_record(
        "http://x.org/p#NationalRegistry", SKOS.broadMatch, D + "ProtectionOfNationalSecurity")
    assert str(pb.mapping_record("a", SKOS.exactMatch, "b")).startswith(
        pb.PROVENANCE_IRI + "#mapping-")
    elian = _node("person-elian-gavilanes")
    assert g.value(registry, PROV.wasAttributedTo) == elian
    consent = _correspondence("Consent", "Consent", SKOS.exactMatch, "base", 1)
    assert g.value(consent, PROV.wasAttributedTo) == elian                # signed "Elian"
    assert set(g.objects(elian, FOAF.nick)) == {Literal("Elian")}
    assert str(g.value(registry, RDFS.comment)) == (
        "Heredada de NationalRegistry de la base (muy parecido, puntaje 0.7512): la decidió "
        "Elian Gavilanes el 2026-10-02 en la base.")
    [rule] = [r for r in pm.records_about(g, ONTO + "Rule")]
    assert pb.local(rule).startswith("inherited-no-match-")
    assert g.value(rule, PROV.wasDerivedFrom) == pb.no_match_record(CORE + "Rule")
    assert str(g.value(rule, RDFS.label)) == "Rule: sin correspondencia (heredado de la base)"


def test_the_rule_says_what_the_ai_proposed(toy):
    g, counts = toy
    assert counts["decider"] == {"accepted": 2, "no_match": 1, "no_proposal": 1,
                                 "suggested": 2, "undecided": 1}
    old = _correspondence("PersonalData", "PersonalData", SKOS.exactMatch, "rule", 1)
    assert g.value(old, _node("decidedBy")) == Literal("regla", lang="es")
    assert str(g.value(old, RDFS.comment)) == (
        "Aceptada automáticamente por la regla (confianza alta de la IA y puntaje >= 0.5). La "
        "IA (modelo-x) respondió con confianza alta: «Es el mismo concepto.». Candidato n.º 1 "
        "del DPV (puntaje 0.81). Precedentes de la base que vio la IA: DataSubject, "
        "Personal_data.")
    deciding = _node("activity-decider")
    plans = {g.value(a, PROV.hadPlan) for a in g.objects(deciding, PROV.qualifiedAssociation)}
    assert plans == {_node("plan-decider-rule"), _node("plan-prompt-decide-v1")}
    assert set(g.objects(deciding, PROV.wasAssociatedWith)) == {
        _node("agent-system"), _node("agent-embeddings-modelo-embeddings"),
        _node("agent-llm-modelo-x")}

    def said(name):
        [r] = [r for r in pm.records_about(g, ONTO + name) if pb.local(r).startswith("decider-")]
        return str(g.value(r, RDFS.label)), str(g.value(r, RDFS.comment))
    assert said("hasThing") == (
        "hasThing: sugerida (no entra a la ontología)",
        "La IA propuso skos:exactMatch dpv:hasRecipient con confianza alta, pero la regla no "
        "la acepta (las propiedades nunca son automáticas): no entra a la ontología.")
    assert said("Thing")[1].startswith(
        "La IA propuso skos:relatedMatch dpv:Thing con confianza alta, pero la regla no la "
        "acepta (no acepta skos:relatedMatch; puntaje 0.4 < 0.5)")
    assert said("Other") == ("Other: sin propuesta",
                             "La regla no aceptó ni sugirió nada: la IA no respondió con la "
                             "confianza que pide. La IA respondió con confianza media: «Quizá "
                             "Policy.».")
    assert said("Lost")[1] == ("La IA no estuvo disponible y la regla sin IA no encontró un "
                               "candidato con puntaje ≥ 0.75.")
    assert said("Pizza")[1] == ("La IA dijo que ningún candidato del DPV corresponde. La IA "
                                "(modelo-x) respondió con confianza alta: «Nada parecido.».")


# ------------------------------------------------------------------ versions

def test_a_change_of_the_person_revises_what_was_there(toy):
    g, counts = toy
    assert counts["user"] == {"approved": 2, "removed": 1} and counts["versions"] == 3
    old = _correspondence("PersonalData", "PersonalData", SKOS.exactMatch, "rule", 1)
    new = _correspondence("PersonalData", "PersonalData", SKOS.closeMatch, "user", 2)
    assert (new, PROV.wasRevisionOf, old) in g
    assert (new, RDF.type, OWL.Axiom) in g and (old, RDF.type, OWL.Axiom) not in g
    assert (old, RDF.type, RDF.Statement) in g                      # described, not asserted
    assert (g.value(old, RDF.subject), g.value(old, RDF.predicate), g.value(old, RDF.object)) == (
        URIRef(ONTO + "PersonalData"), SKOS.exactMatch, URIRef(D + "PersonalData"))
    assert g.value(old, PROV.wasInvalidatedBy) == _node("activity-user-v2")
    assert g.value(old, PROV.invalidatedAtTime) == pm.when("2026-11-02T10:00:00+00:00")
    assert g.value(new, PROV.wasAttributedTo) == _node("person-usuario")
    assert g.value(new, _node("decidedBy")) == Literal("usuario", lang="es")
    assert g.value(new, OWL.versionInfo) == Literal("2")
    assert str(g.value(new, RDFS.comment)) == (
        "Decidida por el usuario (versión 2). Antes: exactMatch PersonalData (regla, versión 1)."
        " Pedido: PersonalData closeMatch PersonalData")
    consent = _correspondence("Consent", "Consent", SKOS.exactMatch, "base", 1)
    [removal] = [r for r in pm.records_about(g, ONTO + "Consent")
                 if pb.local(r).startswith("user-removal-")]
    assert (removal, PROV.wasRevisionOf, consent) in g
    assert g.value(consent, PROV.wasInvalidatedBy) == _node("activity-user-v3")
    assert str(g.value(removal, RDFS.label)) == "Consent: quitada por el usuario (versión 3)"
    assert str(g.value(removal, RDFS.comment)) == (
        "El usuario quitó exactMatch Consent (base, versión 1). Pedido: Consent quitar")
    pizza = _correspondence("Pizza", "Food", SKOS.exactMatch, "user", 3)
    [before] = [r for r in pm.records_about(g, ONTO + "Pizza")
                if pb.local(r).startswith("decider-")]
    assert (pizza, PROV.wasRevisionOf, before) in g
    assert g.value(before, PROV.wasInvalidatedBy) == _node("activity-user-v3")
    other = [r for r in pm.records_about(g, ONTO + "Other")]
    assert all(g.value(r, PROV.wasInvalidatedBy) is None for r in other)


def test_the_versions_and_the_person(toy):
    g, _ = toy
    v1, v2, v3 = (_node(f"version-{v}") for v in (1, 2, 3))
    assert (v3, PROV.wasRevisionOf, v2) in g and (v2, PROV.wasRevisionOf, v1) in g
    assert g.value(v1, PROV.wasRevisionOf) is None
    assert g.value(v1, PROV.wasGeneratedBy) == _node("activity-alignment")
    assert g.value(v2, PROV.wasGeneratedBy) == _node("activity-user-v2")
    assert g.value(v2, PROV.generatedAtTime) == pm.when("2026-11-02T10:00:05+00:00")
    assert g.value(v2, PROV.invalidatedAtTime) == pm.when("2026-11-03T10:00:00+00:00")
    assert g.value(v3, PROV.generatedAtTime) == pm.when("2026-11-03T10:00:05+00:00")
    assert g.value(v3, PROV.invalidatedAtTime) is None
    assert all(g.value(v, PROV.specializationOf) == URIRef(AL) for v in (v1, v2, v3))
    user = _node("activity-user-v3")
    assert g.value(user, PROV.startedAtTime) == pm.when("2026-11-03T10:00:00+00:00")
    assert set(g.objects(user, PROV.wasAssociatedWith)) == {_node("person-usuario")}
    people = set(g.subjects(RDF.type, PROV.Person))
    assert people == {_node("person-usuario"), _node("person-elian-gavilanes")}
    onto = URIRef(up.provenance_iri(ONTO_IRI))
    assert g.value(onto, OWL.versionInfo) == Literal("3")
    assert not list(g.objects(onto, OWL.imports))


# ------------------------------------------------------------------ what the AI added

def test_the_law_its_articles_and_the_rule(toy):
    g, counts = toy
    assert counts["law_concepts"] == {"total": 3, "accepted": 1, "rejected": 1, "repeated": 1}
    extracted = _node("law-concept-" + pm.digest(ONTO_IRI, "Consent"))
    assert set(g.objects(extracted, PROV.hadPrimarySource)) == {_node("law-art-1"),
                                                                _node("law-art-2")}
    assert g.value(_node("law-art-2"), RDFS.label) == Literal("ley.txt, art. 2", lang="es")
    assert g.value(_node("law-art-2"), DCTERMS.isPartOf) == _node("file-law")
    assert str(g.value(extracted, RDFS.comment)) == (
        "Propuesto por la IA con base en los arts. 1 y 2 de ley.txt, como clase: "
        "Manifestación de voluntad.")
    assert g.value(extracted, PROV.wasAttributedTo) == _node("agent-llm-modelo-x")
    accepted = _node("law-acceptance-" + pm.digest(ONTO_IRI, "Consent"))
    assert g.value(accepted, DCTERMS.subject) == URIRef(CORE + "Consent")
    assert str(g.value(accepted, RDFS.comment)) == (
        "Aceptado por la regla: cita su artículo, está bien formado y no se repite. Se le "
        "unieron: ConsentAgain. Reutiliza Consent de OntoPriv-Core (el mismo IRI).")
    bad = _node("law-acceptance-" + pm.digest(ONTO_IRI, "Bad"))
    assert str(g.value(bad, RDFS.comment)) == (
        "Rechazado por la regla: no cita un artículo de la ley.")
    assert g.value(_node("law-concept-" + pm.digest(ONTO_IRI, "Bad")),
                   PROV.wasInvalidatedBy) == _node("activity-law-acceptance")
    again = _node("law-acceptance-" + pm.digest(ONTO_IRI, "ConsentAgain"))
    assert str(g.value(again, RDFS.label)) == "ConsentAgain: repetido"
    assert str(g.value(again, RDFS.comment)) == (
        "Repetido: se une a Consent, que conserva sus artículos; no se escribe aparte.")
    assert g.value(_node("file-ontology"), RDFS.label) == Literal(
        "Perfil de la ley (ley-profile.ttl)", lang="es")
    assert (_node("file-ontology"), PROV.wasGeneratedBy, _node("activity-law-profile")) in g
    assert str(g.value(_node("activity-law-extraction"), RDFS.comment)).endswith(
        "3 artículos procesados, 3 conceptos propuestos.")


def test_labels_definitions_and_the_reviewer(toy):
    g, counts = toy
    assert (counts["labels"], counts["definitions"], counts["reviews"]) == (1, 1, 1)
    [labels] = [r for r in pm.records_about(g, ONTO + "Consent")
                if pb.local(r).startswith("labels-")]
    assert str(g.value(labels, RDFS.comment)) == (
        "La IA agregó «Consent» (en), «Consentimiento» (es). Tus etiquetas no cambian.")
    assert g.value(labels, DCTERMS.isPartOf) == _node("module-labels")
    assert g.value(_node("module-labels"), PROV.specializationOf) == URIRef(ONTO_IRI + "-labels")
    assert not [r for r in pm.records_about(g, ONTO + "Pizza")
                if pb.local(r).startswith("labels-")]                  # nothing added
    [defined] = [r for r in pm.records_about(g, ONTO + "Thing")
                 if pb.local(r).startswith("definition-")]
    assert g.value(defined, PROV.hadPrimarySource) == _node("law-art-3")
    assert str(g.value(defined, RDFS.comment)) == (
        "ley.txt, art. 3 · escrita por la IA: «Una cosa de la ley.».")
    assert g.value(defined, PROV.wasAttributedTo) == _node("agent-llm")   # model not recorded
    old = _correspondence("PersonalData", "PersonalData", SKOS.exactMatch, "rule", 1)
    [note] = list(g.subjects(DCTERMS.subject, old))
    assert str(g.value(note, RDFS.comment)) == (
        "Dijo «coherente» (exactMatch): Lo mismo. Solo explica: no cambia lo aceptado.")
    assert g.value(note, PROV.generatedAtTime) == pm.when("2026-11-01T00:05:00+00:00")


def test_without_a_law_labels_definitions_or_review(tmp_path):
    g, counts = up.build_graph(_inputs(law_concepts=None, law_profile=None, labels=None,
                                       definitions=None, audit=None, records=[], version=1,
                                       versions=[], history={}, inputs={
                                           "ontology": {"original_name": "Mi ontología.ttl",
                                                        "sha256": "a" * 64}}), PEOPLE)
    assert g.value(_node("file-ontology"), RDFS.label) == Literal(
        "Tu ontología (Mi ontología.ttl)", lang="es")
    assert g.value(_node("file-ontology"), DCTERMS.identifier) == Literal("sha256:" + "a" * 64)
    assert (_node("file-law"), None, None) not in g
    assert (counts["law_concepts"], counts["labels"], counts["definitions"],
            counts["reviews"], counts["user"]) == (None, 0, 0, 0, {})
    assert counts["correspondences"]["replaced"] == 0
    assert not list(g.subjects(RDF.type, RDF.Statement))
    assert (None, None, _node("activity-user-v2")) not in g


def test_only_what_the_change_took_out_is_replaced():
    """Version 4: Consent kept its exactMatch when the person replaced ConsentRecord (v2);
    Other went exactMatch (v2) -> closeMatch (v3) -> exactMatch again (v4): the record of v2
    is not the one in force, the one of v4 is."""
    inherited = _inputs().inherited
    consent = inherited["inherited"][0]
    consent["mappings"].append({"dpv_iri": D + "ConsentRecord", "relation": "skos:relatedMatch",
                                "reviewer": "Elian", "decided_at": "2026-10-07T17:01:00+00:00"})
    kept = _was("Consent", "skos:exactMatch", "base", 1)
    records = [
        _record("Consent", "rejected", "2026-11-02T10:00:00+00:00", 2,
                [kept, _was("ConsentRecord", "skos:relatedMatch", "base", 1)],
                [kept, _was("Policy", "skos:closeMatch", "user", 2)], "x", D + "ConsentRecord"),
        _record("Consent", "approved", "2026-11-02T10:00:00+00:00", 2,
                [kept, _was("ConsentRecord", "skos:relatedMatch", "base", 1)],
                [kept, _was("Policy", "skos:closeMatch", "user", 2)], "x", D + "Policy",
                "skos:closeMatch"),
        _record("Other", "approved", "2026-11-02T10:00:00+00:00", 2, [],
                [_was("Policy", "skos:exactMatch", "user", 2)], "y", D + "Policy",
                "skos:exactMatch"),
        _record("Other", "approved", "2026-11-03T10:00:00+00:00", 3,
                [_was("Policy", "skos:exactMatch", "user", 2)],
                [_was("Policy", "skos:closeMatch", "user", 3)], "y", D + "Policy",
                "skos:closeMatch"),
        _record("Other", "approved", "2026-11-04T10:00:00+00:00", 4,
                [_was("Policy", "skos:closeMatch", "user", 3)],
                [_was("Policy", "skos:exactMatch", "user", 4)], "y", D + "Policy",
                "skos:exactMatch")]
    g, counts = up.build_graph(_inputs(inherited=inherited, records=records, version=4,
                                       versions=[], history={}), PEOPLE)
    still = _correspondence("Consent", "Consent", SKOS.exactMatch, "base", 1)
    assert (still, RDF.type, OWL.Axiom) in g and g.value(still, PROV.wasInvalidatedBy) is None
    gone = _correspondence("Consent", "ConsentRecord", SKOS.relatedMatch, "base", 1)
    assert g.value(gone, PROV.wasInvalidatedBy) == _node("activity-user-v2")
    new = _correspondence("Consent", "Policy", SKOS.closeMatch, "user", 2)
    assert set(g.objects(new, PROV.wasRevisionOf)) == {gone}
    v2 = _correspondence("Other", "Policy", SKOS.exactMatch, "user", 2)
    v4 = _correspondence("Other", "Policy", SKOS.exactMatch, "user", 4)
    assert (v2, RDF.type, RDF.Statement) in g and (v2, RDF.type, OWL.Axiom) not in g
    assert (v4, RDF.type, OWL.Axiom) in g
    assert counts["correspondences"]["current"]["usuario"] == 2


def _removed_and_aligned_again() -> up.UploadInputs:
    """Version 3: Thing (a suggestion) got exactMatch in v2 and one more in v3; Consent was
    taken out in v2 and aligned again in v3."""
    thing = _was("Thing", "skos:exactMatch", "user", 2)
    again = _was("Policy", "skos:closeMatch", "user", 3)
    records = [
        _record("Thing", "approved", "2026-11-02T10:00:00+00:00", 2, [], [thing], "a",
                D + "Thing", "skos:exactMatch"),
        _record("Consent", "no_match", "2026-11-02T10:00:00+00:00", 2,
                [_was("Consent", "skos:exactMatch", "base", 1)], [], "quitar"),
        _record("Thing", "approved", "2026-11-03T10:00:00+00:00", 3, [thing],
                [thing, _was("Policy", "skos:relatedMatch", "user", 3)], "b", D + "Policy",
                "skos:relatedMatch"),
        _record("Consent", "approved", "2026-11-03T10:00:00+00:00", 3, [], [again], "c",
                D + "Policy", "skos:closeMatch")]
    return _inputs(records=records)


def test_why_there_was_none_is_replaced_once_and_a_removal_can_be_revised():
    g, _ = up.build_graph(_removed_and_aligned_again(), PEOPLE)
    status = _node("decider-" + pm.digest(AL, ONTO + "Thing"))
    assert list(g.objects(status, PROV.invalidatedAtTime)) == [
        Literal("2026-11-02T10:00:00+00:00", datatype=XSD.dateTime)]
    first = _correspondence("Thing", "Thing", SKOS.exactMatch, "user", 2)
    second = _correspondence("Thing", "Policy", SKOS.relatedMatch, "user", 3)
    assert set(g.objects(first, PROV.wasRevisionOf)) == {status}
    assert not list(g.objects(second, PROV.wasRevisionOf))
    removal = _node("user-removal-" + pm.digest(AL, ONTO + "Consent", "2"))
    assert g.value(removal, PROV.wasInvalidatedBy) == _node("activity-user-v3")
    back = _correspondence("Consent", "Policy", SKOS.closeMatch, "user", 3)
    assert set(g.objects(back, PROV.wasRevisionOf)) == {removal}
    assert (back, RDF.type, OWL.Axiom) in g
    rows = up.records_of(g, "Consent")
    assert [r["current"] for r in rows if r["decidedBy"]] == ["false", "true"]


def test_the_graph_is_stable_and_has_no_blank_nodes(toy):
    g, _ = toy
    again, _ = up.build_graph(_inputs(), PEOPLE)
    assert set(g) == set(again)
    assert not [t for t in g if any(isinstance(x, BNode) for x in t)]
    assert all(o.language for o in g.objects(None, RDFS.label))
    assert all(o.datatype == XSD.dateTime for p in (PROV.generatedAtTime, PROV.endedAtTime,
                                                     PROV.invalidatedAtTime)
               for o in g.objects(None, p))


# ------------------------------------------------------------------ the fixed query

def test_the_fixed_query_gives_the_records_of_a_concept(toy):
    g, _ = toy
    text = up.query_text()
    assert "?name" in text and "SELECT" in text
    rows = up.records_of(g, "PersonalData")
    assert rows == up.records_of(g, ONTO + "PersonalData")
    assert [r["label"] for r in rows] == [
        "PersonalData skos:exactMatch dpv:PersonalData (regla, versión 1)",
        "PersonalData skos:closeMatch dpv:PersonalData (usuario, versión 2)"]
    old, new = rows
    assert (old["decidedBy"], old["replaced"], old["revises"]) == (
        "regla", "2026-11-02T10:00:00+00:00", None)
    assert old["notes"].startswith("IA revisora: PersonalData dpv:PersonalData, coherente: ")
    assert (new["decidedBy"], new["who"], new["version"], new["replaced"]) == (
        "usuario", "usuario", "2", None)
    assert new["revises"] == "PersonalData skos:exactMatch dpv:PersonalData (regla, versión 1)"
    assert up.records_of(g, "Nada") == []
    consent = up.records_of(g, "Consent")
    assert [r["label"] for r in consent][-1] == "Consent: quitada por el usuario (versión 3)"
    assert set(consent[0]["sources"].split("; ")) == {"ley.txt, art. 1", "ley.txt, art. 2"}


def test_render_records(toy):
    g, _ = toy
    lines = up.render_records(up.records_of(g, "PersonalData"), "PersonalData").splitlines()
    assert lines[0] == f"Procedencia de PersonalData ({ONTO}PersonalData)"
    assert lines[1] == ("  · PersonalData skos:exactMatch dpv:PersonalData (regla, versión 1) "
                        "— reemplazada el 2026-11-02 10:00 UTC")
    assert "    Decidida por: regla (ontology-integration-transformer)" in lines
    assert "    Decisor: candidatos del DPV, IA y regla · 2026-11-01 00:00 UTC" in lines
    assert "    Decidida por: usuario" in lines
    assert ("    Revisa: PersonalData skos:exactMatch dpv:PersonalData (regla, versión 1)"
            in lines)
    consent = up.render_records(up.records_of(g, "Consent"), "Consent").splitlines()
    assert "    Extracción de conceptos de ley.txt con IA · modelo-x" in consent
    assert up.render_records([], "Nada") == "La procedencia no dice nada de Nada."


def test_the_console_line():
    entry = {"file": "output/mi-provenance.ttl", "triples": 99,
             "counts": {"correspondences": {"current": {"base": 2, "usuario": 1}, "total": 3,
                                            "replaced": 1}}}
    assert ua.provenance_line(entry) == (
        "Procedencia (PROV-O): output/mi-provenance.ttl (99 tripletas): quién decidió cada una "
        "de las 3 correspondencias (2 base, 1 usuario); 1 reemplazada en versiones anteriores")
    entry["without_who"] = 1
    assert ua.provenance_line(entry).endswith(
        "; 1 reemplazada en versiones anteriores; OJO: 1 correspondencia del módulo sin quién la "
        "decidió")
    entry["counts"]["correspondences"].update(current={}, total=0, replaced=0)
    entry["without_who"] = 0
    assert ua.provenance_line(entry).endswith("cada una de las 0 correspondencias (ninguna)")


# ------------------------------------------------------------------ in every upload

def test_every_upload_writes_its_provenance_and_the_zip_carries_it(aligned):  # noqa: F811
    m = ua.load_manifest(aligned)
    entry = m["provenance"]
    assert entry["iri"] == up.provenance_iri(m["ontology_iri"])
    assert entry["file"] == "output/mi-gdpr-provenance.ttl"
    assert "provenance" not in m["files"] and "provenance" not in m["modules"]  # not validated
    g = load_graph(aligned.path / entry["file"])
    assert entry["triples"] == len(g) and entry["without_who"] == 0
    assert entry["counts"]["correspondences"] == {"current": {"base": 2, "regla": 1},
                                                  "total": 3, "replaced": 0}
    assert up.missing(g, load_graph(aligned.path / m["files"]["alignment"])) == []
    data = ua.build_user_package(aligned, "turtle").data
    z = zipfile.ZipFile(io.BytesIO(data))
    assert "mi-gdpr-provenance.ttl" in z.namelist()
    readme = z.read("LEEME.txt").decode("utf-8")
    assert ("  - mi-gdpr-provenance.ttl: la procedencia (PROV-O): quién decidió cada "
            "correspondencia (base, regla o usuario), cómo se produjo y sus versiones. Ningún "
            "módulo la importa: ábrela aparte.\n") in readme
    catalog = z.read("catalog-v001.xml").decode("utf-8")
    assert f'name="{entry["iri"]}" uri="mi-gdpr-provenance.ttl"' in catalog
    assert ua.provenance_line(entry) in ua.render_console(aligned, m)


def test_a_change_makes_version_2_a_revision_of_version_1(aligned, env, targets):  # noqa: F811
    _apply(aligned, env, targets, "PersonalData broadMatch PersonalData")
    m = ua.load_manifest(aligned)
    assert m["provenance"]["counts"]["correspondences"] == {
        "current": {"base": 2, "usuario": 1}, "total": 3, "replaced": 1}
    g = load_graph(aligned.path / m["provenance"]["file"])
    key = URIRef(m["ontology_iri"].rstrip("#/") + "#PersonalData")
    rows = up.records_of(g, str(key))
    old_row, new_row = [r for r in rows if r["decidedBy"]]
    assert (old_row["decidedBy"], new_row["decidedBy"]) == ("regla", "usuario")
    assert new_row["revises"] == old_row["label"] and old_row["replaced"]
    assert rows[-1] == new_row                                    # the latest version last
    old = next(r for r in g.subjects(RDF.subject, key))
    assert (old, RDF.type, OWL.Axiom) not in g
    v1 = next((aligned.path / ud.HISTORY_DIR / "v1").glob("*-turtle.zip"))
    z = zipfile.ZipFile(v1)
    before = Graph().parse(data=z.read("mi-gdpr-provenance.ttl"), format="turtle")
    assert (URIRef(str(old)), RDF.type, OWL.Axiom) in before          # in force in v1
    assert (URIRef(str(old)), PROV.wasInvalidatedBy, None) not in before
    assert (_version(m, 2), PROV.wasRevisionOf, _version(m, 1)) in g


def _version(m, v):
    return URIRef(m["provenance"]["iri"] + f"#version-{v}")


def test_the_cli(aligned, console, capsys):  # noqa: F811
    assert up._run(["--espacio", aligned.id, "--ver", "PersonalData"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Procedencia de PersonalData (")
    assert "    Decidida por: regla (ontology-integration-transformer)" in out
    assert up._run(["--espacio", aligned.id, "--ver", "Nada"]) == 1
    assert capsys.readouterr().out == "La procedencia no dice nada de Nada.\n"
    assert up._run(["--espacio", "no-existe", "--ver", "X"]) == 1
    assert capsys.readouterr().out.startswith("Error: ")
    import json
    path = aligned.output_dir / ua.MANIFEST_FILE
    m = json.loads(path.read_text(encoding="utf-8"))
    m.pop("provenance")
    path.write_text(json.dumps(m), encoding="utf-8")
    assert up._run(["--espacio", aligned.id, "--ver", "PersonalData"]) == 1
    assert ("todavía no tiene procedencia; vuelve a correrla: py -m "
            f"src.alignment.user_alignment --espacio {aligned.id}") in capsys.readouterr().out
