"""S4-T02: the single list of concepts to align (OntoPriv entities + AI proposals)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import rdflib

from src.core.inventory import build_inventory
from src.alignment.sources import (
    build_alignment_sources, load_ai_concepts, is_borrowed,
    ORIGIN_ONTOLOGY, ORIGIN_AI, ALIGN_CLASS, ALIGN_PROPERTY, ALIGN_INDIVIDUAL,
)

ROOT = Path(__file__).resolve().parents[1]

_TTL = """
@prefix : <http://example.org/lopdp#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

:Law a owl:Class .
:Rights a owl:Class ; rdfs:subClassOf :Law .
:DataSubject a owl:Class ; rdfs:subClassOf :Rights ;
    rdfs:label "Data subject" ; rdfs:comment "Person whose data is processed." .
:Consent a owl:Class , owl:DatatypeProperty ; rdfs:subClassOf :Rights .
:hasRight a owl:ObjectProperty ; rdfs:domain :DataSubject .
:hasDate a owl:DatatypeProperty .
foaf:Person a owl:Class .
:consent001 a owl:NamedIndividual , :Consent .
:banking001 a owl:NamedIndividual , :Rights .
:Law a owl:NamedIndividual .
"""

_PROPOSALS = {
    "source": "lopdp",
    "concepts": [
        {"name": "DataSubject", "label": "titular", "type": "class",
         "definition": "Persona titular de los datos.", "articles": [4, 17]},
        {"name": "PortabilityRight", "label": "Derecho a la portabilidad", "type": "class",
         "definition": "Derecho a recibir sus datos.", "articles": [17]},
        {"name": "hasPurpose", "label": "tiene finalidad", "type": "property",
         "definition": "Relaciona un tratamiento con su finalidad.", "articles": ["10"]},
        {"name": "datasubject", "label": "duplicado", "type": "class", "articles": []},
    ],
}


def _report(ttl: str = _TTL):
    g = rdflib.Graph().parse(data=ttl, format="turtle")
    return SimpleNamespace(graph=g, path="memoria.ttl", source_format="turtle",
                           n_triples=len(g))


def _by_name(sources, name, origin=ORIGIN_ONTOLOGY):
    return next(c for c in sources.concepts if c.name == name and c.origin == origin)


def test_ontology_entities_match_inventory_minus_borrowed():
    rep = _report()
    inv = build_inventory(rep)
    src = build_alignment_sources(rep, proposals={})
    assert len(src.of_origin(ORIGIN_ONTOLOGY)) == len(inv.entries) - len(src.borrowed)
    assert src.borrowed == ["http://xmlns.com/foaf/0.1/Person"]
    assert is_borrowed("https://w3id.org/dpv#Consent")


def test_kinds_are_normalized_and_punning_is_kept():
    src = build_alignment_sources(_report(), proposals={})
    assert _by_name(src, "DataSubject").kinds == (ALIGN_CLASS,)
    assert _by_name(src, "hasRight").kinds == (ALIGN_PROPERTY,)
    assert _by_name(src, "hasDate").kinds == (ALIGN_PROPERTY,)
    assert _by_name(src, "Consent").kinds == (ALIGN_CLASS, ALIGN_PROPERTY)


def test_labels_definitions_and_texts():
    src = build_alignment_sources(_report(), proposals={})
    ds = _by_name(src, "DataSubject")
    assert ds.label == "Data subject"
    assert ds.definition == "Person whose data is processed."
    assert ds.name_text() == "Data Subject"
    assert "Person whose data is processed." in ds.semantic_text()
    assert _by_name(src, "hasRight").family is not None


def test_example_individuals_are_skipped_by_default():
    src = build_alignment_sources(_report(), proposals={})
    assert sorted(src.skipped_individuals) == ["banking001", "consent001"]
    assert src.individuals_total == 3                     # Law is punned: it enters as a class
    assert not any(ALIGN_INDIVIDUAL in c.kinds for c in src.concepts)
    assert _by_name(src, "Law").kinds == (ALIGN_CLASS,)


def test_individuals_can_be_included():
    src = build_alignment_sources(_report(), proposals={}, include_individuals=True)
    ind = _by_name(src, "consent001")
    assert ind.kinds == (ALIGN_INDIVIDUAL,)
    assert ind.family == "Consent"
    assert src.skipped_individuals == []


def test_ai_concepts_from_dict():
    ai = load_ai_concepts(_PROPOSALS)
    assert [c.name for c in ai] == ["DataSubject", "PortabilityRight", "hasPurpose"]  # dedup
    ds = ai[0]
    assert ds.key == "ai:DataSubject" and ds.origin == ORIGIN_AI
    assert ds.label == "titular" and ds.articles == [4, 17]
    assert ai[2].kinds == (ALIGN_PROPERTY,) and ai[2].articles == [10]
    assert "titular" in ds.semantic_text()          # Spanish label goes to the embeddings text
    assert ds.name_text() == "Data Subject"         # English identifier goes to the lexical match


def test_both_flows_together_and_unique_keys(tmp_path):
    f = tmp_path / "profile-proposed-concepts.json"
    f.write_text(json.dumps(_PROPOSALS, ensure_ascii=False), encoding="utf-8")
    src = build_alignment_sources(_report(), proposals=f)
    assert src.proposals_found and src.proposals_path == str(f)
    assert len(src.of_origin(ORIGIN_AI)) == 3
    keys = [c.key for c in src.concepts]
    assert len(keys) == len(set(keys))              # 'DataSubject' in both flows, distinct keys
    assert src.counts()["total"] == len(src.concepts)


def test_missing_proposals_file_is_not_an_error(tmp_path):
    src = build_alignment_sources(_report(), proposals=tmp_path / "no-existe.json")
    assert not src.proposals_found
    assert src.of_origin(ORIGIN_AI) == []
    assert src.of_origin(ORIGIN_ONTOLOGY)


def test_real_ontopriv_and_proposals():
    onto = ROOT / "data" / "input" / "ontopriv.rdf"
    if not onto.exists():
        pytest.skip("Falta data/input/ontopriv.rdf")
    from src.ingest.ontology_loader import load_ontology
    from src.alignment.sources import default_proposals_path
    rep = load_ontology(str(onto))
    inv = build_inventory(rep)
    proposals = ROOT / default_proposals_path()
    src = build_alignment_sources(rep, inventory=inv,
                                  proposals=proposals if proposals.exists() else {})
    assert len(src.of_origin(ORIGIN_ONTOLOGY)) == inv.totals["named_entities"] - len(src.borrowed)
    if proposals.exists():
        expected = {c["name"].lower() for c in json.loads(
            proposals.read_text(encoding="utf-8"))["concepts"]}
        assert len(src.of_origin(ORIGIN_AI)) == len(expected)