"""End-to-end test of the modular-core pipeline (Sprint 3, S3-T09).

The per-task tests check each stage in isolation; this one runs the whole chain on the real
OntoPriv -- load -> inventory -> split -> emit -- and asserts the invariants that must hold
for the modular core to be usable: totals reconcile, every named entity lands in exactly one
module file, the core is self-contained, and the profile imports the core."""
from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from src.ingest.ontology_loader import load_ontology
from src.core.inventory import build_inventory
from src.core.split import assign_modules, CORE, PROFILE
from src.core.emit import materialize_modules, CORE_IRI, PROFILE_IRI

ROOT = Path(__file__).resolve().parents[1]
ONTOPRIV = ROOT / "data" / "input" / "ontopriv.rdf"


@pytest.fixture(scope="module")
def ontopriv():
    if not ONTOPRIV.exists():
        pytest.skip("OntoPriv no disponible en data/input/ontopriv.rdf")
    return load_ontology(str(ONTOPRIV))


def test_pipeline_reconciles_and_splits_every_entity(ontopriv):
    inventory = build_inventory(ontopriv)
    split = assign_modules(ontopriv, inventory=inventory)

    # The inventory agrees with the Sprint 1 loader.
    assert inventory.totals["classes"] == ontopriv.n_classes
    assert inventory.totals["object_properties"] == ontopriv.n_object_props
    assert inventory.totals["data_properties"] == ontopriv.n_data_props

    # Every named entity lands in exactly one module (punning not double-counted).
    total = len(split.module(CORE)) + len(split.module(PROFILE))
    assert total == inventory.totals["named_entities"]
    assert len(split.module(CORE)) > 0 and len(split.module(PROFILE)) > 0


def _named_entities(graph: Graph) -> set[str]:
    subjects: set[str] = set()
    for kind in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty):
        subjects |= {str(s) for s in graph.subjects(RDF.type, kind) if isinstance(s, URIRef)}
    return subjects


def test_pipeline_emits_autonomous_core_and_importing_profile(ontopriv, tmp_path):
    result = materialize_modules(ontopriv, out_dir=tmp_path / "out")

    # The core must be self-contained (no reference into the profile).
    assert result.dangling_core_refs == []

    core = Graph()
    core.parse(result.core_path, format="xml")
    profile = Graph()
    profile.parse(result.profile_path, format="xml")

    # The profile imports the core, and both are valid, non-empty RDF/XML.
    assert (URIRef(PROFILE_IRI), OWL.imports, URIRef(CORE_IRI)) in profile
    assert len(core) > 0 and len(profile) > 0

    # Every named class/property is defined in exactly one of the two files.
    core_entities = _named_entities(core)
    profile_entities = _named_entities(profile)
    assert core_entities and profile_entities
    assert core_entities.isdisjoint(profile_entities)