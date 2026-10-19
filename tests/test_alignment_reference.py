"""S7-T03: the automatic reference (your decisions) and its leak control.

Small hand-made base concepts: the leak control must exclude, as precedent, every base concept
the inheritance rules consider the same (same IRI, same name/label/alias, very similar), with
compatible kinds, and nothing else. ``score`` is checked with exact counts.
"""
import numpy as np
import pytest

from src.alignment.inherit import BaseConcept, BaseIndex
from src.alignment.reference import (REF_BASE, REF_BLIND, Probe, RefItem, Reference, agreement,
                                     base_reference, blind_reference, external_reference,
                                     probe_of, same_concepts, score)
from src.alignment.sources import SourceConcept

D = "https://w3id.org/dpv#"
NS1, NS2 = "http://ex.org/ns1#", "http://ex.org/ns2#"


def _bc(key, name, kinds=("class",), status="aligned", mappings=None, label=None, aliases=()):
    return BaseConcept(key=key, iri=key if not key.startswith("ai:") else "http://p#" + name,
                       origin="ai" if key.startswith("ai:") else "ontology", name=name,
                       label=label, definition=None, kinds=tuple(kinds), status=status,
                       mappings=list(mappings or []), aliases=list(aliases))


def _map(name, relation="skos:broadMatch"):
    return {"dpv_iri": D + name, "dpv_name": name, "relation": relation}


@pytest.fixture
def base():
    return BaseIndex(concepts=[
        _bc(NS1 + "Corrective_measures", "Corrective_measures", mappings=[_map("CorrectiveMeasure")]),
        _bc(NS2 + "Corrective_measures", "Corrective_measures", mappings=[_map("CorrectiveMeasure")]),
        _bc("ai:CorrectiveMeasures", "CorrectiveMeasures", mappings=[_map("CorrectiveMeasure")]),
        _bc(NS1 + "corrective_measures", "corrective_measures", kinds=("property",),
            status="no_match"),
        _bc(NS1 + "Right_of_access", "Right_of_access", aliases=["AccessRight"],
            mappings=[_map("DataSubjectRight")]),
        _bc(NS1 + "Sanction", "Sanction", status="no_match"),
    ], source_id="ontopriv+lopdp")


# --------------------------------------------------------------------------- "the same"

def test_same_name_in_another_namespace_and_the_ai_twin_are_forbidden(base):
    ref = base_reference(base)
    first = ref.by_key()[NS1 + "Corrective_measures"]
    assert first.forbidden == {NS2 + "Corrective_measures": "same_name",
                               "ai:CorrectiveMeasures": "same_name"}


def test_a_property_with_the_same_name_is_not_forbidden_for_a_class(base):
    ref = base_reference(base).by_key()
    assert NS1 + "corrective_measures" not in ref[NS1 + "Corrective_measures"].forbidden
    assert ref[NS1 + "corrective_measures"].forbidden == {}          # no other property


def test_alias_of_a_confirmed_duplicate_counts_as_the_same(base):
    probe = Probe(key="x#AccessRight", iri="x#AccessRight", name="AccessRight", label=None,
                  kinds=("class",))
    assert same_concepts([probe], base, None, None) == [{NS1 + "Right_of_access": "same_name"}]


def test_same_iri_rule_and_self_exclusion(base):
    probe = Probe(key=NS1 + "Sanction", iri=NS1 + "Sanction", name="Penalty", label=None,
                  kinds=("class",))
    assert same_concepts([probe], base, None, None) == [{}]            # itself is excluded
    kept = same_concepts([probe], base, None, None, exclude_self=False)
    assert kept == [{NS1 + "Sanction": "same_iri"}]


def test_very_similar_by_meaning_is_forbidden(base):
    probe = Probe(key="x#Penalty", iri="x#Penalty", name="Penalty", label=None, kinds=("class",))
    vectors = np.eye(len(base.concepts))
    sanction = [c.key for c in base.concepts].index(NS1 + "Sanction")
    probe_vec = vectors[[sanction]]                   # same meaning as "Sanction" only
    cfg = {"alignment": {"duplicate_threshold": 0.6, "duplicate_name_match": 0.95}}
    assert same_concepts([probe], base, probe_vec, vectors, cfg) == [{NS1 + "Sanction": "similar"}]
    assert same_concepts([probe], base, None, None, cfg) == [{}]       # no vectors, no rule 3


def test_probe_of_keeps_only_alignable_kinds():
    c = SourceConcept(key="k", origin="ontology", name="X", kinds=("class", "individual"))
    assert probe_of(c).kinds == ("class",)


# --------------------------------------------------------------------------- references

def test_base_reference_keeps_the_decisions(base):
    ref = base_reference(base)
    assert ref.name == REF_BASE and len(ref.items) == 6 and ref.aligned() == 4
    access = ref.by_key()[NS1 + "Right_of_access"]
    assert access.terms == {D + "DataSubjectRight": "skos:broadMatch"}
    s = ref.summary()
    assert s["with_forbidden"] == 3 and s["forbidden_pairs"] == 6
    assert s["forbidden_by_rule"] == {"same_name": 6}


def test_external_reference_always_forbids_the_concept_it_inherited_from(base):
    concepts = [SourceConcept(key="g#Penalty", origin="ontology", name="Penalty",
                              kinds=("class",)),
                SourceConcept(key="g#Other", origin="ontology", name="Other", kinds=("class",))]
    inherited = {"inherited": [{"key": "g#Penalty", "name": "Penalty", "method": "similar",
                                "base_key": NS1 + "Sanction", "status": "no_match",
                                "mappings": []}]}
    ref = external_reference(concepts, inherited, base)
    assert [i.key for i in ref.items] == ["g#Penalty"]                 # only the inherited ones
    assert ref.items[0].forbidden == {NS1 + "Sanction": "similar"}
    assert ref.items[0].status == "no_match" and ref.aligned() == 0


def test_blind_reference_reads_gold_and_borrows_the_leak_control(base):
    sample = [{"concept_key": NS1 + "Corrective_measures", "concept_name": "Corrective_measures",
               "gold_dpv_name": "CorrectiveMeasure", "gold_relation": "closeMatch"},
              {"concept_key": "ai:AccessRight", "concept_name": "AccessRight",
               "gold_dpv_name": "none", "gold_relation": "none"},
              {"concept_key": "ai:Unlabelled", "gold_dpv_name": "", "gold_relation": ""}]
    outside = {"ai:AccessRight": SourceConcept(key="ai:AccessRight", origin="ai",
                                               name="AccessRight", kinds=("class",))}
    ref = blind_reference(sample, base_reference(base), base, outside)
    assert ref.name == REF_BLIND and len(ref.items) == 2                # unlabelled row skipped
    first, second = ref.items
    assert first.terms == {D + "CorrectiveMeasure": "skos:closeMatch"}
    assert set(first.forbidden) == {NS2 + "Corrective_measures", "ai:CorrectiveMeasures"}
    assert second.status == "no_match" and second.terms == {}
    assert second.forbidden == {NS1 + "Right_of_access": "same_name"}  # its OntoPriv twin


# --------------------------------------------------------------------------- scoring

def _ref():
    return Reference("t", [
        RefItem("a", "A", "aligned", {D + "X": "skos:exactMatch"}),
        RefItem("b", "B", "aligned", {D + "Y": "skos:broadMatch", D + "Z": "skos:relatedMatch"}),
        RefItem("c", "C", "aligned", {D + "W": "skos:closeMatch"}),
        RefItem("d", "D", "no_match"),
        RefItem("e", "E", "no_match"),
        RefItem("f", "F", "no_match"),
    ])


def test_score_counts_precision_coverage_none_and_type():
    outputs = {"a": {"term": D + "X", "relation": "skos:closeMatch"},      # right term, wrong type
               "b": {"term": D + "Z", "relation": "skos:relatedMatch"},    # 2nd approved term
               "c": {"none": True},                                       # wrong "none"
               "d": {"term": D + "Q", "relation": "skos:exactMatch"},     # proposal on no_match
               "e": {"none": True},                                       # right "none"
               "f": {"none": True}}                                       # right "none"
    s = score(outputs, _ref())
    assert (s["proposals"], s["correct"], s["wrong"], s["on_no_match"]) == (3, 2, 1, 1)
    assert s["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert s["coverage"] == pytest.approx(0.6667, abs=1e-4)
    assert s["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert (s["none_claimed"], s["none_ok"], s["none_wrong"]) == (3, 2, 1)
    assert s["none_recall"] == pytest.approx(0.6667, abs=1e-4)
    assert (s["relation_typed"], s["relation_ok"]) == (2, 1)


def test_score_without_any_answer():
    s = score({}, _ref())
    assert s["proposals"] == 0 and s["precision"] is None and s["coverage"] == 0
    assert s["f1"] is None and s["none_claimed"] == 0


def test_agreement_between_two_references():
    other = Reference("o", [RefItem("a", "A", "aligned", {D + "X": "skos:closeMatch"}),
                            RefItem("b", "B", "aligned", {D + "Q": "skos:broadMatch"}),
                            RefItem("d", "D", "no_match"), RefItem("zz", "Z", "no_match")])
    assert agreement(_ref(), other) == {"shared": 3, "agree": 2, "rate": 0.6667}