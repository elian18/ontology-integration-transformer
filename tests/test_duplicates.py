"""S4-T06: are the AI-proposed concepts already in OntoPriv? (flow B, first step)"""
import json
import zlib

import numpy as np
import pytest

from src.alignment.sources import load_ai_concepts, AlignmentSources, SourceConcept, ORIGIN_ONTOLOGY
from src.alignment.lexical import normalize_for_lexical
from src.alignment.candidates import AlignmentSettings
from src.alignment.duplicates import (
    find_duplicates, duplicate_threshold, name_match_threshold, duplicates_summary,
    render_console, STATUS_DUPLICATE, STATUS_NEW, REASON_SCORE, REASON_SAME_NAME,
)

SETTINGS = AlignmentSettings(top_k=3, weight_lexical=0.4, weight_semantic=0.6)


def fake_embed(texts):
    """Deterministic bag-of-words embedding (L2-normalized), no model needed."""
    out = []
    for text in texts:
        v = np.zeros(128)
        for tok in normalize_for_lexical(text).split():
            v[zlib.crc32(tok.encode()) % 128] += 1.0
        n = np.linalg.norm(v)
        out.append((v / n if n else v).tolist())
    return out


def _onto(name, kinds=("class",), definition=None, family="Rights"):
    return SourceConcept(key=f"http://ex.org/lopdp#{name}", origin=ORIGIN_ONTOLOGY, name=name,
                         kinds=kinds, definition=definition, family=family)


ONTO = [
    _onto("Data_subject", definition="natural person whose personal data is processed",
          family="Members_of_the_personal_data_protection_system"),
    _onto("Right_to_portability", definition="right to receive personal data in a portable format"),
    _onto("Consent", family="Principles"),                       # bare name, no definition
    _onto("Exercise_of_rights"),                                 # bare name, no definition
    _onto("Banking", family="Terminology"),
    _onto("hasRecipient", kinds=("property",), family="(sin dominio)"),
]

AI = [
    {"name": "DataSubject", "label": "titular", "type": "class",
     "definition": "natural person whose personal data is processed"},
    {"name": "PortabilityRight", "label": "Derecho a la portabilidad", "type": "class",
     "definition": "right to receive personal data in a portable format"},
    # Same name as OntoPriv, but a long Spanish definition keeps the combined score low:
    {"name": "Consent", "label": "Consentimiento", "type": "class",
     "definition": "Manifestacion de voluntad libre, especifica, informada e inequivoca "
                   "por la que el titular autoriza el tratamiento de sus datos"},
    {"name": "RightsExercise", "label": "Ejercicio de derechos", "type": "class",
     "definition": "Obligacion del Estado y entidades educativas de proveer capacitacion "
                   "y derecho de adolescentes de actuar por representantes"},
    {"name": "DigitalEducationRight", "label": "Derecho a la educación digital", "type": "class",
     "definition": "acceso al conocimiento sobre el uso adecuado de las tecnologias"},
    {"name": "hasDataSubject", "label": "tiene titular", "type": "property"},
]


def _sources():
    return AlignmentSources("memoria", None, concepts=ONTO + load_ai_concepts({"concepts": AI}),
                            proposals_found=True)


def _report(threshold=0.75, name_match=0.95, **kw):
    return find_duplicates(_sources(), embed_fn=fake_embed, settings=SETTINGS,
                           threshold=threshold, name_match=name_match, **kw)


def _check(report, name):
    return next(c for c in report.checks if c.concept.name == name)


# --- configuration --------------------------------------------------------------------------

def test_thresholds_from_config():
    assert duplicate_threshold({}) == 0.75
    assert name_match_threshold({}) == 0.95
    cfg = {"alignment": {"duplicate_threshold": 0.8, "duplicate_name_match": 0.9}}
    assert duplicate_threshold(cfg) == 0.8 and name_match_threshold(cfg) == 0.9
    with pytest.raises(ValueError):
        duplicate_threshold({"alignment": {"duplicate_threshold": 1.5}})
    with pytest.raises(ValueError):
        name_match_threshold({"alignment": {"duplicate_name_match": -0.1}})


# --- score rule -----------------------------------------------------------------------------

def test_existing_concepts_are_marked_by_score():
    r = _report()
    assert r.status_of("ai:DataSubject") == STATUS_DUPLICATE
    assert r.reason_of("ai:DataSubject") == REASON_SCORE
    assert r.status_of("ai:PortabilityRight") == STATUS_DUPLICATE     # same words, other order
    ds = _check(r, "DataSubject")
    assert ds.best.name == "Data_subject"
    assert ds.best.score == pytest.approx(0.4 * ds.best.lexical + 0.6 * ds.best.semantic)


# --- same-name rule (measured on the LOPDP run) ---------------------------------------------

def test_same_name_rescues_low_score_duplicates():
    r = _report()
    for name, onto_name in (("Consent", "Consent"), ("RightsExercise", "Exercise_of_rights")):
        c = _check(r, name)
        assert c.best.score < 0.75                                   # the score alone misses it
        assert c.status == STATUS_DUPLICATE and c.reason == REASON_SAME_NAME
        assert c.best.name == onto_name and c.best.lexical >= 0.95


def test_partial_name_is_not_enough():
    """Sharing one word ('Consent' in 'ConsentWithdrawal') does not trigger the same-name rule."""
    src = AlignmentSources("m", None, concepts=ONTO + load_ai_concepts({"concepts": [
        {"name": "ConsentWithdrawal", "label": "Revocatoria del consentimiento", "type": "class",
         "definition": "Facultad del titular de retirar en cualquier momento su autorizacion"}]}),
        proposals_found=True)
    r = find_duplicates(src, embed_fn=fake_embed, settings=SETTINGS, threshold=0.75,
                        name_match=0.95)
    c = r.checks[0]
    assert c.best.name == "Consent" and c.best.lexical < 0.95
    assert c.status == STATUS_NEW and c.reason is None


def test_really_new_concept_is_marked_new():
    r = _report()
    new = _check(r, "DigitalEducationRight")
    assert new.status == STATUS_NEW and new.reason is None
    assert new.best is not None and new.best.score < 0.75           # matches are still kept


# --- kinds, thresholds, reuse ---------------------------------------------------------------

def test_kinds_are_respected():
    r = _report()
    prop = _check(r, "hasDataSubject")
    assert [m.name for m in prop.matches] == ["hasRecipient"]        # only OntoPriv properties
    cls = _check(r, "DataSubject")
    assert "hasRecipient" not in [m.name for m in cls.matches]
    assert len(cls.matches) == 3


def test_threshold_moves_the_mark():
    assert len(_report(threshold=0.0).of_status(STATUS_NEW)) == 0          # everything matches
    none_by_score = _report(threshold=1.0)
    assert all(c.reason != REASON_SCORE for c in none_by_score.checks)     # nothing is perfect
    assert none_by_score.status_of("ai:Consent") == STATUS_DUPLICATE       # same name still holds


def test_precomputed_vectors_are_reused():
    src = _sources()
    vecs = {c.key: np.asarray(v) for c, v in
            zip(src.concepts, fake_embed([c.semantic_text() for c in src.concepts]))}

    def must_not_run(_texts):
        raise AssertionError("no debe recalcular embeddings")

    r = find_duplicates(src, embed_fn=must_not_run, settings=SETTINGS, threshold=0.75,
                        name_match=0.95, vectors=vecs)
    assert r.status_of("ai:DataSubject") == STATUS_DUPLICATE


def test_no_ai_or_no_ontology():
    only_onto = AlignmentSources("m", None, concepts=list(ONTO))
    assert find_duplicates(only_onto, embed_fn=fake_embed, settings=SETTINGS,
                           threshold=0.75, name_match=0.95).checks == []
    only_ai = AlignmentSources("m", None, concepts=load_ai_concepts({"concepts": AI}),
                               proposals_found=True)
    r = find_duplicates(only_ai, embed_fn=fake_embed, settings=SETTINGS, threshold=0.75,
                        name_match=0.95)
    assert all(c.status == STATUS_NEW for c in r.checks)


def test_summary_bands_and_console():
    r = _report()
    s = duplicates_summary(r)
    json.dumps(s)
    assert s["possible_duplicates"] + s["new"] == len(AI)
    assert s["by_same_name"] == 2
    assert sum(s["bands"].values()) == len(AI)
    text = render_console(r)
    assert "(mismo nombre)" in text and "posibles duplicados: 4 (2 por mismo nombre)" in text