"""S5-T02: human decisions on the AI-proposed concepts (approve, reject, duplicate, undo)."""
import json
from pathlib import Path

import pytest

from src.core.emit import PROFILE_IRI
from src.alignment.decisions import (read_log, DECISIONS_FILE, DECISION_APPROVED,
                                     DECISION_REJECTED, DECISION_DUPLICATE, TARGET_CONCEPT)
from src.alignment.ai_concepts import (
    ai_concepts, ontology_entities, concept_iri, suggested_entity_kind, concept_decisions, decide,
    concept_progress, review_rows, approved_new_concepts, confirmed_duplicates,
    excluded_from_mapping, render_console,
    ACTION_APPROVE, ACTION_REJECT, ACTION_DUPLICATE, ACTION_UNDO,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = "ontopriv+lopdp"
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
FRANC = "http://www.semanticweb.org/franc/ontologies/2023/9/OntologiaLOPDP#"
META = {"embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "generated_at": "2026-09-30T22:10:03+00:00",
        "justification": {"llm_model": "gemini-3.1-flash-lite", "prompt_version": 2}}


def _row(key, origin, name, rank=1, kinds=("class",), **kw):
    base = {"concept_key": key, "origin": origin, "concept_name": name,
            "concept_label": None, "concept_kinds": list(kinds), "concept_definition": None,
            "concept_articles": [], "duplicate_status": None, "duplicate_of": None,
            "duplicate_of_name": None, "duplicate_score": None, "duplicate_reason": None,
            "rank": rank, "dpv_iri": f"https://w3id.org/dpv#T{rank}", "dpv_name": f"T{rank}",
            "score": 0.5, "proposed_relation": None}
    base.update(kw)
    return base


def _data():
    rows = [
        _row(ONTO + "Consent", "ontology", "Consent"),
        _row(FRANC + "Right_to_portability", "ontology", "Right_to_portability"),
        # a new AI concept (3 candidates, listed out of order on purpose)
        _row("ai:DataBreachNotice", "ai", "DataBreachNotice", rank=2, duplicate_status="new"),
        _row("ai:DataBreachNotice", "ai", "DataBreachNotice", rank=1, duplicate_status="new",
             concept_label="Notificacion de vulneracion", concept_articles=[46],
             concept_definition="Aviso a la autoridad de una vulneracion de seguridad."),
        # a possible duplicate by score
        _row("ai:PortabilityRight", "ai", "PortabilityRight", duplicate_status="possible_duplicate",
             duplicate_of=FRANC + "Right_to_portability", duplicate_of_name="Right_to_portability",
             duplicate_score=0.82, duplicate_reason="score", concept_articles=[17]),
        # a possible duplicate with the SAME name as an OntoPriv entity
        _row("ai:Consent", "ai", "Consent", duplicate_status="possible_duplicate",
             duplicate_of=ONTO + "Consent", duplicate_of_name="Consent", duplicate_score=1.0,
             duplicate_reason="same_name", concept_articles=[8]),
        # a property (lowerCamelCase)
        _row("ai:hasRetentionPeriod", "ai", "hasRetentionPeriod", kinds=("property",),
             duplicate_status="new"),
    ]
    return {"metadata": META, "rows": rows}


def _by_name(concepts):
    return {c.name: c for c in concepts}


# ---------------------------------------------------------------- reading the concepts
def test_one_concept_per_ai_key_with_its_best_candidate():
    concepts = ai_concepts(_data())
    assert [c.name for c in concepts] == ["DataBreachNotice", "PortabilityRight", "Consent",
                                          "hasRetentionPeriod"]
    notice = _by_name(concepts)["DataBreachNotice"]
    assert notice.row["rank"] == 1 and notice.articles == [46]
    assert notice.label == "Notificacion de vulneracion"


def test_iri_goes_into_the_profile_and_can_be_another_profile():
    c = _by_name(ai_concepts(_data()))["DataBreachNotice"]
    assert c.iri == PROFILE_IRI + "#DataBreachNotice"
    peru = "http://www.semanticweb.org/profiles/peru-ley-29733"
    assert _by_name(ai_concepts(_data(), peru))["DataBreachNotice"].iri == peru + "#DataBreachNotice"
    assert concept_iri(peru + "#", "X") == peru + "#X"
    with pytest.raises(ValueError, match="Nombre no valido"):
        concept_iri(peru, "Derecho de acceso")


def test_suggested_kind_comes_from_the_name_rule():
    assert suggested_entity_kind(["class"]) == "class"
    assert suggested_entity_kind(["property"]) == "object_property"
    assert _by_name(ai_concepts(_data()))["hasRetentionPeriod"].suggested_kind == "object_property"


def test_same_name_as_an_ontopriv_entity_is_detected():
    concepts = _by_name(ai_concepts(_data()))
    assert concepts["Consent"].name_clash == ONTO + "Consent"
    assert concepts["PortabilityRight"].name_clash is None


# ---------------------------------------------------------------- the four decisions
def test_approve_new_concept_with_confirmed_kind(tmp_path):
    log = tmp_path / DECISIONS_FILE
    c = _by_name(ai_concepts(_data()))["DataBreachNotice"]
    d = decide(c, ACTION_APPROVE, entity_kind="class", reviewer="Elian", source_id=SRC,
               log_path=log, metadata=META)
    assert d.target == TARGET_CONCEPT and d.decision == DECISION_APPROVED
    assert d.snapshot["proposed_iri"] == PROFILE_IRI + "#DataBreachNotice"
    assert d.snapshot["llm_model"] == "gemini-3.1-flash-lite"
    with pytest.raises(ValueError, match="Confirma el tipo"):
        decide(c, ACTION_APPROVE, reviewer="Elian", source_id=SRC, log_path=log)


def test_reject_new_concept(tmp_path):
    log = tmp_path / DECISIONS_FILE
    c = _by_name(ai_concepts(_data()))["hasRetentionPeriod"]
    d = decide(c, ACTION_REJECT, entity_kind="class", reviewer="Elian", source_id=SRC,
               log_path=log)
    assert d.decision == DECISION_REJECTED and d.entity_kind is None   # kind ignored


def test_confirm_detected_duplicate(tmp_path):
    log = tmp_path / DECISIONS_FILE
    concepts = _by_name(ai_concepts(_data()))
    d = decide(concepts["PortabilityRight"], ACTION_DUPLICATE, reviewer="Elian", source_id=SRC,
               log_path=log)
    assert d.decision == DECISION_DUPLICATE
    assert d.same_as == FRANC + "Right_to_portability"         # the detected one by default
    assert d.snapshot["duplicate_of"] == FRANC + "Right_to_portability"


def test_new_concept_can_be_a_duplicate_the_check_missed(tmp_path):
    """e.g. PrincipleOfLawfulness (marked new) is OntoPriv's Juridicity: the person names it."""
    log = tmp_path / DECISIONS_FILE
    data = _data()
    keys = {e["key"] for e in ontology_entities(data)}
    notice = _by_name(ai_concepts(data))["DataBreachNotice"]
    with pytest.raises(ValueError, match="Elige la entidad de OntoPriv"):
        decide(notice, ACTION_DUPLICATE, reviewer="Elian", source_id=SRC, log_path=log)
    with pytest.raises(ValueError, match="no es parte de OntoPriv"):
        decide(notice, ACTION_DUPLICATE, same_as=ONTO + "Nope", ontology_keys=keys,
               reviewer="Elian", source_id=SRC, log_path=log)
    d = decide(notice, ACTION_DUPLICATE, same_as=ONTO + "Consent", ontology_keys=keys,
               reviewer="Elian", source_id=SRC, log_path=log)
    assert d.same_as == ONTO + "Consent"
    dups = confirmed_duplicates(ai_concepts(data), concept_decisions(read_log(log), SRC))
    assert dups[0]["duplicate_of"] == ONTO + "Consent"
    assert dups[0]["detected_duplicate_of"] is None


def test_person_can_correct_the_detected_duplicate(tmp_path):
    log = tmp_path / DECISIONS_FILE
    data = _data()
    portability = _by_name(ai_concepts(data))["PortabilityRight"]
    decide(portability, ACTION_DUPLICATE, same_as=ONTO + "Consent", reviewer="Elian",
           source_id=SRC, log_path=log)
    dups = confirmed_duplicates(ai_concepts(data), concept_decisions(read_log(log), SRC))
    assert dups[0]["duplicate_of"] == ONTO + "Consent"
    assert dups[0]["detected_duplicate_of"] == FRANC + "Right_to_portability"


def test_ontology_entities_for_the_picker():
    entities = ontology_entities(_data())
    assert [e["name"] for e in entities] == ["Consent", "Right_to_portability"]
    assert [e["module"] for e in entities] == ["ley-organica-proteccion-datos-personales",
                                               "OntologiaLOPDP"]
    assert entities[0]["kinds"] == ["class"]


def test_rejected_duplicate_is_decided_as_new(tmp_path):
    """'No es duplicado': the person approves it as a new concept instead."""
    log = tmp_path / DECISIONS_FILE
    c = _by_name(ai_concepts(_data()))["PortabilityRight"]
    decide(c, ACTION_APPROVE, entity_kind="class", reviewer="Elian", source_id=SRC, log_path=log)
    current = concept_decisions(read_log(log), SRC)
    assert current[c.key].decision == DECISION_APPROVED


def test_same_name_cannot_be_approved_as_new(tmp_path):
    log = tmp_path / DECISIONS_FILE
    c = _by_name(ai_concepts(_data()))["Consent"]
    with pytest.raises(ValueError, match="Ya existe en OntoPriv una entidad llamada 'Consent'"):
        decide(c, ACTION_APPROVE, entity_kind="class", reviewer="Elian", source_id=SRC,
               log_path=log)
    assert not log.exists()                                  # nothing was written
    decide(c, ACTION_DUPLICATE, reviewer="Elian", source_id=SRC, log_path=log)
    assert len(read_log(log)) == 1


def test_undo_returns_to_pending_and_keeps_history(tmp_path):
    log = tmp_path / DECISIONS_FILE
    c = _by_name(ai_concepts(_data()))["DataBreachNotice"]
    decide(c, ACTION_REJECT, reviewer="Elian", source_id=SRC, log_path=log)
    decide(c, ACTION_UNDO, reviewer="Elian", source_id=SRC, log_path=log)
    assert c.key not in concept_decisions(read_log(log), SRC)
    assert len(read_log(log)) == 2


# ---------------------------------------------------------------- what the next tasks read
def _decided(tmp_path):
    log = tmp_path / DECISIONS_FILE
    concepts = ai_concepts(_data())
    by = _by_name(concepts)
    decide(by["DataBreachNotice"], ACTION_APPROVE, entity_kind="class", reviewer="Elian",
           source_id=SRC, log_path=log)
    decide(by["PortabilityRight"], ACTION_DUPLICATE, reviewer="Elian", source_id=SRC,
           log_path=log)
    decide(by["Consent"], ACTION_REJECT, reviewer="Elian", source_id=SRC, log_path=log)
    return concepts, concept_decisions(read_log(log), SRC)


def test_progress_and_review_rows(tmp_path):
    concepts, current = _decided(tmp_path)
    assert concept_progress(concepts, current) == {
        "total": 4, "pending": 1, DECISION_APPROVED: 1, DECISION_REJECTED: 1,
        DECISION_DUPLICATE: 1}
    rows = {r["name"]: r for r in review_rows(concepts, current)}
    assert rows["PortabilityRight"]["same_as"] == FRANC + "Right_to_portability"
    assert rows["DataBreachNotice"]["status_label"] == "aprobado"
    assert rows["PortabilityRight"]["status_label"] == "ya existe en OntoPriv"
    assert rows["hasRetentionPeriod"]["status_label"] == "por validar"
    assert rows["DataBreachNotice"]["reviewer"] == "Elian"


def test_outputs_for_the_graph_and_the_mapping_review(tmp_path):
    concepts, current = _decided(tmp_path)
    new = approved_new_concepts(concepts, current)
    assert [(n["name"], n["entity_kind"], n["articles"]) for n in new] == [
        ("DataBreachNotice", "class", [46])]
    dups = confirmed_duplicates(concepts, current)
    assert dups[0]["duplicate_of"] == FRANC + "Right_to_portability"
    assert dups[0]["articles"] == [17]
    assert excluded_from_mapping(concepts, current) == {"ai:PortabilityRight", "ai:Consent"}


def test_other_input_decisions_are_ignored(tmp_path):
    log = tmp_path / DECISIONS_FILE
    c = _by_name(ai_concepts(_data()))["DataBreachNotice"]
    decide(c, ACTION_REJECT, reviewer="Elian", source_id="ley-peru", log_path=log)
    assert concept_decisions(read_log(log), SRC) == {}


def test_console_summary(tmp_path):
    concepts, current = _decided(tmp_path)
    text = render_console(concepts, current)
    assert "Conceptos propuestos por la IA: 4 (2 posibles duplicados de OntoPriv, 2 nuevos)" in text
    assert "Por validar: 1" in text


def test_real_candidates_file():
    real = ROOT / "data/output/alignment-candidates.json"
    if not real.exists():
        pytest.skip("No hay alignment-candidates.json (Sprint 4)")
    concepts = ai_concepts(json.loads(real.read_text(encoding="utf-8")))
    assert len(concepts) == 93
    assert sum(c.is_possible_duplicate for c in concepts) == 34
    assert all(c.iri.startswith(PROFILE_IRI + "#") for c in concepts)
    clashes = sorted(c.name for c in concepts if c.name_clash)
    assert clashes == ["Anonymization", "Confidentiality", "Consent", "Processing"]
    assert all(c.is_possible_duplicate for c in concepts if c.name_clash)