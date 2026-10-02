"""S5-T01: the human decisions log (append-only, latest wins, survives regenerating candidates)."""
import json
from pathlib import Path

import pytest

from src.alignment.decisions import (
    make_decision, append_decision, read_log, latest_decisions, count_decisions,
    snapshot_from_row, review_settings, decisions_path, render_console, decision_key,
    DECISIONS_FILE, DEFAULT_SOURCE_ID, TARGET_CONCEPT, TARGET_MAPPING,
    DECISION_APPROVED, DECISION_REJECTED, DECISION_DUPLICATE, DECISION_NO_MATCH,
    DECISION_PENDING, MAPPING_RELATIONS, DECISION_LABELS,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = "ontopriv+lopdp"
CONSENT = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#Consent"
DPV_CONSENT = "https://w3id.org/dpv#Consent"
DPV_RIGHT = "https://w3id.org/dpv#DataSubjectRight"


def _mapping(decision=DECISION_APPROVED, relation="skos:exactMatch", dpv_iri=DPV_CONSENT,
             reviewer="Elian", at="2026-10-01T20:00:00+00:00", **kw):
    return make_decision(source_id=SRC, target=TARGET_MAPPING, concept_key=CONSENT,
                         dpv_iri=dpv_iri, decision=decision, relation=relation,
                         reviewer=reviewer, decided_at=at, **kw)


# ---------------------------------------------------------------- validation
def test_mapping_approval_needs_a_skos_type():
    with pytest.raises(ValueError, match="tipo SKOS"):
        _mapping(relation=None)
    with pytest.raises(ValueError, match="tipo SKOS"):
        _mapping(relation="none")          # the AI's "none" is a no_match decision, not a type
    assert _mapping().relation == "skos:exactMatch"


def test_only_skos_mapping_properties_are_allowed():
    assert MAPPING_RELATIONS == ("skos:exactMatch", "skos:closeMatch", "skos:broadMatch",
                                 "skos:narrowMatch", "skos:relatedMatch")


def test_reviewer_is_mandatory():
    with pytest.raises(ValueError, match="revisor"):
        _mapping(reviewer="   ")


def test_rejection_and_no_match_carry_no_type():
    with pytest.raises(ValueError, match="solo se registra al aprobar"):
        _mapping(decision=DECISION_REJECTED, relation="skos:closeMatch")
    no_match = _mapping(decision=DECISION_NO_MATCH, relation=None, dpv_iri="")
    assert no_match.dpv_iri == "" and no_match.relation is None
    with pytest.raises(ValueError, match="concepto entero"):
        _mapping(decision=DECISION_NO_MATCH, relation=None, dpv_iri=DPV_CONSENT)


def test_withdrawing_a_no_match_needs_no_dpv_term():
    no_match = _mapping(decision=DECISION_NO_MATCH, relation=None, dpv_iri="")
    back = _mapping(decision=DECISION_PENDING, relation=None, dpv_iri="")
    assert no_match.key == back.key                    # same key: the pending withdraws it
    with pytest.raises(ValueError, match="Falta el termino del DPV"):
        _mapping(decision=DECISION_REJECTED, relation=None, dpv_iri="")


def test_concept_decisions():
    ok = make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:PortabilityRight",
                       decision=DECISION_APPROVED, entity_kind="class", reviewer="Elian")
    assert ok.entity_kind == "class" and ok.decided_at.endswith("+00:00")
    with pytest.raises(ValueError, match="confirmar su tipo"):
        make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:PortabilityRight",
                      decision=DECISION_APPROVED, reviewer="Elian")
    with pytest.raises(ValueError, match="a que entidad de OntoPriv equivale"):
        make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:DataSubject",
                      decision=DECISION_DUPLICATE, reviewer="Elian")
    dup = make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:DataSubject",
                        decision=DECISION_DUPLICATE, reviewer="Elian", same_as=CONSENT)
    assert dup.entity_kind is None and dup.same_as == CONSENT
    with pytest.raises(ValueError, match="solo se registra al marcar"):
        make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:X",
                      decision=DECISION_REJECTED, reviewer="Elian", same_as=CONSENT)
    with pytest.raises(ValueError, match="no lleva entidad equivalente"):
        _mapping(same_as=CONSENT)
    with pytest.raises(ValueError, match="no aplica"):
        make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:X",
                      decision=DECISION_NO_MATCH, reviewer="Elian")
    with pytest.raises(ValueError, match="no lleva termino del DPV"):
        make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:X",
                      decision=DECISION_REJECTED, dpv_iri=DPV_CONSENT, reviewer="Elian")


def test_every_code_has_a_spanish_label():
    for code in (DECISION_APPROVED, DECISION_REJECTED, DECISION_DUPLICATE, DECISION_NO_MATCH,
                 DECISION_PENDING):
        assert DECISION_LABELS[code]


# ---------------------------------------------------------------- the log
def test_append_creates_folder_and_keeps_history(tmp_path):
    path = tmp_path / "review" / DECISIONS_FILE
    append_decision(_mapping(relation="skos:exactMatch"), path)
    append_decision(_mapping(relation="skos:closeMatch", at="2026-10-01T21:00:00+00:00"), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2                                   # nothing is overwritten
    assert all(json.loads(line)["source_id"] == SRC for line in lines)
    records = read_log(path)
    assert [d.relation for d in records] == ["skos:exactMatch", "skos:closeMatch"]


def test_latest_decision_wins_and_pending_withdraws(tmp_path):
    path = tmp_path / DECISIONS_FILE
    append_decision(_mapping(relation="skos:exactMatch"), path)
    append_decision(_mapping(relation="skos:broadMatch"), path)
    current = latest_decisions(read_log(path))
    key = decision_key(SRC, TARGET_MAPPING, CONSENT, DPV_CONSENT)
    assert current[key].relation == "skos:broadMatch"
    append_decision(_mapping(decision=DECISION_PENDING, relation=None), path)
    assert key not in latest_decisions(read_log(path))     # back to "por validar"
    assert len(read_log(path)) == 3                         # ...but the history stays


def test_each_candidate_is_its_own_key(tmp_path):
    path = tmp_path / DECISIONS_FILE
    append_decision(_mapping(relation="skos:closeMatch"), path)
    append_decision(_mapping(decision=DECISION_REJECTED, relation=None, dpv_iri=DPV_RIGHT), path)
    current = latest_decisions(read_log(path))
    assert len(current) == 2
    counts = count_decisions(current)
    assert counts[TARGET_MAPPING][DECISION_APPROVED] == 1
    assert counts[TARGET_MAPPING][DECISION_REJECTED] == 1


def test_decisions_of_another_input_do_not_mix(tmp_path):
    path = tmp_path / DECISIONS_FILE
    append_decision(_mapping(), path)
    other = make_decision(source_id="ley-peru", target=TARGET_MAPPING, concept_key=CONSENT,
                          dpv_iri=DPV_CONSENT, decision=DECISION_REJECTED, reviewer="Elian")
    append_decision(other, path)
    assert len(latest_decisions(read_log(path), SRC)) == 1
    assert latest_decisions(read_log(path), SRC)[
        decision_key(SRC, TARGET_MAPPING, CONSENT, DPV_CONSENT)].decision == DECISION_APPROVED
    assert len(latest_decisions(read_log(path))) == 2


def test_old_lines_without_same_as_still_load(tmp_path):
    path = tmp_path / DECISIONS_FILE
    line = {"source_id": SRC, "target": TARGET_MAPPING, "concept_key": CONSENT,
            "decision": DECISION_APPROVED, "reviewer": "Elian", "decided_at": "2026-10-01",
            "dpv_iri": DPV_CONSENT, "relation": "skos:exactMatch", "entity_kind": None,
            "note": None, "snapshot": {}}
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    assert read_log(path)[0].same_as is None


def test_missing_file_is_an_empty_log(tmp_path):
    assert read_log(tmp_path / "nope.jsonl") == []


def test_damaged_line_is_reported_not_skipped(tmp_path):
    path = tmp_path / DECISIONS_FILE
    append_decision(_mapping(), path)
    with open(path, "a", encoding="utf-8") as h:
        h.write("{esto no es json\n")
    with pytest.raises(ValueError, match="Linea 2"):
        read_log(path)


def test_decisions_survive_regenerating_the_candidates(tmp_path):
    """The log never reads or writes the candidates file: regenerating it changes nothing."""
    candidates = tmp_path / "alignment-candidates.json"
    candidates.write_text('{"rows": [{"review_status": "pending"}]}', encoding="utf-8")
    log = tmp_path / DECISIONS_FILE
    append_decision(_mapping(), log)
    candidates.write_text('{"rows": []}', encoding="utf-8")        # export --force
    assert len(latest_decisions(read_log(log), SRC)) == 1


# ---------------------------------------------------------------- snapshot (provenance)
def test_snapshot_keeps_how_the_proposal_was_produced():
    row = {"concept_name": "Consent", "rank": 1, "dpv_name": "Consent", "score": 0.91,
           "lexical": 1.0, "semantic": 0.85, "proposed_relation": "skos:exactMatch",
           "justification": "Misma nocion.", "evidence_article": 8, "concept_articles": [8],
           "dpv_definition": "no se copia"}
    meta = {"embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "generated_at": "2026-09-30T22:10:03+00:00",
            "justification": {"llm_model": "gemini-3.1-flash-lite", "prompt_version": 2}}
    snap = snapshot_from_row(row, meta)
    assert snap["proposed_relation"] == "skos:exactMatch" and snap["evidence_article"] == 8
    assert snap["llm_model"] == "gemini-3.1-flash-lite" and snap["prompt_version"] == 2
    assert snap["embedding_model"].startswith("paraphrase-multilingual")
    assert "dpv_definition" not in snap
    d = _mapping(snapshot=snap)
    assert d.snapshot["score"] == 0.91


def test_snapshot_from_the_real_candidates_file(tmp_path):
    real = ROOT / "data/output/alignment-candidates.json"
    if not real.exists():
        pytest.skip("No hay alignment-candidates.json (Sprint 4)")
    data = json.loads(real.read_text(encoding="utf-8"))
    row = next(r for r in data["rows"] if r["origin"] == "ontology" and r["rank"] == 1)
    snap = snapshot_from_row(row, data["metadata"])
    assert snap["rank"] == 1 and snap["llm_model"] and snap["prompt_version"] == 2
    path = tmp_path / DECISIONS_FILE
    append_decision(_mapping(snapshot=snap), path)
    assert read_log(path)[0].snapshot == snap               # survives the JSON round trip


# ---------------------------------------------------------------- config + console
def test_review_settings_defaults_and_project_config():
    assert review_settings({}) == {"source_id": DEFAULT_SOURCE_ID,
                                   "decisions_file": "data/review/" + DECISIONS_FILE,
                                   "reviewer": ""}
    s = review_settings()
    assert s["source_id"] == "ontopriv+lopdp"
    assert decisions_path() == ROOT / "data/review" / DECISIONS_FILE


def test_decisions_folder_is_not_ignored_by_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/review" not in ignored


def test_console_summary_in_spanish(tmp_path):
    path = tmp_path / DECISIONS_FILE
    append_decision(_mapping(), path)
    append_decision(_mapping(decision=DECISION_REJECTED, relation=None, dpv_iri=DPV_RIGHT), path)
    text = render_console(read_log(path), SRC)
    assert "Decisiones vigentes: 2" in text
    assert "1 aprobadas, 1 descartadas" in text
    assert "Revisores: Elian" in text