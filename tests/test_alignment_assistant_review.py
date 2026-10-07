"""S5-T08: assisted batch review. The assistant's proposals are NOT decisions: they reach the
log only when a person confirms them, in that person's name, with the proposal as provenance."""
import json
from pathlib import Path

import pytest

from src.alignment.decisions import (read_log, make_decision, append_decision, DECISIONS_FILE,
                                     TARGET_CONCEPT, DECISION_APPROVED, DECISION_NO_MATCH)
from src.alignment.mapping_review import (mapping_concepts, mapping_decisions, concept_status,
                                          approve, STATUS_ALIGNED, STATUS_NO_MATCH,
                                          STATUS_PENDING)
from src.alignment.assistant_review import (
    parse_proposals, load_proposals, proposals_path, open_proposals, proposal_rows, confirm,
    confirm_rows, review_provenance, family_counts, render_console,
    ACTION_ACCEPT, ACTION_NO_MATCH, ACTION_SKIP, CONFIRMED_NOTE, CHANGED_NOTE,
    CONFIRM_ACCEPTED, CONFIRM_CHANGED_TYPE, CONFIRM_CHANGED_TO_NO_MATCH, PROPOSALS_FILE,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = "ontopriv+lopdp"
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
DPV = "https://w3id.org/dpv#"
META = {"embedding_model": "m", "justification": {"llm_model": "gemini", "prompt_version": 2}}
PMETA = {"assistant": "asistente de prueba", "created_at": "2026-10-01"}


def _rows(key, name, cands, kinds=("class",), family=None):
    return [{"concept_key": key, "origin": "ontology", "concept_name": name,
             "concept_label": None, "concept_kinds": list(kinds), "concept_family": family,
             "concept_definition": None, "concept_articles": [], "duplicate_status": None,
             "duplicate_of": None, "duplicate_of_name": None, "duplicate_score": None,
             "duplicate_reason": None, "rank": rank, "dpv_iri": DPV + dpv, "dpv_name": dpv,
             "dpv_kind": "class", "score": 0.5, "lexical": 0.5, "semantic": 0.5,
             "proposed_relation": rel, "justification": "j", "evidence_article": 8}
            for rank, (dpv, rel) in enumerate(cands, start=1)]


def _data():
    rows = []
    rows += _rows(ONTO + "Consent", "Consent", [("Consent", "skos:exactMatch"),
                                                ("ConsentRecord", "skos:relatedMatch")],
                  family="Principles")
    rows += _rows(ONTO + "Turnover", "Turnover", [("Fee", "none"), ("Payment", "none")],
                  family="Terminology")
    rows += _rows(ONTO + "Sanction", "Sanction", [("Fine", "skos:narrowMatch")],
                  family="Sanctions")
    rows += _rows(ONTO + "Verifier", "Verifier", [("Notice", "none")], family="Verification")
    return {"metadata": META, "rows": rows}


def _proposals():
    return {"metadata": PMETA, "proposals": [
        {"concept_key": ONTO + "Consent", "concept_name": "Consent", "family": "Principles",
         "decision": "approve", "reason": "mismo concepto",
         "mappings": [{"dpv_iri": DPV + "Consent", "dpv_name": "Consent",
                       "relation": "skos:exactMatch", "found_by": "candidates"},
                      {"dpv_iri": DPV + "ConsentRecord", "dpv_name": "ConsentRecord",
                       "relation": "skos:relatedMatch", "found_by": "candidates"}]},
        {"concept_key": ONTO + "Turnover", "concept_name": "Turnover", "family": "Terminology",
         "decision": "approve", "reason": "volumen de negocio = turnover",
         "mappings": [{"dpv_iri": DPV + "Turnover", "dpv_name": "Turnover",
                       "dpv_kind": "class", "dpv_label": "Turnover",
                       "relation": "skos:exactMatch", "found_by": "search"}]},
        {"concept_key": ONTO + "Sanction", "concept_name": "Sanction", "family": "Sanctions",
         "decision": "no_match", "reason": "no hay sanciones en el DPV", "mappings": []},
        {"concept_key": ONTO + "Verifier", "concept_name": "Verifier", "family": "Verification",
         "decision": "no_match", "reason": "rama de verificacion", "mappings": []},
    ]}


def _setup(tmp_path):
    log = tmp_path / DECISIONS_FILE
    path = tmp_path / PROPOSALS_FILE
    path.write_text(json.dumps(_proposals()), encoding="utf-8")
    loaded = load_proposals(path)
    concepts = mapping_concepts(_data(), read_log(log), SRC)
    return log, loaded, {c.key: c for c in concepts}, concepts


def _current(log):
    return mapping_decisions(read_log(log), SRC)


def _kw(log):
    return dict(reviewer="Elian", source_id=SRC, log_path=log, metadata=META,
                proposals_metadata=PMETA)


def _by_key(loaded):
    return {p.concept_key: p for p in loaded["proposals"]}


# ---------------------------------------------------------------- the file
def test_parse_proposals_and_its_errors(tmp_path):
    props = parse_proposals(_proposals())
    assert [p.decision for p in props] == ["approve", "approve", "no_match", "no_match"]
    assert props[0].mapping(DPV + "ConsentRecord")["relation"] == "skos:relatedMatch"
    assert load_proposals(tmp_path / "none.json") is None
    bad = _proposals()
    bad["proposals"].append(dict(bad["proposals"][0]))
    with pytest.raises(ValueError, match="mas de una propuesta"):
        parse_proposals(bad)
    bad = _proposals()
    bad["proposals"][2]["mappings"] = bad["proposals"][0]["mappings"]
    with pytest.raises(ValueError, match="no debe tenerlas"):
        parse_proposals(bad)
    bad = _proposals()
    bad["proposals"][0]["mappings"][0]["relation"] = "owl:equivalentClass"
    with pytest.raises(ValueError, match="tipo SKOS no valido"):
        parse_proposals(bad)


def test_proposals_path_from_the_config():
    assert proposals_path({}) == ROOT / "data/review" / PROPOSALS_FILE
    assert proposals_path({"review": {"proposals_file": "x/p.json"}}) == ROOT / "x/p.json"


def test_only_concepts_still_pending_are_offered(tmp_path):
    log, loaded, concepts, _ = _setup(tmp_path)
    approve(concepts[ONTO + "Consent"], DPV + "Consent", "skos:closeMatch", reviewer="Elian",
            source_id=SRC, log_path=log)                    # the person decided it alone
    open_ = open_proposals(loaded["proposals"], concepts, _current(log))
    assert [p.concept_name for p in open_] == ["Turnover", "Sanction", "Verifier"]
    rows = proposal_rows(loaded["proposals"])
    assert [(r["concept_name"], r["dpv_name"]) for r in rows] == [
        ("Consent", "Consent"), ("Consent", "ConsentRecord"), ("Turnover", "Turnover"),
        ("Sanction", ""), ("Verifier", "")]
    assert family_counts(open_) == [("Sanctions", 0, 1), ("Terminology", 1, 0),
                                    ("Verification", 0, 1)]


# ---------------------------------------------------------------- confirming one proposal
def test_accepting_records_the_person_with_the_proposal_as_provenance(tmp_path):
    log, loaded, concepts, _ = _setup(tmp_path)
    c = concepts[ONTO + "Consent"]
    written = confirm(_by_key(loaded)[c.key], c, ACTION_ACCEPT, current=_current(log), **_kw(log))
    assert [(d.dpv_iri, d.relation) for d in written] == [
        (DPV + "Consent", "skos:exactMatch"), (DPV + "ConsentRecord", "skos:relatedMatch")]
    d = read_log(log)[0]
    assert d.reviewer == "Elian" and d.note == CONFIRMED_NOTE
    assert d.snapshot["proposed_by"] == "assistant"
    assert d.snapshot["assistant_confirmation"] == CONFIRM_ACCEPTED
    assert d.snapshot["proposal"]["reason"] == "mismo concepto"
    assert d.snapshot["proposal_assistant"] == "asistente de prueba"
    assert d.snapshot["proposed_relation"] == "skos:exactMatch"     # the Gemini type is kept
    assert concept_status(c, _current(log)) == STATUS_ALIGNED


def test_changing_the_type_is_recorded_as_a_change(tmp_path):
    log, loaded, concepts, _ = _setup(tmp_path)
    c = concepts[ONTO + "Consent"]
    confirm(_by_key(loaded)[c.key], c, ACTION_ACCEPT, accepted=[DPV + "Consent"],
            relations={DPV + "Consent": "skos:closeMatch"}, current=_current(log), **_kw(log))
    d = read_log(log)[0]
    assert d.relation == "skos:closeMatch"
    assert d.snapshot["assistant_confirmation"] == CONFIRM_CHANGED_TYPE
    assert d.note.startswith(CHANGED_NOTE) and "skos:exactMatch -> skos:closeMatch" in d.note
    assert d.snapshot["not_accepted_from_proposal"] == [DPV + "ConsentRecord"]   # partial
    prov = review_provenance(list(concepts.values()), _current(log), loaded["proposals"])
    assert prov["confirmed_from_assistant"] == 1 and prov["changed_by_person"] == 1


def test_no_match_accepted_or_chosen_instead_of_the_proposal(tmp_path):
    log, loaded, concepts, _ = _setup(tmp_path)
    props = _by_key(loaded)
    s = concepts[ONTO + "Sanction"]
    confirm(props[s.key], s, ACTION_ACCEPT, current=_current(log), **_kw(log))
    t = concepts[ONTO + "Turnover"]
    confirm(props[t.key], t, ACTION_NO_MATCH, current=_current(log), **_kw(log))
    first, second = read_log(log)
    assert first.decision == DECISION_NO_MATCH and first.note == CONFIRMED_NOTE
    assert second.decision == DECISION_NO_MATCH
    assert second.snapshot["assistant_confirmation"] == CONFIRM_CHANGED_TO_NO_MATCH
    assert concept_status(t, _current(log)) == STATUS_NO_MATCH


def test_searched_terms_are_still_checked(tmp_path):
    log, loaded, concepts, _ = _setup(tmp_path)
    t = concepts[ONTO + "Turnover"]
    confirm(_by_key(loaded)[t.key], t, ACTION_ACCEPT, current=_current(log), **_kw(log))
    d = read_log(log)[0]
    assert (d.dpv_iri, d.snapshot["found_by"], d.snapshot["dpv_kind"]) == (
        DPV + "Turnover", "search", "class")
    bad = _proposals()
    bad["proposals"][1]["mappings"][0]["dpv_kind"] = "property"        # a class -> property
    prop = parse_proposals(bad)[1]
    log2 = tmp_path / "other.jsonl"
    with pytest.raises(ValueError, match="incompatible"):
        confirm(prop, t, ACTION_ACCEPT, current={}, **_kw(log2))
    assert not log2.exists()


def test_skip_and_concepts_already_reviewed(tmp_path):
    log, loaded, concepts, _ = _setup(tmp_path)
    c = concepts[ONTO + "Consent"]
    assert confirm(_by_key(loaded)[c.key], c, ACTION_SKIP, **_kw(log)) == []
    assert not log.exists()
    approve(c, DPV + "Consent", "skos:exactMatch", reviewer="Elian", source_id=SRC,
            log_path=log)
    with pytest.raises(ValueError, match="ya fue revisado"):
        confirm(_by_key(loaded)[c.key], c, ACTION_ACCEPT, current=_current(log), **_kw(log))


# ---------------------------------------------------------------- the batch (the page table)
def test_confirm_rows_groups_by_concept(tmp_path):
    log, loaded, concepts, concept_list = _setup(tmp_path)
    rows = [
        {"concept_key": ONTO + "Consent", "dpv_iri": DPV + "Consent",
         "relation": "skos:exactMatch", "action": ACTION_ACCEPT},
        {"concept_key": ONTO + "Consent", "dpv_iri": DPV + "ConsentRecord",
         "relation": "skos:relatedMatch", "action": ACTION_NO_MATCH},       # contradictory
        {"concept_key": ONTO + "Turnover", "dpv_iri": DPV + "Turnover",
         "relation": "skos:exactMatch", "action": ACTION_ACCEPT},
        {"concept_key": ONTO + "Sanction", "dpv_iri": "", "relation": None,
         "action": ACTION_ACCEPT},
        {"concept_key": ONTO + "Verifier", "dpv_iri": "", "relation": None,
         "action": ACTION_SKIP},
    ]
    with pytest.raises(ValueError, match="revisor"):
        confirm_rows(rows, _by_key(loaded), concepts, **{**_kw(log), "reviewer": " "})
    summary = confirm_rows(rows, _by_key(loaded), concepts, current=_current(log), **_kw(log))
    assert (summary["concepts"], summary["decisions"], summary["skipped"]) == (2, 2, 1)
    assert (summary["accepted"], summary["changed"]) == (2, 0)
    assert summary["errors"][0][0] == "Consent" and "Sin correspondencia" in summary["errors"][0][1]
    current = _current(log)
    status = {c.name: concept_status(c, current) for c in concept_list}
    assert status == {"Consent": STATUS_PENDING, "Turnover": STATUS_ALIGNED,
                      "Sanction": STATUS_NO_MATCH, "Verifier": STATUS_PENDING}
    prov = review_provenance(concept_list, current, loaded["proposals"])
    assert prov == {"reviewed_by_person": 0, "confirmed_from_assistant": 2,
                    "accepted_as_proposed": 2, "changed_by_person": 0, "open_proposals": 2}
    text = render_console(open_proposals(loaded["proposals"], concepts, current), prov, PMETA)
    assert "no son decisiones" in text and "Abiertas (concepto aun por validar): 2" in text


def test_person_decisions_count_as_their_own(tmp_path):
    log, loaded, concepts, concept_list = _setup(tmp_path)
    append_decision(make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key="ai:X",
                                  decision=DECISION_APPROVED, entity_kind="class",
                                  reviewer="Elian"), log)
    approve(concepts[ONTO + "Consent"], DPV + "Consent", "skos:exactMatch", reviewer="Elian",
            source_id=SRC, log_path=log)
    prov = review_provenance(concept_list, _current(log))
    assert prov["reviewed_by_person"] == 1 and prov["confirmed_from_assistant"] == 0
    assert prov["open_proposals"] is None


# ---------------------------------------------------------------- the real proposals file
def test_real_proposals_file():
    path = ROOT / "data/review" / PROPOSALS_FILE
    if not path.exists():
        pytest.skip("no hay propuestas del asistente")
    loaded = load_proposals(path)
    meta, props = loaded["metadata"], loaded["proposals"]
    assert meta["proposed_by"] == "assistant" and "NO DECISIONES" in meta["status"]
    assert meta["counts"]["proposals"] == len(props)
    assert meta["counts"]["mappings"] == sum(len(p.mappings) for p in props)
    assert all(p.reason for p in props)
    cand = ROOT / "data/output/alignment-candidates.json"
    if not cand.exists():
        pytest.skip("sin el archivo de candidatos (py -m src.alignment.export)")
    from src.alignment.export import load_candidate_file
    from src.alignment.dpv_targets import build_dpv_targets, COMPATIBLE_TARGETS
    from src.ingest.dpv_loader import load_dpv
    from src.alignment.decisions import decisions_path
    records = read_log(decisions_path())                  # the AI concepts approved in S5-T03
    concepts = {c.key: c for c in mapping_concepts(load_candidate_file(cand), records, SRC)}
    targets = {t.iri: t for t in build_dpv_targets(load_dpv(ROOT / "vocab/dpv.ttl")).targets}
    for p in props:
        c = concepts[p.concept_key]                       # every proposal is a real concept
        for m in p.mappings:
            t = targets[m["dpv_iri"]]                     # every term exists in the DPV 2.3
            assert m["dpv_kind"] == t.kind
            assert any(t.kind in COMPATIBLE_TARGETS.get(k, set()) for k in c.kinds)
            assert (c.candidate(m["dpv_iri"]) is not None) == (m["found_by"] == "candidates")