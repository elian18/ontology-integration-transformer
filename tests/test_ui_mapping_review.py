"""S5-T05: the review card of the "Alineación DPV" page (service logic + the page with AppTest).
S5-T08: the tab that confirms the assistant proposals in batches."""
import importlib
import json
import sys
from pathlib import Path

import pytest

from app.services import mapping_review as mr
from src.alignment.dpv_targets import DpvTarget, DpvTargets
from src.alignment.decisions import (read_log, make_decision, append_decision, review_settings,
                                     DECISIONS_FILE, TARGET_CONCEPT, DECISION_APPROVED)
from src.alignment.assistant_review import PROPOSALS_FILE

ROOT = Path(__file__).resolve().parents[1]
SRC = "ontopriv+lopdp"
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
DPV = "https://w3id.org/dpv#"
CFG = {"review": {"source_id": SRC, "reviewer": "Elian"}}


def _rows(key, origin, name, cands, family=None, kinds=("class",), **kw):
    out = []
    for rank, (dpv, rel) in enumerate(cands, start=1):
        r = {"concept_key": key, "origin": origin, "concept_name": name,
             "concept_label": kw.pop("label", None) if rank == 1 else None,
             "concept_kinds": list(kinds), "concept_family": family,
             "concept_definition": None, "concept_articles": [8], "duplicate_status": None,
             "duplicate_of": None, "duplicate_of_name": None, "duplicate_score": None,
             "duplicate_reason": None, "rank": rank, "dpv_iri": DPV + dpv, "dpv_name": dpv,
             "dpv_label": dpv, "dpv_kind": "class", "dpv_definition": f"def {dpv}",
             "dpv_parents": ["Parent"], "score": 0.9 - rank / 10, "lexical": 0.5,
             "semantic": 0.5, "proposed_relation": rel, "justification": f"just {dpv}",
             "evidence_article": 8}
        r.update(kw)
        out.append(r)
    return out


def _files(tmp_path):
    rows = []
    rows += _rows(ONTO + "Consent", "ontology", "Consent",
                  [("Consent", "skos:exactMatch"), ("ConsentRecord", "skos:relatedMatch"),
                   ("Notice", "none")], family="Principles", label="consentimiento")
    rows += _rows(ONTO + "Turnover", "ontology", "Turnover",
                  [("Fee", "none"), ("Payment", "none"), ("Amount", "none")],
                  family="Terminology")
    rows += _rows("ai:ImpactAssessment", "ai", "ImpactAssessment",
                  [("DPIA", "skos:closeMatch"), ("RiskAssessment", "skos:relatedMatch"),
                   ("Assessment", "skos:broadMatch")], duplicate_status="new")
    cand = tmp_path / "alignment-candidates.json"
    cand.write_text(json.dumps({"metadata": {"embedding_model": "m"}, "counts": {"rows": 9},
                                "rows": rows}), encoding="utf-8")
    log = tmp_path / "review" / DECISIONS_FILE
    append_decision(make_decision(source_id=SRC, target=TARGET_CONCEPT,
                                  concept_key="ai:ImpactAssessment", decision=DECISION_APPROVED,
                                  entity_kind="class", reviewer="Elian"), log)
    dpv = tmp_path / "dpv.ttl"
    dpv.write_text("""
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:Turnover a rdfs:Class , skos:Concept ; skos:prefLabel "Turnover"@en ;
    skos:definition "Income of a company."@en .
dpv:hasTurnover a rdf:Property ; skos:prefLabel "has turnover"@en .
""", encoding="utf-8")
    props = tmp_path / "review" / PROPOSALS_FILE
    props.write_text(json.dumps(_proposals()), encoding="utf-8")
    return cand, log, dpv


def _proposals():
    """Assistant proposals for the three concepts (S5-T08)."""
    return {"metadata": {"assistant": "asistente de prueba", "created_at": "2026-10-01"},
            "proposals": [
        {"concept_key": "ai:ImpactAssessment", "concept_name": "ImpactAssessment",
         "concept_label": "Evaluacion de impacto", "family": None, "decision": "approve",
         "reason": "la EIPD es una evaluacion de impacto",
         "mappings": [{"dpv_iri": DPV + "DPIA", "dpv_name": "DPIA", "dpv_definition": "def",
                       "relation": "skos:broadMatch", "found_by": "candidates"}]},
        {"concept_key": ONTO + "Consent", "concept_name": "Consent", "family": "Principles",
         "decision": "approve", "reason": "mismo concepto",
         "mappings": [{"dpv_iri": DPV + "Consent", "dpv_name": "Consent",
                       "relation": "skos:exactMatch", "found_by": "candidates"},
                      {"dpv_iri": DPV + "ConsentRecord", "dpv_name": "ConsentRecord",
                       "relation": "skos:relatedMatch", "found_by": "candidates"}]},
        {"concept_key": ONTO + "Turnover", "concept_name": "Turnover", "family": "Terminology",
         "decision": "no_match", "reason": "no hay volumen de negocio en el DPV",
         "mappings": []}]}


def _state(tmp_path):
    cand, log, _ = _files(tmp_path)
    return mr.load(candidates_path=cand, log_path=log, cfg=CFG), cand, log


def _targets():
    return DpvTargets(dpv_path="x", targets=[
        DpvTarget(iri=DPV + "Turnover", name="Turnover", kind="class", label="Turnover"),
        DpvTarget(iri=DPV + "Fee", name="Fee", kind="class", label="Fee"),
        DpvTarget(iri=DPV + "hasTurnover", name="hasTurnover", kind="property",
                  label="has turnover")])


# ---------------------------------------------------------------- service
def test_load_and_filters(tmp_path):
    assert mr.load(candidates_path=tmp_path / "no.json", log_path=tmp_path / "l", cfg=CFG) is None
    state, _, _ = _state(tmp_path)
    assert [c.name for c in state["concepts"]] == ["ImpactAssessment", "Consent", "Turnover"]
    assert state["progress"]["pending"] == 3
    assert mr.families(state) == ["Principles", "Terminology"]
    names = lambda cs: [c.name for c in cs]
    assert names(mr.filter_concepts(state, origin="ai")) == ["ImpactAssessment"]
    assert names(mr.filter_concepts(state, family="Terminology")) == ["Turnover"]
    assert names(mr.filter_concepts(state, query="consentimiento")) == ["Consent"]
    assert names(mr.filter_concepts(state, query="dpia")) == ["ImpactAssessment"]   # DPV name


def test_labels_for_the_page(tmp_path):
    state, _, _ = _state(tmp_path)
    opts = mr.relation_options()
    assert list(opts.values()) == ["skos:exactMatch", "skos:closeMatch", "skos:broadMatch",
                                   "skos:narrowMatch", "skos:relatedMatch"]
    assert "closeMatch · casi equivalente" in opts
    assert mr.relation_text("none") == "sin correspondencia"
    assert mr.relation_text(None) == "sin tipo"
    label = mr.option_label(state, state["by_key"][ONTO + "Consent"])
    assert label == "Consent · consentimiento  (Principles)  [por validar]"


def test_cards_show_the_ai_suggestion_without_choosing_it(tmp_path):
    state, _, _ = _state(tmp_path)
    cards = mr.candidate_cards(state, ONTO + "Consent")
    assert [c["dpv_name"] for c in cards] == ["Consent", "ConsentRecord", "Notice"]
    assert cards[0]["ai_text"] == "exactMatch · equivalente"
    assert cards[0]["relation"] is None and cards[0]["state"] == "pending"
    assert cards[2]["ai_text"] == "sin correspondencia"


def test_actions_write_the_log(tmp_path):
    state, cand, log = _state(tmp_path)
    key = ONTO + "Consent"
    ok, msg = mr.apply_approve(state, key, DPV + "Consent", "skos:closeMatch", "Elian")
    assert ok and "closeMatch" in msg
    ok, msg = mr.apply_reject(state, key, DPV + "Notice", "Elian")
    assert ok
    state = mr.load(candidates_path=cand, log_path=log, cfg=CFG)
    cards = {c["dpv_name"]: (c["state_label"], c["relation"])
             for c in mr.candidate_cards(state, key)}
    assert cards == {"Consent": ("aprobado", "skos:closeMatch"),
                     "ConsentRecord": ("por validar", None), "Notice": ("descartado", None)}
    assert mr.status_of(state, key) == "aligned"
    ok, msg = mr.apply_no_match(state, key, "Elian")
    assert not ok and "ya tiene 1 correspondencia" in msg       # error in Spanish, no crash


def test_no_match_undo_and_next_pending(tmp_path):
    state, cand, log = _state(tmp_path)
    assert mr.next_pending(state) == "ai:ImpactAssessment"
    ok, _ = mr.apply_no_match(state, ONTO + "Turnover", "Elian", note="no hay en el DPV")
    assert ok
    state = mr.load(candidates_path=cand, log_path=log, cfg=CFG)
    assert mr.status_of(state, ONTO + "Turnover") == "no_match"
    assert mr.next_pending(state, after_key=ONTO + "Consent") == "ai:ImpactAssessment"
    ok, msg = mr.apply_undo(state, ONTO + "Turnover", "Elian")
    assert ok and "sin correspondencia" in msg
    state = mr.load(candidates_path=cand, log_path=log, cfg=CFG)
    assert mr.status_of(state, ONTO + "Turnover") == "pending"


def test_search_excludes_own_candidates_and_wrong_kinds(tmp_path):
    state, cand, log = _state(tmp_path)
    hits = mr.search(state, _targets(), ONTO + "Turnover", "f")
    assert [t.name for t in hits] == []                 # Fee is already a candidate
    hits = mr.search(state, _targets(), ONTO + "Turnover", "turnover")
    assert [t.name for t in hits] == ["Turnover"]       # hasTurnover is a property
    ok, _ = mr.apply_approve(state, ONTO + "Turnover", DPV + "Turnover", "skos:exactMatch",
                             "Elian", target=hits[0])
    assert ok
    state = mr.load(candidates_path=cand, log_path=log, cfg=CFG)
    extra = [c for c in mr.candidate_cards(state, ONTO + "Turnover") if c["found_by"] == "search"]
    assert extra[0]["dpv_name"] == "Turnover" and extra[0]["state"] == "approved"


def test_dpv_targets_load_from_a_file(tmp_path):
    _, _, dpv = _files(tmp_path)
    targets = mr.load_dpv_targets(dpv)
    assert {t.name for t in targets.targets} == {"Turnover", "hasTurnover"}
    assert mr.load_dpv_targets(tmp_path / "none.ttl") is None


def test_default_paths_follow_the_config():
    paths = mr.default_paths()
    assert paths["log"] == ROOT / "data/review" / DECISIONS_FILE
    assert paths["dpv"] == ROOT / "vocab/dpv.ttl"
    assert paths["proposals"] == ROOT / "data/review" / PROPOSALS_FILE


# ---------------------------------------------------------------- assistant proposals (S5-T08)
def _assist(tmp_path):
    state, cand, log = _state(tmp_path)
    return state, mr.load_assistant(state, tmp_path / "review" / PROPOSALS_FILE), cand, log


def test_assistant_table_per_family(tmp_path):
    state, assist, _, _ = _assist(tmp_path)
    assert mr.load_assistant(state, tmp_path / "none.json") is None
    assert mr.proposal_families(assist) == [("Conceptos nuevos de la IA", 1), ("Principles", 1),
                                            ("Terminology", 1)]
    rows = mr.proposal_table(state, assist, "Principles")
    assert [(r["Propuesta del asistente"], r["Tipo SKOS"], r["Acción"]) for r in rows] == [
        ("dpv:Consent", "exactMatch · equivalente", "Omitir"),          # nothing preselected
        ("dpv:ConsentRecord", "relatedMatch · relacionado", "Omitir")]
    assert all(r["Acción"] == "Aceptar" for r in mr.proposal_table(state, assist, "Principles",
                                                                   prefill=True))
    none = mr.proposal_table(state, assist, "Terminology")[0]
    assert none["Propuesta del asistente"] == mr.NO_MATCH_TEXT and none["Tipo SKOS"] is None
    assert mr.count_actions(rows) == {"Aceptar": 0, "Sin correspondencia": 0, "Omitir": 2}


def test_confirm_table_writes_in_the_reviewer_name(tmp_path):
    state, assist, cand, log = _assist(tmp_path)
    rows = mr.proposal_table(state, assist, "Principles")
    ok, msg, _ = mr.confirm_table(state, assist, rows, "Elian")
    assert not ok and "No hay filas" in msg
    rows[0]["Acción"], rows[0]["Tipo SKOS"] = "Aceptar", None
    ok, msg, _ = mr.confirm_table(state, assist, rows, "Elian")
    assert not ok and "Elige el tipo SKOS" in msg and "Consent" in msg
    rows[0]["Tipo SKOS"] = "closeMatch · casi equivalente"              # the person changes it
    ok, msg, summary = mr.confirm_table(state, assist, rows, "Elian")
    assert ok and "1 con cambios" in msg and summary["decisions"] == 1
    mapping = [d for d in read_log(log) if d.target == "mapping"]
    assert (mapping[0].reviewer, mapping[0].relation) == ("Elian", "skos:closeMatch")
    assert mapping[0].snapshot["not_accepted_from_proposal"] == [DPV + "ConsentRecord"]
    state = mr.load(candidates_path=cand, log_path=log, cfg=CFG)
    assist = mr.load_assistant(state, tmp_path / "review" / PROPOSALS_FILE)
    assert [p.concept_name for p in assist["open"]] == ["ImpactAssessment", "Turnover"]
    rows = mr.proposal_table(state, assist, "Terminology", prefill=True)
    ok, msg, _ = mr.confirm_table(state, assist, rows, "Elian")
    assert ok and "1 tal como se propusieron" in msg
    state = mr.load(candidates_path=cand, log_path=log, cfg=CFG)
    assert mr.status_of(state, ONTO + "Turnover") == "no_match"
    assert mr.load_assistant(state, tmp_path / "review" / PROPOSALS_FILE)["provenance"][
        "confirmed_from_assistant"] == 2


# ---------------------------------------------------------------- the page (AppTest)
@pytest.fixture
def page(tmp_path, monkeypatch):
    """Run the real view with both services pointed at temporary files."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    app_dir = str(ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    # The view imports "services.*" (app/ on sys.path, as app/app.py does): patch those modules.
    view_mr = importlib.import_module("services.mapping_review")
    view_al = importlib.import_module("services.alignment")
    cand, log, dpv = _files(tmp_path)
    props = tmp_path / "review" / PROPOSALS_FILE
    monkeypatch.setattr(view_mr, "default_paths",
                        lambda: {"candidates": cand, "log": log, "dpv": dpv, "proposals": props})
    monkeypatch.setattr(view_al, "_output_dir", lambda: tmp_path)
    at = AppTest.from_file(str(ROOT / "app/views/alignment.py"), default_timeout=30)
    return at, log


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_page_shows_the_first_pending_concept_with_empty_types(page):
    at, _ = page
    at.run()
    assert not at.exception
    assert at.header[0].value == "Alineación con el DPV"
    assert at.subheader[0].value == "ImpactAssessment"            # approved AI concept first
    assert at.metric[0].value == "3"
    assert at.text_input(key="reviewer").value == review_settings()["reviewer"]
    types = [s for s in at.selectbox if s.key and s.key.startswith("rel_")]
    assert len(types) == 3 and all(s.value is None for s in types)  # no preselection
    approve = [b for b in at.button if b.label == "Aprobar"]
    assert all(b.disabled for b in approve)                         # until a type is chosen


def test_approve_from_the_page(page):
    at, log = page
    at.run()
    at.selectbox(key="rel_ai:ImpactAssessment_0").set_value("exactMatch · equivalente").run()
    approve = [b for b in at.button if b.label == "Aprobar"][0]
    assert not approve.disabled
    approve.click().run()
    assert not at.exception
    d = [x for x in read_log(log) if x.target == "mapping"][0]
    assert (d.concept_key, d.dpv_iri, d.relation) == ("ai:ImpactAssessment", DPV + "DPIA",
                                                      "skos:exactMatch")
    assert d.snapshot["proposed_relation"] == "skos:closeMatch"     # AI kept as evidence
    assert at.subheader[0].value == "ImpactAssessment"              # stays on the card
    assert "Aprobado" in at.success[0].value
    assert at.metric[1].value == "1"                                 # alineados


def test_no_match_from_the_page_moves_on(page):
    at, log = page
    at.run()
    _button(at, "Sin correspondencia en el DPV").click().run()
    assert not at.exception
    d = [x for x in read_log(log) if x.target == "mapping"][0]
    assert (d.concept_key, d.decision) == ("ai:ImpactAssessment", "no_match")
    assert at.subheader[0].value == "Consent"                        # next pending


def test_search_and_approve_from_the_page(page):
    at, log = page
    at.run()
    at.selectbox[2].set_value("Terminology").run()                   # Familia
    assert at.subheader[0].value == "Turnover"
    at.text_input(key=f"q_{ONTO}Turnover").input("turnover").run()
    assert at.selectbox(key=f"hit_{ONTO}Turnover").value == DPV + "Turnover"
    at.selectbox(key=f"srel_{ONTO}Turnover").set_value("exactMatch · equivalente").run()
    _button(at, "Aprobar este término").click().run()
    assert not at.exception
    d = [x for x in read_log(log) if x.target == "mapping"][0]
    assert (d.dpv_iri, d.snapshot["found_by"]) == (DPV + "Turnover", "search")


def test_table_tab_still_shows_the_sprint4_table(page):
    at, _ = page
    at.run()
    assert len(at.tabs) == 4
    assert at.tabs[3].label == "Tabla de candidatos"
    assert any("fila(s) · tipo propuesto" in c.value for c in at.caption)


def test_assistant_tab_shows_proposals_without_recording_them(page):
    at, log = page
    at.run()
    assert not at.exception
    assert at.tabs[1].label == "Confirmar propuestas del asistente"
    assert any("no decisiones" in w.value for w in at.warning)
    assert at.selectbox(key="ap_family").value == "Conceptos nuevos de la IA (1)"
    assert at.button(key="ap_confirm").disabled                      # all rows start "Omitir"
    assert [d.target for d in read_log(log)] == ["concept"]          # nothing written


def test_confirm_a_family_from_the_page(page):
    at, log = page
    at.run()
    at.checkbox(key="ap_prefill_Conceptos nuevos de la IA").check().run()
    assert not at.button(key="ap_confirm").disabled
    at.button(key="ap_confirm").click().run()
    assert not at.exception
    d = [x for x in read_log(log) if x.target == "mapping"][0]
    assert (d.concept_key, d.dpv_iri, d.relation) == ("ai:ImpactAssessment", DPV + "DPIA",
                                                      "skos:broadMatch")
    assert d.reviewer == review_settings()["reviewer"]
    assert d.note == "propuesta del asistente confirmada"
    assert d.snapshot["proposed_by"] == "assistant"
    assert "Registradas 1 decisión(es)" in at.success[0].value
    assert at.metric[6].value == "1"                         # confirmados desde propuestas
    assert at.selectbox(key="ap_family").value == "Principles (1)"