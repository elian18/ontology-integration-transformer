"""S5-T04: human decisions on the DPV correspondences (approve with a type, reject, no match,
search outside the top-3, undo) and the counts the next tasks read."""
import json
from pathlib import Path

import pytest

from src.core.emit import PROFILE_IRI
from src.alignment.dpv_targets import DpvTarget, DpvTargets
from src.alignment.decisions import (read_log, make_decision, append_decision, DECISIONS_FILE,
                                     TARGET_CONCEPT, DECISION_APPROVED, DECISION_REJECTED,
                                     DECISION_DUPLICATE, DECISION_NO_MATCH)
from src.alignment.mapping_review import (
    mapping_concepts, mapping_decisions, concept_status, search_dpv, compatible_kinds,
    approve, reject, mark_no_match, undo, row_states, mapping_progress, next_pending,
    approved_mappings, ai_agreement, render_console,
    STATUS_PENDING, STATUS_ALIGNED, STATUS_NO_MATCH, ROW_PENDING,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = "ontopriv+lopdp"
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
DPV = "https://w3id.org/dpv#"
META = {"embedding_model": "m", "justification": {"llm_model": "gemini", "prompt_version": 2}}


def _rows(key, origin, name, cands, kinds=("class",), family=None, **kw):
    out = []
    for rank, (dpv, rel) in enumerate(cands, start=1):
        r = {"concept_key": key, "origin": origin, "concept_name": name, "concept_label": None,
             "concept_kinds": list(kinds), "concept_family": family, "concept_definition": None,
             "concept_articles": [], "duplicate_status": None, "duplicate_of": None,
             "duplicate_of_name": None, "duplicate_score": None, "duplicate_reason": None,
             "rank": rank, "dpv_iri": DPV + dpv, "dpv_name": dpv, "dpv_kind": "class",
             "score": 0.9 - rank / 10, "lexical": 0.5, "semantic": 0.5,
             "proposed_relation": rel, "justification": f"just {dpv}", "evidence_article": 8}
        r.update(kw)
        out.append(r)
    return out


def _data():
    rows = []
    rows += _rows(ONTO + "Consent", "ontology", "Consent",
                  [("Consent", "skos:exactMatch"), ("ExplicitlyExpressedConsent",
                                                    "skos:narrowMatch"),
                   ("ConsentRecord", "skos:relatedMatch")], family="Principles")
    rows += _rows(ONTO + "Turnover", "ontology", "Turnover",
                  [("Fee", "none"), ("Payment", "none"), ("Amount", "none")],
                  family="Terminology")
    rows += _rows(ONTO + "has_purpose", "ontology", "has_purpose",
                  [("hasPurpose", "skos:exactMatch"), ("hasContext", "none"),
                   ("hasData", "none")], kinds=("property",), family="Processing")
    # AI concepts: one approved as new, one rejected, one duplicate, one still pending
    for name in ("ImpactAssessment", "ExplicitConsent", "DataSubject", "Recidivism"):
        rows += _rows(f"ai:{name}", "ai", name, [("DPIA", "skos:closeMatch"),
                                                 ("RiskAssessment", "skos:relatedMatch"),
                                                 ("Assessment", "skos:broadMatch")],
                      duplicate_status="new")
    return {"metadata": META, "rows": rows}


def _concept_decisions(log):
    for key, decision, extra in (("ai:ImpactAssessment", DECISION_APPROVED, {"entity_kind": "class"}),
                                 ("ai:ExplicitConsent", DECISION_REJECTED, {}),
                                 ("ai:DataSubject", DECISION_DUPLICATE,
                                  {"same_as": ONTO + "Data_Owner"})):
        append_decision(make_decision(source_id=SRC, target=TARGET_CONCEPT, concept_key=key,
                                      decision=decision, reviewer="Elian", **extra), log)


def _setup(tmp_path):
    log = tmp_path / DECISIONS_FILE
    _concept_decisions(log)
    data = _data()
    concepts = mapping_concepts(data, read_log(log), SRC)
    return data, log, {c.name: c for c in concepts}, concepts


def _current(log):
    return mapping_decisions(read_log(log), SRC)


def _targets():
    t = [DpvTarget(iri=DPV + n, name=n, kind=k, label=l, definition=dfn)
         for n, k, l, dfn in (
             ("Consent", "class", "Consent", "Consent of the data subject."),
             ("Turnover", "class", "Turnover", None),
             ("FinancialStatus", "class", "Financial status", "Includes turnover of a company."),
             ("hasTurnover", "property", "has turnover", None))]
    return DpvTargets(dpv_path="dpv.ttl", targets=t)


# ---------------------------------------------------------------- who is reviewed
def test_only_approved_ai_concepts_plus_ontopriv_enter_the_review(tmp_path):
    _, _, by, concepts = _setup(tmp_path)
    assert [c.name for c in concepts] == ["ImpactAssessment", "Consent", "has_purpose",
                                          "Turnover"]   # AI first, then OntoPriv by family
    assert [c.family for c in concepts[1:]] == ["Principles", "Processing", "Terminology"]
    assert "ExplicitConsent" not in by and "DataSubject" not in by and "Recidivism" not in by
    assert by["ImpactAssessment"].iri == PROFILE_IRI + "#ImpactAssessment"
    assert by["Consent"].iri == ONTO + "Consent"
    assert [r["rank"] for r in by["Consent"].candidates] == [1, 2, 3]


# ---------------------------------------------------------------- approve / reject
def test_approve_with_the_type_the_person_chose(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    d = approve(by["Consent"], DPV + "Consent", "skos:closeMatch", reviewer="Elian",
                source_id=SRC, log_path=log, metadata=META)
    assert d.relation == "skos:closeMatch"
    assert d.snapshot["proposed_relation"] == "skos:exactMatch"     # what the AI said
    assert d.snapshot["found_by"] == "candidates" and d.snapshot["subject_iri"] == ONTO + "Consent"
    assert concept_status(by["Consent"], _current(log)) == STATUS_ALIGNED
    with pytest.raises(ValueError, match="Elige el tipo SKOS"):
        approve(by["Consent"], DPV + "Consent", None, reviewer="Elian", source_id=SRC,
                log_path=log)


def test_more_than_one_mapping_and_rejections(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    c = by["Consent"]
    approve(c, DPV + "Consent", "skos:exactMatch", reviewer="Elian", source_id=SRC, log_path=log)
    approve(c, DPV + "ConsentRecord", "skos:relatedMatch", reviewer="Elian", source_id=SRC,
            log_path=log)
    reject(c, DPV + "ExplicitlyExpressedConsent", reviewer="Elian", source_id=SRC, log_path=log)
    states = {s["row"]["dpv_name"]: (s["state"], s["relation"]) for s in row_states(c, _current(log))}
    assert states == {"Consent": ("approved", "skos:exactMatch"),
                      "ExplicitlyExpressedConsent": ("rejected", None),
                      "ConsentRecord": ("approved", "skos:relatedMatch")}


def test_only_rejections_keep_the_concept_pending(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    for r in by["Turnover"].candidates:
        reject(by["Turnover"], r["dpv_iri"], reviewer="Elian", source_id=SRC, log_path=log)
    assert concept_status(by["Turnover"], _current(log)) == STATUS_PENDING


# ---------------------------------------------------------------- no match
def test_no_match_and_its_rules(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    mark_no_match(by["Turnover"], reviewer="Elian", source_id=SRC, log_path=log,
                  current=_current(log), metadata=META)
    assert concept_status(by["Turnover"], _current(log)) == STATUS_NO_MATCH
    approve(by["Consent"], DPV + "Consent", "skos:exactMatch", reviewer="Elian", source_id=SRC,
            log_path=log)
    with pytest.raises(ValueError, match="ya tiene 1 correspondencia"):
        mark_no_match(by["Consent"], reviewer="Elian", source_id=SRC, log_path=log,
                      current=_current(log))


def test_approving_withdraws_a_previous_no_match(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    t = by["Turnover"]
    mark_no_match(t, reviewer="Elian", source_id=SRC, log_path=log, current=_current(log))
    approve(t, DPV + "Fee", "skos:relatedMatch", reviewer="Elian", source_id=SRC, log_path=log,
            current=_current(log))
    current = _current(log)
    assert concept_status(t, current) == STATUS_ALIGNED
    assert "" not in current[t.key]                         # the no match was withdrawn
    assert len(read_log(log)) == 3 + 3                       # 3 concept + no_match, pending, approve


# ---------------------------------------------------------------- search outside the top-3
def test_search_respects_the_kind(tmp_path):
    targets = _targets()
    assert compatible_kinds(["class"]) == {"class"}
    assert compatible_kinds(["class", "property"]) == {"class", "property"}
    assert [x.name for x in search_dpv(targets, "turnover", ["class"])] == \
        ["Turnover", "FinancialStatus"]                      # name first, then definition
    assert [x.name for x in search_dpv(targets, "turnover", ["property"])] == ["hasTurnover"]
    assert search_dpv(targets, "  ", ["class"]) == []


def test_approve_a_searched_term(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    targets = {x.name: x for x in _targets().targets}
    t = by["Turnover"]
    with pytest.raises(ValueError, match="no es candidato"):
        approve(t, DPV + "Turnover", "skos:exactMatch", reviewer="Elian", source_id=SRC,
                log_path=log)
    with pytest.raises(ValueError, match="tipo incompatible"):
        approve(t, DPV + "hasTurnover", "skos:exactMatch", reviewer="Elian", source_id=SRC,
                log_path=log, target=targets["hasTurnover"])
    d = approve(t, DPV + "Turnover", "skos:exactMatch", reviewer="Elian", source_id=SRC,
                log_path=log, target=targets["Turnover"])
    assert d.snapshot["found_by"] == "search" and d.snapshot["dpv_name"] == "Turnover"
    extra = [s for s in row_states(t, _current(log)) if s["found_by"] == "search"]
    assert extra[0]["dpv_name"] == "Turnover" and extra[0]["state"] == "approved"
    m = approved_mappings([t], _current(log))[0]
    assert (m["found_by"], m["dpv_name"], m["ai_relation"]) == ("search", "Turnover", None)


# ---------------------------------------------------------------- undo
def test_undo_returns_the_candidate_to_pending(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    c = by["Consent"]
    approve(c, DPV + "Consent", "skos:exactMatch", reviewer="Elian", source_id=SRC, log_path=log)
    undo(c, DPV + "Consent", reviewer="Elian", source_id=SRC, log_path=log)
    assert concept_status(c, _current(log)) == STATUS_PENDING
    assert row_states(c, _current(log))[0]["state"] == ROW_PENDING
    mark_no_match(c, reviewer="Elian", source_id=SRC, log_path=log, current=_current(log))
    undo(c, reviewer="Elian", source_id=SRC, log_path=log)
    assert concept_status(c, _current(log)) == STATUS_PENDING


# ---------------------------------------------------------------- progress and outputs
def test_progress_next_pending_and_agreement(tmp_path):
    _, log, by, concepts = _setup(tmp_path)
    approve(by["Consent"], DPV + "Consent", "skos:exactMatch", reviewer="Elian", source_id=SRC,
            log_path=log)                                    # same type as the AI
    approve(by["has_purpose"], DPV + "hasPurpose", "skos:closeMatch", reviewer="Elian",
            source_id=SRC, log_path=log)                     # person changed it
    mark_no_match(by["Turnover"], reviewer="Elian", source_id=SRC, log_path=log,
                  current=_current(log))                     # AI said "none" for all three
    current = _current(log)
    p = mapping_progress(concepts, current)
    assert (p["total"], p["reviewed"], p["pending"], p["aligned"], p["no_match"]) == (4, 3, 1, 2, 1)
    assert p["ai_pending"] == 1 and p["ontology_pending"] == 0
    assert p["relations"]["skos:exactMatch"] == 1 and p["relations"]["skos:closeMatch"] == 1
    assert next_pending(concepts, current) == "ai:ImpactAssessment"
    a = ai_agreement(concepts, current)
    assert (a["approved_with_ai_type"], a["same_type_as_ai"], a["type_agreement"]) == (2, 1, 0.5)
    assert (a["no_match"], a["no_match_ai_also_none"]) == (1, 1)
    text = render_console(concepts, current)
    assert "552" not in text and "Revisados: 3 (2 alineados, 1 sin correspondencia)" in text
    assert "1 de 2 (50.0 %)" in text


def test_approved_mappings_for_the_graph(tmp_path):
    _, log, by, concepts = _setup(tmp_path)
    approve(by["ImpactAssessment"], DPV + "DPIA", "skos:exactMatch", reviewer="Elian",
            source_id=SRC, log_path=log)
    m = approved_mappings(concepts, _current(log))
    assert m == [{
        "concept_key": "ai:ImpactAssessment", "subject_iri": PROFILE_IRI + "#ImpactAssessment",
        "origin": "ai", "concept_name": "ImpactAssessment", "dpv_iri": DPV + "DPIA",
        "dpv_name": "DPIA", "relation": "skos:exactMatch", "ai_relation": "skos:closeMatch",
        "found_by": "candidates", "justification": "just DPIA", "evidence_article": 8,
        "reviewer": "Elian", "decided_at": m[0]["decided_at"], "note": None}]


def test_other_input_decisions_do_not_mix(tmp_path):
    _, log, by, _ = _setup(tmp_path)
    append_decision(make_decision(source_id="ley-peru", target="mapping",
                                  concept_key=by["Consent"].key, dpv_iri=DPV + "Consent",
                                  decision=DECISION_APPROVED, relation="skos:exactMatch",
                                  reviewer="X"), log)
    assert concept_status(by["Consent"], _current(log)) == STATUS_PENDING


def test_real_files():
    cand = ROOT / "data/output/alignment-candidates.json"
    log = ROOT / "data/review" / DECISIONS_FILE
    if not cand.exists() or not log.exists():
        pytest.skip("Faltan los candidatos (Sprint 4) o el registro de decisiones (S5-T03)")
    data = json.loads(cand.read_text(encoding="utf-8"))
    records = read_log(log)
    concepts = mapping_concepts(data, records, SRC)
    onto = [c for c in concepts if c.origin == "ontology"]
    ai = [c for c in concepts if c.origin == "ai"]
    assert len(onto) == 527
    assert 0 < len(ai) <= 93
    assert all(c.iri.startswith(PROFILE_IRI + "#") for c in ai)
    assert all(len(c.candidates) == 3 for c in concepts)