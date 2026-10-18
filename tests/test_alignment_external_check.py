"""S6-T09: automatic check on an ontology from outside the base, with no manual labelling.

The concepts that inherited a validated decision are solved again as if they were new, with the
same steps the web uses (S6-T06), and compared with what the person decided."""
from __future__ import annotations

import hashlib
import json
import sys
import types

import pytest

from src.alignment import external_check as ec
from src.alignment import inherit as inh
from src.alignment.export import CANDIDATES_JSON
from src.alignment.external_check import (ABSTAIN_OK, CHECK_DIR, HIT, LOST_AI, LOST_RANK,
                                          LOST_THRESHOLD, RESULT_FILE, WRONG, check_workspace,
                                          compare, known_answers, render_console, summarize)
from src.alignment.inherit import inherit_for_workspace, load_base
from src.alignment.propose import STATUS_AI_NONE, STATUS_BELOW, STATUS_PROPOSED
from src.alignment.propose import propose_for_workspace
from src.workspace.store import open_ontology
from tests.test_alignment_propose import CFG, EX, TTL, Embed, FakeLLM, targets  # noqa: F401
from tests.test_alignment_write import DPV, _decide_all

CHECK_CFG = {**CFG, "auto": {**CFG["auto"], "threshold": 0.8}}


@pytest.fixture
def ws(tmp_path):
    """Workspace of the propose tests: Consent inherits from the base (same name)."""
    data, log = _decide_all(tmp_path)
    cand = tmp_path / "base.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    w = open_ontology("gdpr.ttl", TTL.encode("utf-8"),
                      cfg={"workspaces": {"dir": str(tmp_path / "ws")}})
    inherit_for_workspace(w, base=load_base(cand, log), use_embeddings=False)
    return w


def _known(ws):
    """Add three more concepts with a known answer: two with a DPV term, one without."""
    path = ws.work_dir / inh.RESULT_FILE
    result = json.loads(path.read_text(encoding="utf-8"))
    for name, mappings in (("PersonalData", [(DPV + "PersonalData", "skos:exactMatch")]),
                           ("DataController", [(DPV + "DataController", "skos:closeMatch")]),
                           ("Pizza", [])):
        result["inherited"].append({
            "key": EX + name, "name": name, "kinds": ["class"], "method": "same_name",
            "score": 1.0, "base_name": name, "status": "aligned" if mappings else "no_match",
            "mappings": [{"dpv_iri": iri, "dpv_name": iri.rsplit("#", 1)[-1], "relation": rel}
                         for iri, rel in mappings]})
    path.write_text(json.dumps(result), encoding="utf-8")
    return ws


def _verdicts(result):
    return {r["name"]: r["verdict"] for r in result["concepts"]}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ pure parts

def test_known_answers_come_from_the_inheritance():
    inherited = {"inherited": [
        {"key": "u#B", "name": "B", "method": "similar", "base_name": "b", "mappings": []},
        {"key": "u#A", "name": "A", "method": "same_name", "base_name": "a",
         "mappings": [{"dpv_iri": "d#X", "relation": "skos:exactMatch", "dpv_name": "X"},
                      {"dpv_iri": "d#Y", "relation": "skos:relatedMatch"}]}]}
    known = known_answers(inherited)
    assert [k["name"] for k in known] == ["A", "B"]
    assert known[0]["gold"] == {"d#X": "skos:exactMatch", "d#Y": "skos:relatedMatch"}
    assert known[0]["gold_names"] == {"d#X": "X", "d#Y": "Y"}
    assert known[1]["gold"] == {}


def _k(name, gold):
    return {"key": "u#" + name, "name": name, "method": "same_name", "base_name": name,
            "gold": gold, "gold_names": {iri: iri[2:] for iri in gold}}


def _e(name, iri, status, relation=None, score=0.9):
    return {"key": "u#" + name, "name": name, "dpv_iri": iri, "dpv_name": iri[2:],
            "score": score, "status": status, "relation": relation}


def test_compare_gives_one_verdict_per_concept():
    known = [_k("Hit", {"d#A": "skos:exactMatch"}), _k("Wrong", {"d#A": "skos:closeMatch"}),
             _k("Low", {"d#A": "skos:exactMatch"}), _k("NoneAI", {"d#A": "skos:exactMatch"}),
             _k("Rank3", {"d#C": "skos:broadMatch"}), _k("Missing", {"d#Z": "skos:exactMatch"}),
             _k("Empty", {}), _k("FalseAlarm", {})]
    order = {"Wrong": ("d#B", "d#A", "d#C")}            # 'Wrong': the right term was 2nd
    rows = [{"concept_key": "u#" + k["name"], "dpv_iri": iri, "rank": n}
            for k in known
            for n, iri in enumerate(order.get(k["name"], ("d#A", "d#B", "d#C")), start=1)]
    entries = [_e("Hit", "d#A", STATUS_PROPOSED, "skos:closeMatch"),
               _e("Wrong", "d#B", STATUS_PROPOSED, "skos:closeMatch"),
               _e("Low", "d#A", STATUS_BELOW, score=0.5),
               _e("NoneAI", "d#A", STATUS_AI_NONE),
               _e("Rank3", "d#A", STATUS_BELOW, score=0.5),
               _e("Missing", "d#A", STATUS_BELOW, score=0.5),
               _e("Empty", "d#A", STATUS_BELOW, score=0.5),
               _e("FalseAlarm", "d#A", STATUS_PROPOSED, "skos:exactMatch")]
    got = {r["name"]: (r["verdict"], r["gold_rank"], r["relation_ok"])
           for r in compare(known, {"rows": rows}, entries)}
    assert got == {"Hit": (HIT, 1, False), "Wrong": (WRONG, 2, None),
                   "Low": (LOST_THRESHOLD, 1, None), "NoneAI": (LOST_AI, 1, None),
                   "Rank3": (LOST_RANK, 3, None), "Missing": (LOST_RANK, None, None),
                   "Empty": (ABSTAIN_OK, None, None), "FalseAlarm": (WRONG, None, None)}


def test_summary_counts_precision_coverage_and_positions():
    rows = [{"name": n, "method": m, "gold": g, "gold_rank": rank, "verdict": v,
             "relation_ok": ok}
            for n, m, g, rank, v, ok in [
                ("A", "same_name", [1], 1, HIT, True),
                ("B", "same_name", [1], 1, HIT, False),
                ("C", "same_name", [1], 2, WRONG, None),
                ("D", "similar", [1], 1, LOST_THRESHOLD, None),
                ("E", "similar", [1], 7, LOST_RANK, None),
                ("F", "similar", [1], None, LOST_RANK, None),
                ("G", "same_name", [], None, ABSTAIN_OK, None),
                ("H", "same_name", [], None, WRONG, None)]]
    s = summarize(rows, threshold=0.75)
    a = s["all"]
    assert (a["concepts"], a["aligned"], a["no_match"]) == (8, 6, 2)
    assert (a["proposals"], a["correct"], a["precision"], a["coverage"]) == (4, 2, 0.5, 0.3333)
    assert (a["relation_typed"], a["relation_ok"], a["no_match_ok"]) == (2, 1, 1)
    assert a["hits"] == {"1": 3, "3": 4, "10": 5} and a["not_in_depth"] == 1
    assert s["by_method"]["same_name"]["precision"] == 0.5
    assert s["by_method"]["similar"]["coverage"] == 0.0


# ------------------------------------------------------------------ on a workspace

def test_check_solves_the_known_concepts_again(ws, targets):
    llm = FakeLLM()
    result = check_workspace(_known(ws), targets=targets, embed_fn=Embed(), llm=llm,
                             cfg=CHECK_CFG, sleep_fn=lambda s: None)
    assert _verdicts(result) == {"Consent": LOST_RANK, "DataController": HIT,
                                 "PersonalData": HIT, "Pizza": ABSTAIN_OK}
    rows = {r["name"]: r for r in result["concepts"]}
    assert rows["Consent"]["gold_rank"] is None              # its DPV terms are not in this DPV
    assert rows["PersonalData"]["relation_ok"] is False      # person: exact, AI: close
    assert rows["DataController"]["relation_ok"] is True
    a = result["summary"]["all"]
    assert (a["proposals"], a["correct"], a["coverage"], a["no_match_ok"]) == (2, 2, 0.6667, 1)
    assert result["summary"]["depth"] == 10 and result["ai"]["calls"] == 1 == llm.calls
    saved = json.loads((ws.work_dir / CHECK_DIR / CANDIDATES_JSON).read_text(encoding="utf-8"))
    ranked = [r for r in saved["rows"] if r["concept_name"] == "PersonalData"]
    assert len(ranked) == 5                     # every class of this DPV, not just the top 3
    assert json.loads((ws.work_dir / RESULT_FILE).read_text(encoding="utf-8"))["summary"] \
        == result["summary"]
    text = render_console(result)
    assert "Comprobación automática con una ontología externa (S6-T09)" in text
    assert "Propuestas: 2; correctas: 2 (precisión 100 %)" in text
    assert "Tipo SKOS igual al de la persona: 1 de 2 aciertos con tipo de la IA" in text
    assert "PersonalData: PersonalData (exactMatch) | 1.º PersonalData" in text


def test_second_run_reuses_its_candidates_and_the_ai_types(ws, targets):
    first = check_workspace(_known(ws), targets=targets, embed_fn=Embed(), llm=FakeLLM(),
                            cfg=CHECK_CFG, sleep_fn=lambda s: None)
    embed, llm = Embed(), FakeLLM()
    again = check_workspace(ws, targets=targets, embed_fn=embed, llm=llm, cfg=CHECK_CFG)
    assert embed.calls == 0 and llm.calls == 0
    assert again["summary"] == first["summary"]


def test_ai_none_is_reported_as_its_own_loss(ws, targets):
    result = check_workspace(_known(ws), targets=targets, embed_fn=Embed(),
                             llm=FakeLLM(relation="none"), cfg=CHECK_CFG,
                             sleep_fn=lambda s: None)
    v = _verdicts(result)
    assert v["PersonalData"] == v["DataController"] == LOST_AI
    assert "el correcto era el 1.º pero la IA dijo none: 2" in render_console(result)


def test_without_ai_nothing_is_typed(ws, targets):
    result = check_workspace(_known(ws), targets=targets, embed_fn=Embed(), use_llm=False,
                             cfg=CHECK_CFG)
    assert _verdicts(result)["PersonalData"] == HIT
    assert all(r["relation"] is None for r in result["concepts"])
    assert "Tipo SKOS: no se midió" in render_console(result)
    assert not result["ai"]["available"]


def test_the_user_alignment_is_not_touched(ws, targets):
    propose_for_workspace(ws, embed_fn=Embed(), llm=FakeLLM(), targets=targets, cfg=CFG,
                          sleep_fn=lambda s: None)
    files = [ws.work_dir / CANDIDATES_JSON, ws.work_dir / "proposals.json",
             ws.work_dir / inh.RESULT_FILE]
    before = [_sha(p) for p in files]
    check_workspace(ws, targets=targets, embed_fn=Embed(), llm=FakeLLM(), cfg=CHECK_CFG,
                    sleep_fn=lambda s: None)
    assert [_sha(p) for p in files] == before
    assert (ws.work_dir / CHECK_DIR / CANDIDATES_JSON).exists()


def test_needs_the_inheritance_and_some_known_concept(tmp_path, targets):
    w = open_ontology("gdpr.ttl", TTL.encode("utf-8"),
                      cfg={"workspaces": {"dir": str(tmp_path / "ws")}})
    with pytest.raises(ValueError, match="Primero hay que heredar"):
        check_workspace(w, targets=targets, embed_fn=Embed(), use_llm=False, cfg=CHECK_CFG)
    (w.work_dir / inh.RESULT_FILE).write_text(json.dumps({"inherited": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no heredó ningún concepto"):
        check_workspace(w, targets=targets, embed_fn=Embed(), use_llm=False, cfg=CHECK_CFG)


def test_cli(tmp_path, monkeypatch, capsys, ws, targets):
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()                                    # no real model in the tests
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.config.load_config",
                        lambda: {"workspaces": {"dir": str(tmp_path / "ws")}, **CHECK_CFG})
    monkeypatch.setattr("src.workspace.store.open_ontology", lambda *a, **k: _known(ws))
    monkeypatch.setattr(ec, "load_dpv_targets", lambda cfg=None: targets)
    monkeypatch.setattr(ec, "report_path", lambda w: tmp_path / "out" / "gdpr-external-check.json")
    source = tmp_path / "gdpr.ttl"
    source.write_text(TTL, encoding="utf-8")
    assert ec._run([str(source), "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Conceptos con respuesta conocida (heredados de la base): 4" in out
    assert "IA: no usada" in out and "gdpr-external-check.json" in out
    saved = json.loads((tmp_path / "out" / "gdpr-external-check.json").read_text(encoding="utf-8"))
    assert saved["summary"]["all"]["concepts"] == 4
    assert ec._run([str(tmp_path / "no.ttl")]) == 1


def test_report_goes_next_to_the_base_outputs(ws, tmp_path):
    assert ec.report_path(ws, tmp_path).name == "gdpr-external-check.json"