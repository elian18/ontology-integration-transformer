"""S6-T06: automatic proposals (best DPV candidate above the calibrated threshold, typed by the
AI when it is available) for the concepts of an uploaded ontology that inherited nothing."""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.alignment import inherit as inh
from src.alignment import propose as prop
from src.alignment.export import CANDIDATES_JSON
from src.alignment.inherit import inherit_for_workspace, load_base
from src.alignment.dpv_targets import build_dpv_targets
from src.alignment.propose import (STATUS_AI_NONE, STATUS_BELOW, STATUS_PROPOSED, decide,
                                   mark_for_ai, propose_for_workspace, render_console)
from src.ingest.dpv_loader import load_dpv
from src.workspace.store import DONE, open_ontology
from tests.test_alignment_export import _DPV
from tests.test_alignment_write import _decide_all

EX = "http://example.org/gdpr#"
TTL = f"""
@prefix ex: <{EX}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/gdpr> a owl:Ontology .
ex:Consent a owl:Class .
ex:PersonalData a owl:Class ; rdfs:comment "Information about an identified person."@en .
ex:DataController a owl:Class .
ex:Pizza a owl:Class .
"""
CFG = {"auto": {"threshold": 0.6, "use_llm": True, "llm_max_calls": 5},
       "alignment": {"top_k": 3, "weight_lexical": 0.4, "weight_semantic": 0.6,
                     "justify_batch_size": 15, "justify_pause_seconds": 0}}


class Embed:
    """Same direction for texts about personal data / controllers, others elsewhere."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.zeros(16)
            low = t.lower()
            if "personal" in low:
                v[0] = 1.0
            elif "controller" in low or "decides" in low:
                v[1] = 1.0
            else:
                v[2 + sum(map(ord, low)) % 14] = 1.0
            out.append(v)
        return out


class FakeLLM:
    """Answers every candidate in the prompt with one relation (or raises an error)."""

    def __init__(self, relation="closeMatch", error=None):
        self.relation, self.error, self.calls = relation, error, 0

    def ask(self, prompt, context="", max_tokens=1024):
        self.calls += 1
        if self.error:
            raise Exception(self.error)
        lines, cid = [], None
        for line in context.splitlines():
            if m := re.match(r"=== (C\d+) ===", line):
                cid = m.group(1)
            elif line.startswith("- ") and cid:
                lines.append(json.dumps({"concept": cid, "dpv": line[2:].split(" | ")[0],
                                         "relation": self.relation,
                                         "justification": "Ambas definiciones coinciden."}))
        return "\n".join(lines)


@pytest.fixture
def targets(tmp_path):
    path = tmp_path / "dpv.ttl"
    path.write_text(_DPV, encoding="utf-8")
    return build_dpv_targets(load_dpv(path))


@pytest.fixture
def ws(tmp_path):
    """Workspace whose inheritance is done: Consent inherits (same name), the rest does not."""
    data, log = _decide_all(tmp_path)
    cand = tmp_path / "base.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    w = open_ontology("gdpr.ttl", TTL.encode("utf-8"),
                      cfg={"workspaces": {"dir": str(tmp_path / "ws")}})
    inherit_for_workspace(w, base=load_base(cand, log), use_embeddings=False)
    return w


def _by_name(result):
    return {e["name"]: e for e in result["proposals"]}


# ------------------------------------------------------------------ pure decisions

def test_decide_statuses():
    rows = [{"concept_key": "a", "concept_name": "A", "rank": 1, "score": 0.9,
             "dpv_iri": "d#X", "proposed_relation": "skos:closeMatch", "justification": "j"},
            {"concept_key": "b", "concept_name": "B", "rank": 1, "score": 0.9,
             "dpv_iri": "d#Y", "proposed_relation": "none"},
            {"concept_key": "c", "concept_name": "C", "rank": 1, "score": 0.4,
             "dpv_iri": "d#Z", "proposed_relation": None},
            {"concept_key": "d", "concept_name": "D", "rank": 1, "score": 0.8,
             "dpv_iri": "d#W", "proposed_relation": None},
            {"concept_key": "a", "concept_name": "A", "rank": 2, "score": 0.95, "dpv_iri": "d#Q"}]
    got = {e["name"]: (e["status"], e["relation"]) for e in decide({"rows": rows}, 0.75)}
    assert got == {"A": (STATUS_PROPOSED, "skos:closeMatch"), "B": (STATUS_AI_NONE, None),
                   "C": (STATUS_BELOW, None), "D": (STATUS_PROPOSED, None)}


def test_only_rank_one_above_threshold_goes_to_the_ai():
    data = {"rows": [{"rank": 1, "score": 0.8}, {"rank": 2, "score": 0.9},
                     {"rank": 1, "score": 0.5}]}
    assert mark_for_ai(data, 0.75) == 1
    assert [r["needs_justification"] for r in data["rows"]] == [True, False, False]


# ------------------------------------------------------------------ the workspace run

def test_full_run_with_the_ai(ws, targets):
    llm = FakeLLM()
    result = propose_for_workspace(ws, embed_fn=Embed(), llm=llm, targets=targets, cfg=CFG,
                                   sleep_fn=lambda s: None)
    by = _by_name(result)
    assert "Consent" not in by                                   # inherited in S6-T05
    assert set(by) == {"PersonalData", "DataController", "Pizza"}
    assert by["PersonalData"]["status"] == STATUS_PROPOSED
    assert by["PersonalData"]["dpv_name"] == "PersonalData"
    assert by["PersonalData"]["relation"] == "skos:closeMatch"
    assert by["PersonalData"]["validated"] is False and by["PersonalData"]["justification"]
    assert by["Pizza"]["status"] == STATUS_BELOW and by["Pizza"]["relation"] is None
    s = result["summary"]
    assert s["proposed"] == s["proposed_typed"] and s["ai"]["available"] and llm.calls == 1
    assert ws.step("candidates")["status"] == DONE and ws.step("propose")["status"] == DONE
    assert (ws.work_dir / CANDIDATES_JSON).exists()              # inside the workspace
    assert prop.load_result(ws)["summary"] == s


def test_rerun_reuses_candidates_and_ai_answers(ws, targets):
    embed, llm = Embed(), FakeLLM()
    first = propose_for_workspace(ws, embed_fn=embed, llm=llm, targets=targets, cfg=CFG,
                                  sleep_fn=lambda s: None)
    calls = (embed.calls, llm.calls)
    again = propose_for_workspace(ws, embed_fn=embed, llm=llm, targets=targets, cfg=CFG,
                                  sleep_fn=lambda s: None)
    assert (embed.calls, llm.calls) == calls                     # nothing recomputed
    assert again["proposals"] == first["proposals"]


def test_without_ai_proposals_stay_untyped(ws, targets):
    result = propose_for_workspace(ws, embed_fn=Embed(), use_llm=False, targets=targets,
                                   cfg=CFG)
    by = _by_name(result)
    assert by["PersonalData"]["status"] == STATUS_PROPOSED
    assert by["PersonalData"]["relation"] is None
    assert result["summary"]["proposed_untyped"] == result["summary"]["proposed"]
    assert "IA: no usada" in render_console(result)


def test_ai_none_drops_the_proposal(ws, targets):
    result = propose_for_workspace(ws, embed_fn=Embed(), llm=FakeLLM(relation="none"),
                                   targets=targets, cfg=CFG, sleep_fn=lambda s: None)
    by = _by_name(result)
    assert by["PersonalData"]["status"] == STATUS_AI_NONE
    assert result["summary"]["proposed"] == 0


def test_quota_exhausted_is_reported_and_proposals_stay_untyped(ws, targets):
    llm = FakeLLM(error="429 RESOURCE_EXHAUSTED: quota exceeded")
    result = propose_for_workspace(ws, embed_fn=Embed(), llm=llm, targets=targets, cfg=CFG,
                                   sleep_fn=lambda s: None)
    ai = result["summary"]["ai"]
    assert ai["available"] and ai["quota_exhausted"]
    assert _by_name(result)["PersonalData"]["relation"] is None
    assert "cuota agotada" in render_console(result)


def test_missing_api_key_is_reported(ws, targets, monkeypatch):
    def no_key(*a, **k):
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr(prop, "justify_candidates", no_key)
    result = propose_for_workspace(ws, embed_fn=Embed(), targets=targets, cfg=CFG)
    assert result["summary"]["ai"] == {"available": False,
                                       "error": "Falta la variable 'LLM_API_KEY' en .env"}


def test_needs_the_inheritance_first(tmp_path, targets):
    w = open_ontology("x.ttl", TTL.encode("utf-8"),
                      cfg={"workspaces": {"dir": str(tmp_path / "ws")}})
    with pytest.raises(ValueError, match="Primero hay que heredar"):
        propose_for_workspace(w, embed_fn=Embed(), targets=targets, cfg=CFG)


def test_cli_without_ai(tmp_path, monkeypatch, capsys, targets):
    data, log = _decide_all(tmp_path)
    cand = tmp_path / "base.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    base = load_base(cand, log)
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()                                    # no real model in the tests
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.config.load_config",
                        lambda: {"workspaces": {"dir": str(tmp_path / "ws")}, **CFG})
    monkeypatch.setattr(inh, "load_base", lambda **kw: base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)
    path = tmp_path / "gdpr.ttl"
    path.write_text(TTL, encoding="utf-8")
    assert prop._run([str(path), "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Propuestas automáticas (S6-T06)" in out
    assert "PersonalData -> dpv:PersonalData  sin tipo (rdfs:seeAlso)" in out
    assert prop._run([str(tmp_path / "no.ttl")]) == 1