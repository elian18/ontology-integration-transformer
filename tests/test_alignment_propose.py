"""S7-T06: the decider and its fixed rule for the concepts of an upload that inherited nothing
(and the Sprint 6 proposer, kept for the S6-T09 check).

Fast tests: a fake embedder, a fake LLM that answers like the decider's (``DeciderLLM``, which
also answers the labels prompt of S7-T08, so the whole web flow runs with it), a small
DPV and a small validated base written to temporary files, and a temporary ChromaDB where the
knowledge collections are built from them."""
from __future__ import annotations

import json
import re
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from src.ai.rag import vector_store
from src.alignment import decider as dc
from src.alignment import inherit as inh
from src.alignment import propose as prop
from src.alignment.decider import CACHE_FILE
from src.alignment.export import CANDIDATES_JSON
from src.alignment.inherit import inherit_for_workspace, load_base
from src.alignment.dpv_targets import build_dpv_targets
from src.core.dpv_proximity import _humanize
from src.alignment.propose import (ACCEPTED, BY_RULE, BY_RULE_WITHOUT_AI, NO_MATCH, NO_PROPOSAL,
                                   RESULT_FILE, STATUS_AI_NONE, STATUS_BELOW, STATUS_PROPOSED,
                                   SUGGESTED, UNDECIDED, decide, is_current, mark_for_ai,
                                   propose_for_workspace, render_console)
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
TTL_PROPERTY = TTL + "ex:hasRecipient a owl:ObjectProperty .\n"
CFG = {"auto": {"threshold": 0.6, "use_llm": True, "llm_max_calls": 5},
       "alignment": {"top_k": 3, "weight_lexical": 0.4, "weight_semantic": 0.6,
                     "justify_batch_size": 15, "justify_pause_seconds": 0},
       "decider": {"candidates": 3, "precedents": 2, "batch_size": 10, "pause_seconds": 0},
       "labels": {"pause_seconds": 0},
       "validation": {"engine": "owlrl"},           # fast: Jena has its own tests
       "audit": {"pause_seconds": 0}}


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
            elif "recipient" in low:
                v[2] = 1.0
            else:
                v[3 + sum(map(ord, low)) % 13] = 1.0
            out.append(v)
        return out


class FakeLLM:
    """Sprint 6 proposer (justify prompt): answers every candidate with one relation."""

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


ANSWERS = {"PersonalData": ("PersonalData", "exactMatch", "alta"),
           "DataController": ("DataController", "relatedMatch", "alta"),
           "hasRecipient": ("hasRecipient", "exactMatch", "alta")}
LABELS_ES = {"Consent": "Consentimiento", "PersonalData": "Dato personal",
             "DataController": "Responsable del tratamiento", "Pizza": "Pizza",
             "hasRecipient": "Tiene destinatario"}


class DeciderLLM:
    """Answers like the decider's LLM: the term of ``answers`` if it is a candidate, otherwise
    'ninguno' with high confidence. To the labels prompt (S7-T08) it answers the Spanish label
    of ``LABELS_ES`` ("Etiqueta de <name>" for the others) and the name split into words; to
    the reviewer (S8-T09), "coherente"."""

    def __init__(self, answers=None, error=None, model="modelo-de-prueba"):
        self.answers = ANSWERS if answers is None else answers
        self.error, self.model, self.calls, self.contexts = error, model, 0, []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        if self.error:
            raise Exception(self.error)
        out = []
        if '"labels"' in prompt:
            for block in context.split("=== ")[1:]:
                name = re.search(r"^Concepto: (\S+) \(", block, re.M).group(1)
                out.append({"concept": block.split(" ===", 1)[0], "en": _humanize(name),
                            "es": LABELS_ES.get(name, f"Etiqueta de {name}")})
            return json.dumps({"labels": out}, ensure_ascii=False)
        if '"reviews"' in prompt:
            for block in context.split("=== ")[1:]:
                name = re.search(r"^Concepto: (\S+) \(", block, re.M).group(1)
                out.append({"concept": block.split(" ===", 1)[0], "verdict": "coherente",
                            "relation": "exactMatch",
                            "explanation": f"Revisión de prueba para {name}."})
            return json.dumps({"reviews": out}, ensure_ascii=False)
        for block in context.split("=== ")[1:]:
            cid = block.split(" ===", 1)[0]
            name = re.search(r"^Concepto: (\S+) \(", block, re.M).group(1)
            cands = re.findall(r"^\d+\. (\S+) \|", block, flags=re.M)
            dpv, rel, conf = self.answers.get(name, ("ninguno", "", "alta"))
            if dpv != "ninguno" and dpv not in cands:
                dpv, rel = "ninguno", ""
            out.append({"concept": cid, "dpv": dpv, "relation": rel, "confidence": conf,
                        "justification": f"Decisión de prueba para {name}."})
        return json.dumps({"decisions": out}, ensure_ascii=False)


@pytest.fixture
def targets(tmp_path):
    path = tmp_path / "dpv.ttl"
    path.write_text(_DPV, encoding="utf-8")
    return build_dpv_targets(load_dpv(path))


def make_env(tmp_path) -> SimpleNamespace:
    """The small validated base on disk + a config that points to it (and to the small DPV),
    so the knowledge collections are built from real files, in a temporary ChromaDB."""
    data, log = _decide_all(tmp_path)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    cand = out / CANDIDATES_JSON
    cand.write_text(json.dumps(data), encoding="utf-8")
    dpv = tmp_path / "dpv.ttl"
    if not dpv.exists():
        dpv.write_text(_DPV, encoding="utf-8")
    cfg = {**CFG, "outputs": {"dir": str(out)}, "review": {"decisions_file": str(log)},
           "inputs": {"dpv": str(dpv)}, "workspaces": {"dir": str(tmp_path / "ws")}}
    return SimpleNamespace(base=load_base(cand, log), cfg=cfg, chroma=tmp_path / "chroma")


@pytest.fixture
def env(tmp_path, targets):
    return make_env(tmp_path)


def _workspace(env, ttl=TTL, name="gdpr.ttl"):
    w = open_ontology(name, ttl.encode("utf-8"), cfg=env.cfg)
    inherit_for_workspace(w, base=env.base, use_embeddings=False)
    return w


@pytest.fixture
def ws(env):
    """Workspace whose inheritance is done: Consent inherits (same name), the rest does not."""
    return _workspace(env)


def run(ws, env, targets, llm=None, **kwargs):
    kwargs.setdefault("embed_fn", Embed())
    return propose_for_workspace(ws, llm=llm, targets=targets, cfg=env.cfg, base=env.base,
                                 chroma_path=env.chroma, sleep_fn=lambda s: None, **kwargs)


def _by_name(result):
    return {e["name"]: e for e in result["entries"]}


# ------------------------------------------------------------------ Sprint 6 proposer (S6-T09)

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


# ------------------------------------------------------------------ the rule on decisions

D = "https://w3id.org/dpv#"


def _d(key, status=dc.STATUS_DECIDED, term="X", relation="skos:exactMatch", confidence="alta",
       score=0.8, best=0.8, kinds=("class",), none=False, candidates=("X",)):
    return dc.Decision(key=key, name=key, status=status, none=none,
                       dpv_iri=None if none or term is None else D + term, dpv_name=term,
                       relation=relation, confidence=confidence, score=score, best_score=best,
                       kinds=kinds, candidates=[D + c for c in candidates])


def test_entries_follow_the_rule_and_the_rule_without_ai():
    pending = dict(status=dc.STATUS_PENDING, term=None, relation=None, confidence=None,
                   score=None)
    decisions = [_d("accepted"), _d("weak", relation="skos:relatedMatch"),
                 _d("none", none=True), _d("medium", confidence="media"),
                 _d("invalid", status=dc.STATUS_INVALID, term=None),
                 _d("at_threshold", best=0.6, **pending), _d("below", best=0.59, **pending),
                 _d("property", kinds=("property",), **pending),
                 _d("no_candidates", candidates=(), best=None, **pending)]
    entries = {e["key"]: e for e in prop.entries_from(decisions, dc.RuleSettings(), 0.6)}
    status = {k: (e["status"], e["decided_by"]) for k, e in entries.items()}
    assert status == {"accepted": (ACCEPTED, BY_RULE), "weak": (SUGGESTED, BY_RULE),
                      "none": (NO_MATCH, BY_RULE), "medium": (NO_PROPOSAL, BY_RULE),
                      "invalid": (NO_PROPOSAL, BY_RULE),
                      "at_threshold": (ACCEPTED, BY_RULE_WITHOUT_AI),
                      "below": (UNDECIDED, BY_RULE_WITHOUT_AI),
                      "property": (UNDECIDED, BY_RULE_WITHOUT_AI),
                      "no_candidates": (UNDECIDED, BY_RULE_WITHOUT_AI)}
    fallback = entries["at_threshold"]
    assert (fallback["dpv_iri"], fallback["dpv_name"], fallback["relation"], fallback["score"],
            fallback["rank"]) == (D + "X", "X", None, 0.6, 1)
    assert entries["weak"]["relation"] == "skos:relatedMatch"
    assert entries["medium"]["dpv_iri"] is None and entries["none"]["dpv_iri"] is None
    assert [e["key"] for e in prop.entries_from(decisions, dc.RuleSettings(), 0.6)][:2] == [
        "accepted", "at_threshold"]                              # best score first
    summary = prop.decision_summary(list(entries.values()), {"available": True},
                                    dc.RuleSettings(), 0.6)
    assert (summary["accepted"], summary["accepted_by_rule"], summary["accepted_without_ai"],
            summary["suggested"], summary["pending_ai"]) == (2, 1, 1, 1, 4)
    assert summary["by_relation"] == {"skos:exactMatch": 1, "rdfs:seeAlso": 1}


# ------------------------------------------------------------------ the decider in the flow

def test_full_run_with_the_decider(ws, env, targets):
    llm = DeciderLLM()
    result = run(ws, env, targets, llm)
    by = _by_name(result)
    assert "Consent" not in by                                   # inherited in S6-T05
    assert set(by) == {"PersonalData", "DataController", "Pizza"}
    pd = by["PersonalData"]
    assert (pd["status"], pd["decided_by"], pd["dpv_name"], pd["relation"]) == (
        ACCEPTED, BY_RULE, "PersonalData", "skos:exactMatch")
    assert pd["score"] >= 0.5 and pd["confidence"] == "alta" and pd["version"] == 1
    assert pd["justification"] == "Decisión de prueba para PersonalData."
    dc = by["DataController"]                                    # relatedMatch: left out
    assert (dc["status"], dc["dpv_name"], dc["relation"]) == (SUGGESTED, "DataController",
                                                               "skos:relatedMatch")
    assert by["Pizza"]["status"] == NO_MATCH and by["Pizza"]["dpv_iri"] is None
    s = result["summary"]
    assert (s["accepted"], s["suggested"], s["no_match"], s["pending_ai"]) == (1, 1, 1, 0)
    assert s["ai"]["available"] and s["ai"]["calls"] == llm.calls == 1
    assert s["rule"].startswith("confianza alta de la IA y puntaje >= 0.5")
    assert result["rule"]["property_relations"] == [] and result["prompt"]["name"] == "decide"
    assert ws.step("candidates")["status"] == DONE and ws.step("propose")["status"] == DONE
    assert prop.load_result(ws)["summary"] == s
    assert (ws.work_dir / CACHE_FILE).exists()                   # cache of the workspace
    assert vector_store.count(name="dpv", path=env.chroma) == len(targets.targets)
    assert vector_store.count(name="decisions", path=env.chroma) == len(env.base.concepts)


def test_the_llm_sees_precedents_but_no_law_article(ws, env, targets):
    llm = DeciderLLM()
    run(ws, env, targets, llm)
    context = llm.contexts[0]
    assert "Precedentes (decisiones de una persona en conceptos parecidos):" in context
    assert "Articulo" not in context


def test_concepts_that_are_only_properties_are_never_automatic(env, targets):
    w = _workspace(env, TTL_PROPERTY, "props.ttl")
    by = _by_name(run(w, env, targets, DeciderLLM()))
    assert by["hasRecipient"]["status"] == SUGGESTED              # exactMatch, alta, but a property
    assert by["hasRecipient"]["relation"] == "skos:exactMatch"


def test_medium_confidence_gives_no_proposal(ws, env, targets):
    llm = DeciderLLM(answers={"PersonalData": ("PersonalData", "exactMatch", "media")})
    pd = _by_name(run(ws, env, targets, llm))["PersonalData"]
    assert pd["status"] == NO_PROPOSAL and pd["dpv_iri"] is None and pd["confidence"] == "media"


def test_rerun_uses_the_cache_of_the_workspace(ws, env, targets):
    first = run(ws, env, targets, DeciderLLM())
    llm = DeciderLLM()
    again = run(ws, env, targets, llm)
    assert llm.calls == 0 and again["summary"]["ai"]["cached"] == 3
    status = (lambda r: {e["name"]: (e["status"], e["dpv_iri"]) for e in r["entries"]})  # noqa
    assert status(again) == status(first)


def test_without_ai_the_rule_of_sprint_6_decides(ws, env, targets):
    result = run(ws, env, targets, use_llm=False)
    by = _by_name(result)
    for name in ("PersonalData", "DataController"):              # first candidate >= 0.6
        e = by[name]
        assert (e["status"], e["decided_by"], e["relation"]) == (ACCEPTED, BY_RULE_WITHOUT_AI,
                                                                 None)
        assert e["dpv_name"] == name and e["score"] >= 0.6 and e["confidence"] is None
    assert by["Pizza"]["status"] == UNDECIDED
    s = result["summary"]
    assert s["accepted_without_ai"] == 2 and s["pending_ai"] == 3 and not s["ai"]["available"]
    text = render_console(result)
    assert "IA: no usada (IA desactivada (--sin-ia))" in text and "regla sin IA" in text


def test_without_ai_properties_still_are_never_automatic(env, targets):
    w = _workspace(env, TTL_PROPERTY, "props.ttl")
    assert _by_name(run(w, env, targets, use_llm=False))["hasRecipient"]["status"] == UNDECIDED


def test_without_ai_the_cached_answers_still_count(ws, env, targets, monkeypatch):
    run(ws, env, targets, DeciderLLM())
    monkeypatch.setenv("LLM_MODEL", "modelo-de-prueba")         # the model of .env names the cache
    by = _by_name(run(ws, env, targets, use_llm=False))
    assert by["PersonalData"]["decided_by"] == BY_RULE                 # from the cache
    assert by["DataController"]["status"] == SUGGESTED


def test_quota_exhausted_falls_back_and_is_reported(ws, env, targets):
    llm = DeciderLLM(error="429 RESOURCE_EXHAUSTED: quota exceeded")
    result = run(ws, env, targets, llm)
    ai = result["summary"]["ai"]
    assert ai["available"] and ai["quota_exhausted"]
    assert _by_name(result)["PersonalData"]["decided_by"] == BY_RULE_WITHOUT_AI
    assert "cuota agotada" in render_console(result)


def test_missing_api_key_is_reported(ws, env, targets, monkeypatch):
    def no_key(*a, **k):
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr("src.ai.llm_client.LLMClient", no_key)
    result = run(ws, env, targets)
    ai = result["summary"]["ai"]
    assert ai["available"] is False and ai["error"] == "Falta la variable 'LLM_API_KEY' en .env"
    assert _by_name(result)["PersonalData"]["decided_by"] == BY_RULE_WITHOUT_AI


def test_nothing_to_decide_builds_nothing_and_asks_nothing(env, targets):
    w = _workspace(env, TTL.split("ex:PersonalData")[0], "solo-consent.ttl")
    llm = DeciderLLM()
    result = run(w, env, targets, llm)
    assert result["entries"] == [] and llm.calls == 0
    assert not vector_store.exists("dpv", env.chroma)


def test_when_a_saved_result_can_be_reused(ws, env, targets):
    assert not is_current(None, env.cfg)
    assert not is_current({"method": "otro"}, env.cfg)
    done = run(ws, env, targets, DeciderLLM())
    assert is_current(done, env.cfg)
    assert not is_current({**done, "method_version": 0}, env.cfg)
    other = {**env.cfg, "decider": {**env.cfg["decider"], "rule": {"min_score": 0.7}}}
    assert not is_current(done, other)                           # another rule: decide again
    partial = run(ws, env, targets, use_llm=False)
    assert not is_current(partial, env.cfg)                      # AI available: complete it
    assert is_current(partial, env.cfg, use_llm=False)
    threshold = {**env.cfg, "auto": {**env.cfg["auto"], "threshold": 0.9}}
    assert not is_current(done, threshold)


def test_needs_the_inheritance_first(env, targets):
    w = open_ontology("x.ttl", TTL.encode("utf-8"), cfg=env.cfg)
    with pytest.raises(ValueError, match="Primero hay que heredar"):
        run(w, env, targets)


def test_cli_without_ai(tmp_path, monkeypatch, capsys, env, targets):
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()                                    # no real model in the tests
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma-cli"))
    monkeypatch.setattr("src.config.load_config", lambda: env.cfg)
    monkeypatch.setattr(inh, "load_base", lambda **kw: env.base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)
    path = tmp_path / "gdpr.ttl"
    path.write_text(TTL, encoding="utf-8")
    assert prop._run([str(path), "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Decisor en tu ontología (S7-T06)" in out
    assert "PersonalData -> dpv:PersonalData  sin tipo (rdfs:seeAlso)" in out
    assert RESULT_FILE in out
    assert prop._run([str(tmp_path / "no.ttl")]) == 1