"""S7-T04: the alignment decider (retrieve -> decide with precedents -> fixed rule).

Fast tests: a fake embedder, a fake LLM that reads the prompt like Gemini would, and a
temporary ChromaDB with the two collections of S7-T01 built from small hand-made data. One slow
test asks the real Gemini to decide a small batch.
"""
import hashlib
import json
import re

import numpy as np
import pytest

from src.alignment import decider as dc
from src.alignment import knowledge as kn
from src.alignment.dpv_targets import DpvTarget, DpvTargets
from src.alignment.inherit import BaseConcept, BaseIndex
from src.alignment.justify import QUOTA_WAIT_SECONDS, TRANSIENT_WAIT_SECONDS
from src.alignment.sources import SourceConcept

D = "https://w3id.org/dpv#"
B = "http://base.org/onto#"
FAST = dc.DeciderSettings(candidates=3, precedents=2, batch_size=2, max_calls=20,
                          pause_seconds=0)


class FakeEmbed:
    def __call__(self, texts):
        out = []
        for t in texts:
            v = np.zeros(32)
            for w in re.findall(r"[a-z]+", str(t).lower()):
                v[int(hashlib.md5(w[:5].encode()).hexdigest(), 16) % 32] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


class FakeLLM:
    """Answers like Gemini would: follows the CLOSEST precedent (its term if it is a candidate,
    'ninguno' if it had no correspondence); without precedents it says 'ninguno'."""

    def __init__(self, fail=None, answer=None, only=None, model="modelo-de-prueba"):
        self.model = model
        self.calls = []
        self.fail = list(fail or [])
        self.answer = answer                  # fn(cid, candidates, precedents) -> dict
        self.only = only                      # answer only these ids

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls.append({"prompt": prompt, "context": context, "json_mode": json_mode})
        if self.fail:
            exc = self.fail.pop(0)
            if exc is not None:
                raise exc
        out = []
        for block in context.split("=== ")[1:]:
            cid = block.split(" ===", 1)[0]
            if self.only and cid not in self.only:
                continue
            cands = re.findall(r"^\d+\. (\S+) \|", block, flags=re.M)
            precs = [None if d.strip() == "sin correspondencia" else
                     re.match(r"(\w+) \((\w+)\)", d.strip()).groups()
                     for d in re.findall(r"^- .*?: (.+)$", block, flags=re.M)]
            if self.answer:
                out.append({"concept": cid, **self.answer(cid, cands, precs)})
                continue
            first = precs[0] if precs else None
            if first and first[0] in cands:
                out.append({"concept": cid, "dpv": first[0], "relation": first[1],
                            "confidence": "alta", "justification": f"Sigue el precedente ({cid})."})
            else:
                out.append({"concept": cid, "dpv": "ninguno", "relation": "",
                            "confidence": "media", "justification": "Nada corresponde."})
        return json.dumps({"decisions": out}, ensure_ascii=False)


@pytest.fixture
def targets():
    return DpvTargets(dpv_path="t", targets=[
        DpvTarget(D + "Consent", "Consent", "class", "Consent", "Consent of the data subject", []),
        DpvTarget(D + "DataSubjectRight", "DataSubjectRight", "class", "Data Subject Right",
                  "The rights of a data subject such as access or rectification", ["Right"]),
        DpvTarget(D + "Right", "Right", "class", "Right", "A right", []),
        DpvTarget(D + "hasPurpose", "hasPurpose", "property", "has purpose",
                  "Indicates association with purpose", [])])


def _bc(name, status, mappings=(), kinds=("class",), label=None):
    return BaseConcept(B + name, B + name, "ontology", name, label, None, tuple(kinds), status,
                       [{"dpv_iri": D + t, "dpv_name": t, "relation": "skos:" + r}
                        for t, r in mappings])


@pytest.fixture
def base():
    return BaseIndex(concepts=[
        _bc("Right_of_access", "aligned", [("DataSubjectRight", "broadMatch")],
            label="derecho de acceso"),
        _bc("Right_to_portability", "aligned", [("DataSubjectRight", "broadMatch")],
            label="derecho a la portabilidad"),
        _bc("Sanction", "no_match", label="sancion"),
        _bc("has_goal", "aligned", [("hasPurpose", "relatedMatch")], kinds=("property",))],
        source_id="ontopriv+lopdp")


@pytest.fixture
def chroma(tmp_path, targets, base):
    path = tmp_path / "chroma"
    embed = FakeEmbed()
    kn.ensure_collection("dpv", "fp", lambda: kn.dpv_records(targets), embed_fn=embed,
                         model_name="fake-model", path=path)
    kn.ensure_collection("decisions", "fp", lambda: kn.decision_records(base), embed_fn=embed,
                         model_name="fake-model", path=path)
    return path


def _concepts():
    return [SourceConcept(key="u#Right_to_rectification", origin="ontology",
                          name="Right_to_rectification", kinds=("class",),
                          label="derecho de rectificacion"),
            SourceConcept(key="u#Penalty", origin="ontology", name="Penalty", kinds=("class",),
                          label="sancion por infraccion"),
            SourceConcept(key="u#has_aim", origin="ontology", name="has_aim",
                          kinds=("property",), label="tiene como finalidad")]


def _run(targets, chroma, llm=None, *, concepts=None, settings=FAST, cache=None, sleep=None,
         **kwargs):
    llm = llm or FakeLLM()
    sleeps = sleep if sleep is not None else []
    report = dc.decide(concepts if concepts is not None else _concepts(), targets, llm=llm,
                       embed_fn=FakeEmbed(), settings=settings, chroma_path=chroma,
                       embedding_model="fake-model", cache_path=cache,
                       sleep_fn=sleeps.append, **kwargs)
    return report, llm


# --------------------------------------------------------------------------- settings / prompt

def test_settings_come_from_the_decider_block():
    s = dc.settings_from_config({"decider": {"precedents": 3, "batch_size": 8}})
    assert (s.candidate_source, s.candidates, s.precedents, s.batch_size) == ("hybrid", 10, 3, 8)
    assert dc.settings_from_config() == dc.DeciderSettings()     # config.yaml = the defaults
    with pytest.raises(ValueError, match="bloque decider"):
        dc.settings_from_config({"decider": {"precedents": 11}})


def test_prompt_v1_is_loaded_and_has_no_fixed_examples_from_the_base():
    assert dc.PROMPT.name == "decide" and dc.PROMPT.version == 1
    text = dc.PROMPT.text
    assert '"ninguno"' in text and '"decisions"' in text and "precedentes" in text
    for base_name in ("Data_Controller", "Right_to_portability", "Sensitive_data",
                      "for_serious_infractions"):
        assert base_name not in text


# --------------------------------------------------------------------------- retrieve

def test_precedents_respect_kind_self_and_leak_control(chroma, base):
    embed = FakeEmbed()
    rect = _concepts()[0]
    vec = embed([rect.semantic_text()])[0]
    found = dc.find_precedents(rect, vec, 4, path=chroma)
    assert {p["key"] for p in found} == {B + "Right_of_access", B + "Right_to_portability",
                                         B + "Sanction"}               # never the property
    hidden = dc.find_precedents(rect, vec, 4, forbidden={B + "Right_of_access"}, path=chroma)
    assert B + "Right_of_access" not in {p["key"] for p in hidden}
    twin = SourceConcept(key=B + "Sanction", origin="ontology", name="Sanction",
                         kinds=("class",), label="sancion")
    own = dc.find_precedents(twin, embed([twin.semantic_text()])[0], 4, path=chroma)
    assert B + "Sanction" not in {p["key"] for p in own}              # never itself
    prop = _concepts()[2]
    only_props = dc.find_precedents(prop, embed([prop.semantic_text()])[0], 4, path=chroma)
    assert [p["key"] for p in only_props] == [B + "has_goal"]


def test_ties_between_identical_precedents_are_broken_by_key(tmp_path):
    names = ("Zeta_measures", "Yota_measures", "Xi_measures", "Alpha_measures", "Beta_measures")
    twins = BaseIndex(concepts=[_bc(n, "no_match") for n in names], source_id="s")
    path = tmp_path / "ties"
    kn.ensure_collection("decisions", "fp", lambda: kn.decision_records(twins),
                         embed_fn=lambda docs: [[1.0, 0.0] for _ in docs],
                         model_name="fake-model", path=path)
    probe = SourceConcept(key="u#X", origin="ontology", name="X", kinds=("class",))
    found = dc.find_precedents(probe, [1.0, 0.0], 2, path=path)
    assert [p["key"] for p in found] == [B + "Alpha_measures", B + "Beta_measures"]


def test_decisions_collection_must_be_built_with_the_same_model(chroma):
    dc.check_decisions_collection("fake-model", chroma)
    with pytest.raises(ValueError, match="knowledge"):
        dc.check_decisions_collection("otro-modelo", chroma)


# --------------------------------------------------------------------------- the prompt body

def test_body_shows_candidates_precedents_and_evidence_but_no_scores(targets, chroma):
    article = {"number": 16, "title": "Derecho de rectificacion", "text": "El titular puede..."}
    three = dc.DeciderSettings(candidates=3, precedents=3, batch_size=2, pause_seconds=0)
    _, llm = _run(targets, chroma, concepts=_concepts()[:1], settings=three,
                  evidence_fn=lambda c: article)
    context = llm.calls[0]["context"]
    assert "=== C1 ===" in context and "Concepto: Right_to_rectification (tipo: class)" in context
    assert "Etiqueta: derecho de rectificacion" in context
    assert "Articulo 16 de la ley (Derecho de rectificacion): El titular puede..." in context
    assert re.search(r"^1\. \S+ \| .+ \| definicion: .+ \| padres en el DPV: ", context, re.M)
    assert "Precedentes (decisiones de una persona en conceptos parecidos):" in context
    assert "DataSubjectRight (broadMatch)" in context and "sin correspondencia" in context
    assert not re.search(r"\b0\.\d", context)                          # no scores shown


def test_without_precedents_the_section_is_absent(targets, chroma):
    _, llm = _run(targets, chroma, concepts=_concepts()[:1], use_precedents=False)
    assert "Precedentes" not in llm.calls[0]["context"]


# --------------------------------------------------------------------------- calling the LLM

def test_llm_gets_the_decide_prompt_in_json_mode_in_batches(targets, chroma):
    sleeps = []
    report, llm = _run(targets, chroma, settings=dc.DeciderSettings(
        candidates=3, precedents=2, batch_size=2, pause_seconds=5), sleep=sleeps)
    assert report.calls == 2 and len(llm.calls) == 2                   # 3 concepts, 2 per call
    assert all(c["json_mode"] and c["prompt"] == dc.PROMPT.text for c in llm.calls)
    assert sleeps == [5]                                               # between calls only


def test_answers_become_decisions_with_provenance(targets, chroma):
    report, _ = _run(targets, chroma, evidence_fn=lambda c: {"number": 7, "title": "", "text": "t"})
    d = {x.key: x for x in report.decisions}
    rect = d["u#Right_to_rectification"]
    assert rect.status == dc.STATUS_DECIDED and rect.dpv_iri == D + "DataSubjectRight"
    assert rect.relation == "skos:broadMatch" and rect.confidence == "alta"
    assert rect.score is not None and rect.rank in (1, 2, 3)
    assert rect.justification == "Sigue el precedente."                 # batch id cleaned
    assert rect.candidates and set(rect.precedents) <= {B + "Right_of_access",
                                                        B + "Right_to_portability",
                                                        B + "Sanction"}
    assert rect.evidence_article == 7 and rect.model == "modelo-de-prueba"
    assert rect.prompt == {"name": "decide", "version": 1, "sha256": dc.PROMPT.sha256}
    assert d["u#Penalty"].none is True and d["u#Penalty"].dpv_iri is None
    assert d["u#has_aim"].dpv_iri == D + "hasPurpose"
    assert rect.kinds == ("class",) and not rect.is_property and d["u#has_aim"].is_property


def test_leak_control_hides_forbidden_precedents(targets, chroma):
    banned = {"u#Right_to_rectification": {B + "Right_of_access", B + "Right_to_portability"}}
    report, llm = _run(targets, chroma, concepts=_concepts()[:1], forbidden=banned)
    assert "Right_of_access" not in llm.calls[0]["context"]
    assert report.decisions[0].none is True            # without its precedents it says ninguno


def test_invented_term_is_rejected(targets, chroma):
    llm = FakeLLM(answer=lambda cid, c, p: {"dpv": "InventedTerm", "relation": "exactMatch",
                                           "confidence": "alta"})
    report, _ = _run(targets, chroma, llm, concepts=_concepts()[:1])
    d = report.decisions[0]
    assert d.status == dc.STATUS_INVALID and d.dpv_iri is None
    assert "no está entre los candidatos" in d.issues[0]
    assert dc.apply_rule(report.decisions) == {}


def test_number_aliases_and_bad_relation_are_handled(targets, chroma):
    llm = FakeLLM(answer=lambda cid, c, p: (
        {"dpv": "2", "relation": "BroadMatch", "confidence": "High"} if cid == "C1" else
        {"dpv": c[0], "relation": "sameAs", "confidence": "quizas"}))
    report, _ = _run(targets, chroma, llm, concepts=_concepts()[:2])
    first, second = report.decisions
    assert first.rank == 2 and first.relation == "skos:broadMatch" and first.confidence == "alta"
    assert second.dpv_iri and second.relation is None and second.confidence == "baja"
    assert "relación SKOS no válida" in second.issues
    assert any("confianza no válida" in i for i in second.issues)


def test_broken_json_leaves_the_batch_pending_and_next_run_asks_again(targets, chroma, tmp_path):
    cache = tmp_path / "cache.jsonl"

    class Broken(FakeLLM):
        def ask(self, *a, **k):
            super().ask(*a, **k)
            return "no es json"

    first, _ = _run(targets, chroma, Broken(), cache=cache)
    assert all(d.status == dc.STATUS_PENDING for d in first.decisions)
    assert first.errors and "JSON" in first.errors[0]["error"]
    second, llm = _run(targets, chroma, cache=cache)
    assert len(llm.calls) == 2 and all(d.status == dc.STATUS_DECIDED for d in second.decisions)


def test_cache_avoids_new_calls_until_something_changes(targets, chroma, tmp_path):
    cache = tmp_path / "cache.jsonl"
    _run(targets, chroma, cache=cache)
    again, llm = _run(targets, chroma, cache=cache)
    assert llm.calls == [] and again.cached == 3
    assert all(d.from_cache for d in again.decisions)
    other_model, llm = _run(targets, chroma, FakeLLM(model="otro"), cache=cache)
    assert len(llm.calls) == 2                                          # new model: new calls
    no_prec, llm = _run(targets, chroma, cache=cache, use_precedents=False)
    assert len(llm.calls) == 2                                          # what it sees changed


def test_unanswered_concept_stays_pending_and_is_not_cached(targets, chroma, tmp_path):
    cache = tmp_path / "cache.jsonl"
    report, _ = _run(targets, chroma, FakeLLM(only={"C1"}), concepts=_concepts()[:2], cache=cache)
    assert [d.status for d in report.decisions] == [dc.STATUS_DECIDED, dc.STATUS_PENDING]
    assert len(dc.load_cache(cache)) == 1


def test_a_repeated_answer_for_the_same_concept_keeps_the_first(targets, chroma, tmp_path):
    cache = tmp_path / "cache.jsonl"

    class Twice(FakeLLM):
        def ask(self, *a, **k):
            super().ask(*a, **k)
            return json.dumps({"decisions": [
                {"concept": "C1", "dpv": "ninguno", "confidence": "alta"},
                {"concept": "C1", "dpv": "Consent", "relation": "exactMatch",
                 "confidence": "alta"}]})

    report, _ = _run(targets, chroma, Twice(), concepts=_concepts()[:1], cache=cache)
    assert report.decisions[0].none is True and report.decisions[0].dpv_iri is None
    assert len(dc.load_cache(cache)) == 1
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 1


def test_quota_stops_cleanly_and_the_next_run_resumes(targets, chroma, tmp_path):
    cache = tmp_path / "cache.jsonl"
    quota = Exception("Error code: 429 RESOURCE_EXHAUSTED quota")
    sleeps = []
    first, _ = _run(targets, chroma, FakeLLM(fail=[None, quota, quota]), cache=cache,
                    sleep=sleeps)
    assert first.quota_exhausted and QUOTA_WAIT_SECONDS in sleeps
    assert [d.status for d in first.decisions].count(dc.STATUS_DECIDED) == 2
    second, llm = _run(targets, chroma, cache=cache)
    assert len(llm.calls) == 1 and second.cached == 2
    assert all(d.status == dc.STATUS_DECIDED for d in second.decisions)


def test_server_overload_waits_once_then_continues(targets, chroma):
    sleeps = []
    report, _ = _run(targets, chroma, FakeLLM(fail=[Exception("503 UNAVAILABLE overloaded")]),
                     sleep=sleeps)
    assert TRANSIENT_WAIT_SECONDS in sleeps and not report.errors
    assert all(d.status == dc.STATUS_DECIDED for d in report.decisions)


def test_call_limit_leaves_the_rest_pending(targets, chroma):
    settings = dc.DeciderSettings(candidates=3, precedents=2, batch_size=1, max_calls=1,
                                  pause_seconds=0)
    report, _ = _run(targets, chroma, settings=settings)
    assert report.call_limit_reached and report.calls == 1
    assert [d.status for d in report.decisions].count(dc.STATUS_PENDING) == 2


def test_concept_without_candidates_is_not_sent(targets, chroma):
    odd = SourceConcept(key="u#X", origin="ontology", name="X", kinds=())
    report, llm = _run(targets, chroma, concepts=[odd])
    assert report.decisions[0].status == dc.STATUS_NO_CANDIDATES and llm.calls == []


# --------------------------------------------------------------------------- parsing / rule

@pytest.mark.parametrize("raw", [
    '{"decisions": [{"concept": "C1", "dpv": "ninguno"}]}',
    '```json\n{"decisions": [{"concept": "C1", "dpv": "ninguno"}]}\n```',
    '[{"concept": "C1", "dpv": "ninguno"}]',
    'Aqui va: {"decisiones": [{"concept": "C1", "dpv": "ninguno"}]} gracias',
])
def test_parse_answer_variants(raw):
    assert dc.parse_answer(raw) == [{"concept": "C1", "dpv": "ninguno"}]


def test_parse_answer_rejects_text_without_decisions():
    with pytest.raises(ValueError, match="lista de decisiones"):
        dc.parse_answer('{"otra": 1}')


def _dec(key, status=dc.STATUS_DECIDED, term=True, confidence="alta", score=0.8, none=False,
         relation="skos:closeMatch", kinds=("class",)):
    return dc.Decision(key=key, name=key, status=status,
                       dpv_iri=(D + "X") if term and not none else None,
                       relation=relation if term and not none else None,
                       none=none, confidence=confidence, score=score, kinds=kinds)


def test_fixed_rule_variants():
    decisions = [_dec("a"), _dec("b", confidence="media"), _dec("c", none=True),
                 _dec("d", none=True, confidence="baja"), _dec("e", score=0.5),
                 _dec("p", status=dc.STATUS_PENDING), _dec("i", status=dc.STATUS_INVALID)]
    only_high = dc.apply_rule(decisions)
    assert set(only_high) == {"a", "c", "e"}
    assert only_high["a"] == {"term": D + "X", "relation": "skos:closeMatch", "none": False}
    assert only_high["c"] == {"term": None, "relation": None, "none": True}
    assert set(dc.apply_rule(decisions, accept=("alta", "media"))) == {"a", "b", "c", "e"}
    assert set(dc.apply_rule(decisions, min_score=0.6)) == {"a", "c"}
    assert set(dc.apply_rule(decisions, accept_none=())) == {"a", "e"}


def test_fixed_rule_by_relation_and_kind():
    prop, both = ("property",), ("class", "property")
    decisions = [_dec("exact", relation="skos:exactMatch"), _dec("close"),
                 _dec("related", relation="skos:relatedMatch"),
                 _dec("untyped", relation=None),
                 _dec("p_exact", relation="skos:exactMatch", kinds=prop),
                 _dec("p_close", kinds=prop),
                 _dec("punned", relation="skos:relatedMatch", kinds=both),
                 _dec("p_none", none=True, kinds=prop)]
    keys = lambda **kw: set(dc.apply_rule(decisions, **kw))           # noqa: E731
    assert keys() == {d.key for d in decisions}                        # no filter: everything
    strong = keys(relations=dc.SKOS_STRONG)                            # properties: the same
    assert strong == {"exact", "close", "p_exact", "p_close", "p_none"}
    assert keys(relations=dc.SKOS_STRONG, property_relations=dc.SKOS_EXACT) == {
        "exact", "close", "p_exact", "p_none"}
    assert keys(relations=dc.SKOS_STRONG, property_relations=()) == {"exact", "close", "p_none"}
    assert keys(relations=dc.SKOS_EXACT, property_relations=None) == {
        "exact", "p_exact", "p_close", "p_none"}                       # None = any relation
    assert "punned" in keys(property_relations=())                     # punned counts as class


def test_suggestions_are_high_confidence_terms_the_rule_left_out():
    decisions = [_dec("kept"), _dec("related", relation="skos:relatedMatch"),
                 _dec("medium", relation="skos:relatedMatch", confidence="media"),
                 _dec("none", none=True), _dec("none_medium", none=True, confidence="media"),
                 _dec("i", status=dc.STATUS_INVALID)]
    auto = dc.apply_rule(decisions, relations=dc.SKOS_STRONG)
    assert set(auto) == {"kept", "none"}
    assert dc.suggested(decisions, auto) == {
        "related": {"term": D + "X", "relation": "skos:relatedMatch"}}
    assert set(dc.suggested(decisions, auto, levels=("alta", "media"))) == {"related", "medium"}


def test_console_is_in_spanish(targets, chroma):
    report, _ = _run(targets, chroma)
    text = dc.render_console(report)
    assert "Decisor de alineación (S7-T04)" in text and "Llamadas al LLM: 2" in text
    assert "Confianza: alta" in text


# --------------------------------------------------------------------------- real LLM

@pytest.mark.slow
def test_real_llm_decides_a_small_batch(targets, chroma):
    report = dc.decide(_concepts(), targets, embed_fn=FakeEmbed(), settings=FAST,
                       chroma_path=chroma, embedding_model="fake-model", sleep_fn=lambda s: None)
    assert report.calls >= 1 and not report.errors
    assert all(d.status in (dc.STATUS_DECIDED, dc.STATUS_INVALID) for d in report.decisions)
    print("\n" + dc.render_console(report))