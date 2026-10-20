"""S7-T08: labels in Spanish and English for an uploaded ontology (the LLM writes the missing
ones with the glossary of the LOPDP, in a module apart; no original label changes).

Fast tests: a fake LLM that answers the labels prompt, the small validated base of the decider
tests (its glossary: "evaluación de impacto" and "volumen de negocio"), temporary workspaces.
The slow test makes one real call."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, XSD

from src.transform import labels as lab
from src.transform.labels import (COMPLETE, PENDING, TRANSLATED, LabelItem, LabelSettings,
                                  TranslateReport, added_text, base_glossary, clean_label,
                                  entries_from, final_labels, glossary_sha, glossary_text,
                                  is_current, label_items, label_ontology, labels_for_workspace,
                                  labels_graph, labels_settings, parse_labels, render_console,
                                  translate, validate_answer)
from src.workspace.store import DONE, ERROR, open_ontology
from tests.test_alignment_propose import make_env

EX = "http://example.org/gdpr#"
GLOSSARY = [{"es": "titular", "en": ["Data Subject", "Data Owner"]}]
FAST = LabelSettings(batch_size=2, max_calls=10, pause_seconds=5, max_tokens=1024)
TTL = f"""
@prefix ex: <{EX}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
<http://example.org/gdpr> a owl:Ontology .
ex:DataSubject a owl:Class ; rdfs:label "Data Subject" ;
    rdfs:comment "A natural person whose personal data is processed."@en .
ex:Consent a owl:Class ; rdfs:label "Consent"@en, "Consentimiento"@es .
ex:Controller a owl:Class ; skos:prefLabel "Controller"@EN-GB .
ex:hasRecipient a owl:ObjectProperty .
ex:alice a owl:NamedIndividual, ex:DataSubject .
"""
SPANISH = {"DataSubject": "Titular", "Controller": "Responsable del tratamiento",
           "hasRecipient": "Tiene destinatario"}


class LabelsLLM:
    """Answers the labels prompt: Spanish from ``SPANISH``, English = the name split."""

    def __init__(self, error=None, errors=(), answers=None, raw=None, model="modelo-de-prueba"):
        self.error, self.errors, self.answers, self.raw = error, list(errors), answers, raw
        self.model, self.calls, self.contexts, self.prompts = model, 0, [], []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        self.prompts.append((prompt, max_tokens, json_mode))
        if self.errors:
            raise Exception(self.errors.pop(0))
        if self.error:
            raise Exception(self.error)
        if self.raw is not None:
            return self.raw
        out = []
        for block in context.split("=== ")[1:]:
            cid = block.split(" ===", 1)[0]
            name = re.search(r"^Concepto: (\S+) \(", block, re.M).group(1)
            answer = (self.answers or {}).get(name)
            if answer is None:
                answer = {"es": SPANISH.get(name, f"Etiqueta de {name}"),
                          "en": re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)}
            out.append({"concept": cid, **answer})
        return json.dumps({"labels": out}, ensure_ascii=False)


def _graph(ttl=TTL) -> Graph:
    return Graph().parse(data=ttl, format="turtle")


def _concepts(ttl=TTL):
    from src.alignment.sources import build_alignment_sources
    from src.ingest.ontology_loader import load_ontology
    import tempfile
    path = Path(tempfile.mkdtemp()) / "o.ttl"
    path.write_text(ttl, encoding="utf-8")
    return build_alignment_sources(load_ontology(path), proposals={}).concepts


def _item(name, **kw) -> LabelItem:
    return LabelItem(key=EX + name, name=name, kinds=("class",), **kw)


@pytest.fixture
def env(tmp_path):
    return make_env(tmp_path)


@pytest.fixture
def ws(env):
    return open_ontology("Mi GDPR.ttl", TTL.encode("utf-8"), cfg=env.cfg)


# ------------------------------------------------------------------ settings and prompt

def test_settings_defaults_config_and_validation():
    assert labels_settings({}) == LabelSettings() == LabelSettings(40, 20, 5.0, 4096, 200)
    assert labels_settings() == LabelSettings()                       # config.yaml = defaults
    block = {"batch_size": 3, "max_calls": 0, "pause_seconds": 1, "max_tokens": 512,
             "definition_chars": 50}
    assert labels_settings({"labels": block}) == LabelSettings(3, 0, 1.0, 512, 50)
    for bad in ({"batch_size": 0}, {"max_calls": -1}, {"pause_seconds": -1},
                {"max_tokens": 255}):
        with pytest.raises(ValueError, match="bloque labels"):
            labels_settings({"labels": bad})


def test_the_prompt_asks_for_json_with_both_languages():
    assert lab.PROMPT.name == "labels" and lab.PROMPT.version == 1
    for words in ('"labels"', '"concept"', '"es"', '"en"', "glosario", "sin idioma",
                  "titular", "data subject"):
        assert words in lab.PROMPT.text


# ------------------------------------------------------------------ inventory

def test_labels_are_read_by_language():
    items = {i.name: i for i in label_items(_graph(), _concepts())}
    assert set(items) == {"DataSubject", "Consent", "Controller", "hasRecipient"}   # no alice
    ds = items["DataSubject"]
    assert ds.untagged == ["Data Subject"] and ds.tagged == {} and ds.missing == ["es", "en"]
    assert ds.definition.startswith("A natural person")
    assert items["Consent"].missing == [] and items["Consent"].tagged == {
        "en": ["Consent"], "es": ["Consentimiento"]}
    assert items["Controller"].tagged == {"en": ["Controller"]}          # EN-GB is en
    assert items["Controller"].missing == ["es"]
    assert items["hasRecipient"].kinds == ("property",)


def test_empty_repeated_and_non_literal_labels_are_ignored():
    g = _graph(TTL + f'<{EX}DataSubject> rdfs:label "  ", <{EX}Consent> ; '
                     f'skos:prefLabel "Data Subject" .\n')
    ds = next(i for i in label_items(g, _concepts()) if i.name == "DataSubject")
    assert ds.untagged == ["Data Subject"] and ds.tagged == {}


# ------------------------------------------------------------------ glossary

def test_glossary_of_the_base_has_the_kept_concepts_only(env):
    assert base_glossary(env.cfg) == [
        {"es": "evaluación de impacto", "en": ["Impact Assessment"]},
        {"es": "volumen de negocio", "en": ["Business Volume", "Turnover"]}]   # + OntoPriv name


def test_glossary_without_a_base_is_empty(tmp_path):
    cfg = {"outputs": {"dir": str(tmp_path / "nada")},
           "review": {"decisions_file": str(tmp_path / "nada" / "log.jsonl")}}
    assert base_glossary(cfg) == []
    assert glossary_text([]) == "Glosario: (vacio)"


def test_glossary_merges_repeated_terms_sorts_and_skips(env, monkeypatch):
    from dataclasses import replace
    from src.alignment import ai_concepts
    real = ai_concepts.ai_concepts

    def more(data, *a, **kw):
        out = real(data, *a, **kw)
        clone = [c for c in out if c.name == "ImpactAssessment"][0]      # approved
        return out + [replace(clone, label="Evaluación de impacto ", name="ImpactAssessment"),
                      replace(clone, label="evaluación  de impacto", name="DPIA"),
                      replace(clone, label="Auditoría", name="Audit"),
                      replace(clone, label=None, name="NoLabel"),
                      replace(clone, key="ai:Undecided", label="sin decidir")]
    monkeypatch.setattr(ai_concepts, "ai_concepts", more)
    assert base_glossary(env.cfg) == [
        {"es": "Auditoría", "en": ["Audit"]},
        {"es": "evaluación de impacto", "en": ["Impact Assessment", "DPIA"]},
        {"es": "volumen de negocio", "en": ["Business Volume", "Turnover"]}]


def test_glossary_text_and_fingerprint():
    text = glossary_text(GLOSSARY)
    assert text == "Glosario (espanol = ingles):\n- titular = Data Subject / Data Owner"
    assert glossary_sha(GLOSSARY) != glossary_sha([]) and len(glossary_sha([])) == 64


# ------------------------------------------------------------------ answers

def test_parse_labels_accepts_the_usual_shapes():
    one = [{"concept": "C1", "es": "a", "en": "b"}]
    assert parse_labels(json.dumps({"labels": one})) == one
    assert parse_labels("```json\n" + json.dumps({"labels": one}) + "\n```") == one
    assert parse_labels("```json\n" + json.dumps(one) + "\n```") == one        # bare, fenced
    assert parse_labels("Aquí va: " + json.dumps({"labels": one}) + " listo") == one
    assert parse_labels(json.dumps(one + ["x", 3])) == one
    for bad in ("", "nada", json.dumps({"otra": one}), json.dumps({"labels": "x"})):
        with pytest.raises(ValueError, match="lista de etiquetas"):
            parse_labels(bad)


def test_clean_label_and_validate():
    assert clean_label('  "Dato  personal."  ') == "Dato personal"
    assert clean_label("“Titular”") == "Titular" and clean_label("Art. 5") == "Art. 5"
    assert clean_label("Línea\nuna") == "Línea una"
    assert clean_label("") is None and clean_label(None) is None and clean_label('"."') is None
    assert clean_label("x" * 150) == "x" * 150 and clean_label("x" * 151) is None
    assert validate_answer({"es": "Titular", "en": "Data subject"}) == {
        "es": "Titular", "en": "Data subject"}
    assert validate_answer({"es": "Titular", "en": " "}) is None
    assert validate_answer({"es": "Titular"}) is None


# ------------------------------------------------------------------ translate

def test_translate_in_batches_with_pause_and_cache(tmp_path):
    items = [_item(n) for n in ("DataSubject", "Controller", "hasRecipient")]
    llm, pauses, cache = LabelsLLM(), [], tmp_path / "cache.jsonl"
    r = translate(items, GLOSSARY, llm=llm, settings=FAST, cache_path=cache,
                  sleep_fn=pauses.append)
    assert (r.calls, r.cached, llm.calls, pauses) == (2, 0, 2, [5])
    assert r.answers[EX + "DataSubject"] == {"es": "Titular", "en": "Data Subject"}
    assert all(c.startswith(glossary_text(GLOSSARY) + "\n\n=== C1 ===\nConcepto: ")
               for c in llm.contexts)
    assert "=== C2 ===" in llm.contexts[0] and "=== C2 ===" not in llm.contexts[1]
    assert {(p, m, j) for p, m, j in llm.prompts} == {(lab.PROMPT.text, 1024, True)}
    saved = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 3 and saved[0]["prompt"] == lab.PROMPT.manifest()
    assert saved[0]["model"] == "modelo-de-prueba" and saved[0]["concept_key"] == EX + "DataSubject"

    again = translate(items, GLOSSARY, llm=LabelsLLM(), settings=FAST, cache_path=cache)
    assert (again.calls, again.cached) == (0, 3) and again.answers == r.answers
    other = translate(items, [], llm=LabelsLLM(), settings=FAST, cache_path=cache)
    assert (other.calls, other.cached) == (2, 0)                       # another glossary
    model = translate(items[:1], GLOSSARY, llm=LabelsLLM(model="otro"), settings=FAST,
                      cache_path=cache)
    assert (model.calls, model.cached) == (1, 0)                       # another model


def test_the_key_depends_on_what_the_llm_sees():
    body = lab.label_body(_item("Consent"), FAST)
    assert body == "Concepto: Consent (tipo: class)\nEtiquetas actuales: ninguna"
    key = lab.cache_key(body, "m", "g")
    hotter = lab.cache_key(body, "m", "g", {"temperature": 0.7})
    assert key == lab.cache_key(body, "m", "g", {}) != hotter
    assert key != lab.cache_key(body + " ", "m", "g") != lab.cache_key(body, "n", "g")
    full = lab.label_body(_item("Consent", tagged={"es": ["Consentimiento"], "en": ["Consent"]},
                                untagged=["consent"], definition="Una  " + "x" * 300), FAST)
    assert full.splitlines()[1] == ('Etiquetas actuales: "Consent" (en); "Consentimiento" (es); '
                                    '"consent" (sin idioma)')            # sorted: stable key
    assert full.splitlines()[2] == "Definicion: Una " + "x" * 196


def test_without_llm_nothing_is_asked_and_the_cache_still_answers(tmp_path):
    cache = tmp_path / "c.jsonl"
    translate([_item("DataSubject")], GLOSSARY, llm=LabelsLLM(), settings=FAST, cache_path=cache)
    r = translate([_item("DataSubject"), _item("Controller")], GLOSSARY, llm=None,
                  settings=FAST, cache_path=cache, model_name="modelo-de-prueba")
    assert (r.calls, r.cached, r.call_limit_reached) == (0, 1, False)
    assert list(r.answers) == [EX + "DataSubject"]


def test_call_limit(tmp_path):
    items = [_item(f"C{n}x") for n in range(5)]
    r = translate(items, GLOSSARY, llm=LabelsLLM(),
                  settings=LabelSettings(batch_size=2, max_calls=1, pause_seconds=0),
                  cache_path=tmp_path / "c.jsonl")
    assert (r.calls, len(r.answers), r.call_limit_reached) == (1, 2, True)
    r0 = translate(items, GLOSSARY, llm=LabelsLLM(), settings=LabelSettings(max_calls=0))
    assert (r0.calls, r0.answers, r0.call_limit_reached) == (0, {}, True)


def test_quota_retries_once_then_stops(tmp_path):
    waits, llm = [], LabelsLLM(error="429 RESOURCE_EXHAUSTED quota")
    r = translate([_item("A"), _item("B"), _item("C")], GLOSSARY, llm=llm,
                  settings=LabelSettings(batch_size=1, pause_seconds=0), sleep_fn=waits.append)
    from src.alignment.justify import QUOTA_WAIT_SECONDS
    assert (llm.calls, r.calls, r.quota_exhausted, waits) == (2, 1, True, [QUOTA_WAIT_SECONDS])
    assert r.errors == [{"concepts": ["A"], "error": "429 RESOURCE_EXHAUSTED quota"}]


def test_a_transient_error_is_retried_and_another_error_skips_the_batch():
    from src.alignment.justify import TRANSIENT_WAIT_SECONDS
    waits, llm = [], LabelsLLM(errors=["503 UNAVAILABLE"])
    r = translate([_item("A")], GLOSSARY, llm=llm, settings=FAST, sleep_fn=waits.append)
    assert (llm.calls, r.calls, waits, r.errors) == (2, 1, [TRANSIENT_WAIT_SECONDS], [])
    assert EX + "A" in r.answers
    llm = LabelsLLM(errors=["400 bad request"])
    r = translate([_item("A"), _item("B")], GLOSSARY, llm=llm,
                  settings=LabelSettings(batch_size=1, pause_seconds=0))
    assert (llm.calls, r.calls, list(r.answers), r.quota_exhausted) == (2, 2, [EX + "B"], False)
    assert r.errors[0]["error"] == "400 bad request"
    r = translate([_item("A")], GLOSSARY, llm=LabelsLLM(error="x" * 400), settings=FAST)
    assert r.errors[0]["error"] == "x" * 300


def test_bad_json_unknown_ids_and_invalid_answers(tmp_path):
    r = translate([_item("A")], GLOSSARY, llm=LabelsLLM(raw="no es json"), settings=FAST,
                  cache_path=tmp_path / "c.jsonl")
    assert r.answers == {} and "lista de etiquetas" in r.errors[0]["error"]
    raw = json.dumps({"labels": [{"concept": "c1", "es": "Uno", "en": "One"},
                                 {"concept": "C1", "es": "Otro", "en": "Other"},
                                 {"concept": "C9", "es": "Nueve", "en": "Nine"},
                                 {"concept": "C2", "es": "", "en": "Two"}]})
    r = translate([_item("A"), _item("B")], GLOSSARY, llm=LabelsLLM(raw=raw), settings=FAST,
                  cache_path=tmp_path / "c.jsonl")
    assert r.answers == {EX + "A": {"es": "Uno", "en": "One"}}         # first answer wins
    assert r.invalid == [EX + "B"]
    lines = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1                                              # invalid: not cached
    none_valid = json.dumps({"labels": [{"concept": "C1", "es": "", "en": ""}]})
    translate([_item("A")], GLOSSARY, llm=LabelsLLM(raw=none_valid), settings=FAST,
              cache_path=tmp_path / "otra.jsonl")
    assert not (tmp_path / "otra.jsonl").exists()


def test_a_broken_cache_line_is_skipped(tmp_path):
    cache = tmp_path / "c.jsonl"
    translate([_item("A")], GLOSSARY, llm=LabelsLLM(), settings=FAST, cache_path=cache)
    cache.write_text("{roto\n" + '{"sin": "clave"}\n3\n' + cache.read_text(encoding="utf-8"),
                     encoding="utf-8")
    r = translate([_item("A")], GLOSSARY, llm=None, settings=FAST, cache_path=cache,
                  model_name="modelo-de-prueba")
    assert r.cached == 1


# ------------------------------------------------------------------ result

def test_only_missing_labels_are_added_and_untagged_originals_are_kept():
    items = [_item("DataSubject", untagged=["Data Subject"]),
             _item("Controller", tagged={"en": ["Controller"]}),
             _item("Consent", tagged={"es": ["Consentimiento"], "en": ["Consent"]}),
             _item("Pizza")]
    report = TranslateReport(answers={
        EX + "DataSubject": {"es": "Titular", "en": "Data subject"},
        EX + "Controller": {"es": "Responsable del tratamiento", "en": "Data controller"}})
    entries = {e["name"]: e for e in entries_from(items, report)}
    ds = entries["DataSubject"]
    assert ds["status"] == TRANSLATED
    assert ds["added"] == {"es": "Titular", "en": "Data Subject"}      # the original text
    assert ds["existing"] == {"es": [], "en": [], "none": ["Data Subject"]}
    assert entries["Controller"]["added"] == {"es": "Responsable del tratamiento"}
    assert entries["Consent"]["status"] == COMPLETE and entries["Consent"]["added"] == {}
    assert entries["Pizza"]["status"] == PENDING and entries["Pizza"]["added"] == {}
    assert final_labels(entries["Controller"]) == {"es": "Responsable del tratamiento",
                                                   "en": "Controller"}
    assert final_labels(entries["Pizza"]) == {"es": None, "en": None}
    assert [added_text(entries[n]) for n in ("DataSubject", "Controller", "Consent", "Pizza")] \
        == ["es, en", "es", "", "pendiente"]
    assert lab.summarize(list(entries.values())) == {
        "concepts": 4, "complete": 1, "translated": 2, "pending": 1,
        "added": {"es": 2, "en": 1}}


def test_label_ontology_with_and_without_ai(tmp_path):
    g, concepts = _graph(), _concepts()
    llm = LabelsLLM()
    r = label_ontology(g, concepts, llm=llm, cfg={}, glossary=GLOSSARY,
                       cache_path=tmp_path / "c.jsonl", sleep_fn=lambda s: None)
    assert r["summary"] == {"concepts": 4, "complete": 1, "translated": 3, "pending": 0,
                            "added": {"es": 3, "en": 2}}
    assert r["ai"] == {"available": True, "model": "modelo-de-prueba", "calls": 1, "cached": 0,
                       "invalid": 0, "quota_exhausted": False, "call_limit_reached": False,
                       "errors": []}
    assert r["glossary"] == {"entries": 1, "sha256": glossary_sha(GLOSSARY)}
    assert r["prompt"] == lab.PROMPT.manifest() and r["method_version"] == lab.METHOD_VERSION
    off = label_ontology(g, concepts, use_llm=False, cfg={}, glossary=GLOSSARY)
    assert off["summary"]["pending"] == 3 and off["ai"]["available"] is False
    assert off["ai"]["error"] == "IA desactivada (--sin-ia)"
    both = ' ; rdfs:label "Titular"@es, "Data subject"@en ;\n    rdfs:comment'
    done = _graph(TTL.replace(' ;\n    rdfs:comment', both))
    one = [c for c in concepts if c.name == "DataSubject"]
    nothing = label_ontology(done, one, llm=LabelsLLM(), cfg={}, glossary=GLOSSARY)
    assert nothing["ai"] == {"available": True, "needed": False, "model": None, "calls": 0,
                             "cached": 0, "invalid": 0, "quota_exhausted": False,
                             "call_limit_reached": False, "errors": []}


def test_the_errors_of_the_ai_are_kept():
    r = label_ontology(_graph(), _concepts(), llm=LabelsLLM(error="400 bad request"), cfg={},
                       glossary=GLOSSARY)
    assert r["ai"]["errors"] == ["400 bad request"] and r["summary"]["pending"] == 3
    quota = label_ontology(_graph(), _concepts(), llm=LabelsLLM(error="429 quota"), cfg={},
                           glossary=GLOSSARY, sleep_fn=lambda s: None)
    assert quota["ai"]["quota_exhausted"] is True and quota["ai"]["call_limit_reached"] is False


def test_a_missing_api_key_is_reported(monkeypatch):
    import src.ai.llm_client as client

    def no_key(*a, **kw):
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr(client, "LLMClient", no_key)
    r = label_ontology(_graph(), _concepts(), cfg={}, glossary=GLOSSARY)
    assert r["ai"]["available"] is False and "LLM_API_KEY" in r["ai"]["error"]
    assert r["summary"]["pending"] == 3


# ------------------------------------------------------------------ workspace

def test_labels_for_workspace_saves_and_marks_the_step(ws, env):
    llm = LabelsLLM()
    r = labels_for_workspace(ws, llm=llm, cfg=env.cfg, sleep_fn=lambda s: None)
    assert lab.load_result(ws) == r and r["glossary"]["entries"] == 2
    assert "volumen de negocio = Business Volume / Turnover" in llm.contexts[0]
    step = ws.step("labels")
    assert step["status"] == DONE and step["detail"] == "3 en español y 2 en inglés agregadas"
    assert (step["translated"], step["pending"]) == (3, 0)
    assert (ws.work_dir / lab.CACHE_FILE).exists()
    again = labels_for_workspace(ws, llm=llm, cfg=env.cfg)
    assert llm.calls == 1 and again["ai"]["cached"] == 3


def test_the_pause_uses_time_sleep_unless_told(ws, env, monkeypatch):
    pauses = []
    monkeypatch.setattr(lab.time, "sleep", pauses.append)
    cfg = {**env.cfg, "labels": {"batch_size": 1, "pause_seconds": 2}}
    labels_for_workspace(ws, llm=LabelsLLM(), cfg=cfg)
    assert pauses == [2, 2]                                           # 3 calls, 2 pauses


def test_without_ai_by_config_and_errors_mark_the_step(ws, env, monkeypatch):
    offline = {**env.cfg, "auto": {**env.cfg["auto"], "use_llm": False}}
    r = labels_for_workspace(ws, llm=LabelsLLM(), cfg=offline)
    assert r["summary"]["pending"] == 3 and r["ai"]["available"] is False
    monkeypatch.setattr(lab, "label_ontology", lambda *a, **kw: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        labels_for_workspace(ws, cfg=env.cfg)
    assert ws.step("labels")["status"] == ERROR and "division" in ws.step("labels")["detail"]
    assert lab.load_result(ws) == r                                    # the last good one stays


def test_is_current(ws, env):
    assert is_current(None, env.cfg) is False
    r = labels_for_workspace(ws, llm=LabelsLLM(), cfg=env.cfg)
    assert is_current(r, env.cfg) and is_current(r, env.cfg, use_llm=False)
    assert not is_current({**r, "method_version": 0}, env.cfg)
    assert not is_current({**r, "prompt": {**r["prompt"], "sha256": "x"}}, env.cfg)
    assert not is_current({**r, "glossary": {"entries": 0, "sha256": glossary_sha([])}},
                          env.cfg)
    pending = {**r, "summary": {**r["summary"], "pending": 1}}
    assert not is_current(pending, env.cfg) and is_current(pending, env.cfg, use_llm=False)
    offline = {**env.cfg, "auto": {**env.cfg["auto"], "use_llm": False}}
    assert is_current(pending, offline)


# ------------------------------------------------------------------ module

def test_the_module_has_only_the_added_labels(ws, env):
    r = labels_for_workspace(ws, llm=LabelsLLM(), cfg=env.cfg)
    g = labels_graph(r, iri="http://example.org/gdpr-labels", onto_iri="http://example.org/gdpr",
                     name="Mi GDPR.ttl", created="2026-10-20T12:00:00+00:00")
    node = URIRef("http://example.org/gdpr-labels")
    assert set(g.subjects(RDF.type, OWL.Ontology)) == {node}
    assert g.value(node, OWL.imports) == URIRef("http://example.org/gdpr")
    assert g.value(node, OWL.versionInfo) == Literal("1")
    assert g.value(node, DCTERMS.created) == Literal("2026-10-20T12:00:00+00:00",
                                                     datatype=XSD.dateTime)
    assert str(g.value(node, RDFS.label)) == "Etiquetas en español e inglés de Mi GDPR.ttl"
    assert "Ninguna etiqueta original se cambia" in str(g.value(node, RDFS.comment))
    labels = {(str(s).split("#")[-1], o.language, str(o)) for s, o in g.subject_objects(RDFS.label)
              if s != node}
    assert labels == {("DataSubject", "es", "Titular"), ("DataSubject", "en", "Data Subject"),
                      ("Controller", "es", "Responsable del tratamiento"),
                      ("hasRecipient", "es", "Tiene destinatario"),
                      ("hasRecipient", "en", "has Recipient")}
    assert len(g) == 6 + 5
    assert labels_graph({"entries": []}, iri="a", onto_iri="b", name="n", created="c") is None


# ------------------------------------------------------------------ console and CLI

def test_console_and_ai_line(ws, env):
    r = labels_for_workspace(ws, llm=LabelsLLM(), cfg=env.cfg)
    text = render_console(r)
    assert text.startswith("Etiquetas es/en (S7-T08)")
    assert "Conceptos: 4 · ya tenían es y en: 1" in text
    assert "Completadas por la IA: 3 (agregadas: 3 en español, 2 en inglés) · pendientes: 0" \
        in text
    assert "Glosario de la base: 2 términos de la LOPDP" in text
    assert "IA: 1 llamadas, 0 desde la caché, 0 respuestas inválidas" in text
    assert "    DataSubject: Titular (es) | Data Subject (en)" in text
    assert "Consent:" not in text and len(render_console(r, examples=1).splitlines()) == 7
    line = lab.ai_line
    assert line({"ai": {"available": False, "error": "x"}}) == "no usada (x)"
    assert line({"ai": {"available": True, "needed": False}}).startswith("no hizo falta")
    assert line({"ai": {"available": True, "quota_exhausted": True}}).endswith(
        "(cuota agotada: vuelve a correr mañana)")
    assert "(límite de llamadas" in line({"ai": {"available": True, "call_limit_reached": True}})
    assert line({"ai": {"available": True, "errors": ["boom", "otro"]}}).endswith("; error: boom")


def test_cli(tmp_path, monkeypatch, capsys, env):
    monkeypatch.setattr("src.config.load_config", lambda: env.cfg)
    path = tmp_path / "gdpr.ttl"
    path.write_text(TTL, encoding="utf-8")
    assert lab._run([str(path), "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Etiquetas es/en (S7-T08)" in out and "pendientes: 3" in out
    assert "IA: no usada (IA desactivada (--sin-ia))" in out and "labels.json" in out
    assert lab._run([str(tmp_path / "no.ttl")]) == 1
    bad = tmp_path / "o.owx"
    bad.write_text("<x/>", encoding="utf-8")
    assert lab._run([str(bad)]) == 1
    assert "Error:" in capsys.readouterr().out


# ------------------------------------------------------------------ real LLM

@pytest.mark.slow
def test_real_llm_labels_a_few_gdprtext_concepts(tmp_path):
    """One real call (spends quota) on 8 GDPRtEXT concepts with the real glossary."""
    if not os.environ.get("LLM_API_KEY"):
        from src import config  # noqa: F401  (loads .env)
    if not os.environ.get("LLM_API_KEY"):
        pytest.skip("Falta LLM_API_KEY en .env")
    gdpr = Path(__file__).resolve().parents[1] / "data" / "input" / "gdprtext-ontology.ttl"
    if not gdpr.exists():
        pytest.skip("Falta data/input/gdprtext-ontology.ttl")
    from src.alignment.sources import build_alignment_sources
    from src.ingest.ontology_loader import load_ontology
    from src.transform.convert import load_graph
    wanted = {"DataSubject", "Controller", "Processor", "Consent", "PersonalData",
              "RightOfErasure", "DataBreach", "ShareDataWithThirdParty"}
    concepts = [c for c in build_alignment_sources(load_ontology(gdpr), proposals={}).concepts
                if c.name in wanted]
    assert len(concepts) == 8
    r = label_ontology(load_graph(gdpr), concepts, cfg={"labels": {"pause_seconds": 0}},
                       cache_path=tmp_path / "c.jsonl")
    print("\n" + render_console(r, examples=len(concepts)))
    assert r["ai"]["calls"] == 1 and not r["ai"]["errors"]
    assert r["summary"]["pending"] == 0 and r["summary"]["translated"] == 8
    final = {e["name"]: final_labels(e) for e in r["entries"]}
    assert final["DataSubject"]["en"] == "Data Subject"                # GDPRtEXT's own text
    assert "titular" in final["DataSubject"]["es"].lower()             # term of the glossary