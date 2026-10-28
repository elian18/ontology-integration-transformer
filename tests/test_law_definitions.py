from __future__ import annotations

import csv
import json
import re
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, SKOS

from app.services import align_ontology as svc
from src.alignment import law_definitions as ld
from src.alignment.law_definitions import (ARTICLES, INVALID, LAW_CONCEPT, NO_SOURCE,
                                           NOT_APPLICABLE, PENDING, DefinitionSettings,
                                           ask_definitions, definition_settings, validate_answer)
from src.alignment.law_evidence import align_with_law
from src.alignment.user_alignment import align_workspace, render_console
from src.workspace.store import DONE, KIND_LAW, KIND_ONTOLOGY, open_upload
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_propose import DeciderLLM, Embed, make_env, targets  # noqa: F401

EX = "http://example.org/gdpr#"
TTL = f"""
@prefix ex: <{EX}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/gdpr> a owl:Ontology .
ex:Consent a owl:Class .
ex:PersonalData a owl:Class ; rdfs:comment "Information about an identified person."@en .
ex:DataController a owl:Class .
ex:Erasure a owl:Class ; rdfs:label "erasure" .
ex:Pizza a owl:Class .
"""
LAW = ("Artículo 1°. Datos personales. La información personal de una persona.\n"
       "Artículo 2°. Responsable. El controller decide sobre el tratamiento.\n"
       "Artículo 3°. Consentimiento. El consentimiento del titular es libre.\n"
       "Artículo 4°. Supresión. El titular puede pedir la erasure de sus datos.\n"
       "Artículo 5°. Otros. Nada que ver con lo anterior.\n")
EXTRACTED = {
    2: [{"name": "DataController", "label": "Responsable", "type": "class", "article": 2,
         "definition": "Quien decide sobre el tratamiento de los datos personales."}],
    3: [{"name": "Consent", "label": "Consentimiento", "type": "class", "article": 3,
         "definition": "Manifestación de voluntad libre e informada del titular."}],
}
DEFINED = {"Erasure": "Derecho del titular a pedir que se borren sus datos."}
CFG = {"law_definitions": {"enabled": True, "pause_seconds": 0},
       "extraction": {"pause_seconds": 0}}
LABELS = ["Separar la ley en artículos", "Indexar los artículos de la ley",
          "Heredar lo validado", "Extraer los conceptos de la ley con la IA",
          "Aceptar los conceptos por regla", "Completar las definiciones con la ley",
          "Buscar candidatos del DPV", "Completar las etiquetas en español e inglés",
          "Buscar en la ley los artículos de cada concepto", "Escribir la ontología alineada",
          "Preparar la descarga"]


class LawEmbed(Embed):
    """Embed of the decider tests plus one direction of its own for "erasure"."""

    def __call__(self, texts):
        out = []
        for t, v in zip(texts, super().__call__(texts)):
            w = np.zeros(20)
            if "erasure" in t.lower():
                w[16] = 1.0
            else:
                w[:16] = v
            out.append(w)
        return out


class LawLLM(DeciderLLM):
    """Extracts EXTRACTED from the law, writes the definitions of DEFINED (the others: none)
    and decides and labels as DeciderLLM."""

    def __init__(self, *args, defined=None, article=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.defined = DEFINED if defined is None else defined
        self.article, self.asked = article, []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        if "Extrae los conceptos" in prompt:
            self.calls += 1
            numbers = [int(n) for n in re.findall(r"=== ARTICULO (\d+) ", context)]
            return "\n".join(json.dumps(c, ensure_ascii=False)
                             for n in numbers for c in EXTRACTED.get(n, []))
        if '"definitions"' in prompt:
            self.calls += 1
            self.contexts.append(context)
            if self.error:
                raise Exception(self.error)
            out = []
            for cid, block in re.findall(r"=== (C\d+) ===\n(.*?)(?=\n\n=== |\Z)", context, re.S):
                name = re.search(r"^Concepto: (\S+) \(", block, re.M).group(1)
                numbers = re.search(r"^Articulos: (.+)$", block, re.M).group(1).split(", ")
                self.asked.append(name)
                text = self.defined.get(name, "")
                out.append({"concept": cid, "definition": text,
                            "article": (self.article or int(numbers[0])) if text else None})
            return json.dumps({"definitions": out}, ensure_ascii=False)
        return super().ask(prompt, context, max_tokens, json_mode)


@pytest.fixture
def env(tmp_path, targets):  # noqa: F811
    env = make_env(tmp_path)
    env.targets = targets
    env.cfg = {**env.cfg, **CFG}
    return env


def _kw(env, **extra):
    return dict(base=env.base, targets=env.targets, embed_fn=LawEmbed(), cfg=env.cfg,
                chroma_path=env.chroma, sleep_fn=lambda s: None, **extra)


def _align(env, llm, ttl=TTL, law=LAW, **extra):
    seen = []
    ws, m, path = align_with_law("Mi GDPR.ttl", ttl.encode("utf-8"), "ley.txt",
                                 law.encode("utf-8"), llm=llm, progress=seen.append,
                                 **_kw(env, **extra))
    return SimpleNamespace(ws=ws, manifest=m, zip=path, seen=seen)


def _csv(ws):
    with open(ws.output_dir / "alignment.csv", encoding="utf-8-sig") as handle:
        return {r["concept_name"]: r for r in csv.DictReader(handle)}


def _by_name(ws):
    return {e["name"]: e for e in ld.load_result(ws)["entries"]}


# ------------------------------------------------------------------ settings

def test_settings():
    assert definition_settings({}) == DefinitionSettings()
    assert not DefinitionSettings().enabled                  # without the block: off
    s = definition_settings({"law_definitions": {"enabled": True, "min_missing": 0.2,
                                                 "from_articles": False, "max_calls": 0}})
    assert (s.enabled, s.min_missing, s.from_law_concepts, s.from_articles, s.max_calls) == (
        True, 0.2, True, False, 0)
    assert s.to_decider and not definition_settings(
        {"law_definitions": {"to_decider": False}}).to_decider
    for bad in ({"min_missing": 1.5}, {"articles_per_concept": 0}, {"batch_size": 0},
                {"article_chars": 50}, {"max_calls": -1}, {"max_tokens": 10}):
        with pytest.raises(ValueError, match="bloque law_definitions"):
            definition_settings({"law_definitions": bad})


def test_ten_percent_or_more_missing_asks_the_law():
    on = DefinitionSettings(enabled=True)
    assert ld.skip_reason(on, 10, 1) is None and ld.skip_reason(on, 30, 3) is None   # 10 %
    assert ld.skip_reason(on, 11, 1) == ld.SKIP_FEW and ld.skip_reason(on, 5, 0) == ld.SKIP_FEW
    assert ld.skip_reason(DefinitionSettings(), 5, 5) == ld.SKIP_DISABLED


def test_the_project_turns_it_on():
    s = definition_settings()
    assert s.enabled and s.from_law_concepts and s.from_articles and s.min_missing == 0.10
    assert not s.to_decider                                  # S8-T05: they did not help it


def test_the_prompt_is_ascii_and_names_the_language():
    assert ld.PROMPT.text.isascii() and ld.PROMPT.version == 1
    assert "{idioma}" in ld.PROMPT.text and "{idioma}" not in ld.prompt_text("es")
    assert "La ley esta en espanol" in ld.prompt_text("es")
    assert "La ley esta en ingles" in ld.prompt_text("en")


# ------------------------------------------------------------------ answers

def test_validate_answer():
    ok = validate_answer({"definition": ' "Derecho del titular a borrar sus datos." ',
                          "article": "4°"}, [4, 2])
    assert ok == {"status": ARTICLES, "definition": "Derecho del titular a borrar sus datos.",
                  "article": 4}
    for empty in ("", "No aplica.", "n/a", None):
        assert validate_answer({"definition": empty, "article": None}, [4])["status"] == \
            NOT_APPLICABLE
    assert validate_answer({"definition": "Algo definido.", "article": 9}, [4])["status"] == INVALID
    assert validate_answer({"definition": "Algo definido.", "article": None}, [4])["status"] == \
        INVALID
    assert validate_answer({"definition": "x" * 601, "article": 4}, [4])["status"] == INVALID


def test_parse_definitions():
    one = {"concept": "C1", "definition": "d", "article": 1}
    assert ld.parse_definitions(json.dumps({"definitions": [one]})) == [one]
    assert ld.parse_definitions(json.dumps([one, "x"])) == [one]
    assert ld.parse_definitions("```json\n" + json.dumps({"definitions": [one]}) + "\n```") == \
        [one]
    with pytest.raises(ValueError, match="definiciones en JSON"):
        ld.parse_definitions("no es json")


# ------------------------------------------------------------------ law concepts

def _law_entry(name, definition, status="accepted", kind="class", label=None, articles=(2,)):
    return {"name": name, "label": label, "definition": definition, "kind": kind,
            "articles": list(articles), "status": status}


def _source(name, label=None, kinds=("class",)):
    from src.alignment.sources import SourceConcept
    return SourceConcept(key=EX + name, origin="ontology", name=name, kinds=kinds, label=label,
                         iri=EX + name)


def test_match_law_concepts_by_name_label_and_never_on_a_tie():
    entries = [_law_entry("DataController", "Quien decide sobre el tratamiento."),
               _law_entry("Holder", "La persona dueña de los datos.", label="Titular"),
               _law_entry("Noise", "Algo que la regla rechazó.", status="rejected"),
               _law_entry("Recipient", "Quien recibe los datos.", label="Destino"),
               _law_entry("Receiver", "Otra cosa distinta.", label="Destino"),
               _law_entry("hasPurpose", "Relaciona un tratamiento con su fin.", kind="property")]
    concepts = [_source("DataController"), _source("DataOwner", label="titular"),
                _source("Noise"), _source("Target", label="destino"),
                _source("HasPurpose")]
    found = ld.match_law_concepts(concepts, entries, embed_fn=LawEmbed(), cfg={})
    assert {k.rsplit("#", 1)[1]: (v["entry"]["name"], v["method"]) for k, v in found.items()} == {
        "DataController": ("DataController", "same_name"), "DataOwner": ("Holder", "same_name")}
    assert ld.match_law_concepts([], entries) == {} and ld.match_law_concepts(concepts, []) == {}


def test_the_law_digest_follows_the_accepted_concepts():
    law = {"entries": [_law_entry("A", "Una definición larga de A."),
                       _law_entry("B", "Otra.", status="rejected")]}
    other = {"entries": [_law_entry("A", "Otra definición de A, distinta.")]}
    assert ld.law_digest(law) == ld.law_digest({"entries": law["entries"][:1]})
    assert ld.law_digest(law) != ld.law_digest(other) and ld.law_digest(None) is None


# ------------------------------------------------------------------ the AI

def _articles():
    return {n: {"number": n, "title": f"Título {n}", "text": f"Texto del artículo {n}."}
            for n in (1, 2, 3, 4)}


def test_ask_definitions_batches_caches_and_limits(tmp_path):
    items = [(_source("Erasure"), [4, 1]), (_source("Pizza"), [1, 2]),
             (_source("Retention"), [3])]
    llm = LawLLM(defined={"Erasure": "Derecho a borrar.", "Retention": "Tiempo de guarda."})
    cache = tmp_path / "cache.jsonl"
    s = DefinitionSettings(batch_size=2, pause_seconds=0)
    r = ask_definitions(items, _articles(), "es", llm=llm, settings=s, cache_path=cache)
    assert (r.calls, r.cached, llm.asked) == (2, 0, ["Erasure", "Pizza", "Retention"])
    assert {k.rsplit("#", 1)[1]: (a["status"], a["article"]) for k, a in r.answers.items()} == {
        "Erasure": (ARTICLES, 4), "Pizza": (NOT_APPLICABLE, None), "Retention": (ARTICLES, 3)}
    first = llm.contexts[0]
    assert first.startswith("=== ARTICULO 1: Título 1 ===\nTexto del artículo 1.")
    assert "=== ARTICULO 3" not in first and "Articulos: 4, 1" in first
    again = ask_definitions(items, _articles(), "es", llm=llm, settings=s, cache_path=cache)
    assert (again.calls, again.cached, again.answers) == (0, 3, r.answers)
    other = ask_definitions(items, _articles(), "en", llm=LawLLM(), settings=s,
                            cache_path=cache)
    assert other.cached == 0                                  # another language: asked again
    limited = ask_definitions(items, _articles(), "es", llm=LawLLM(), cache_path=None,
                              settings=DefinitionSettings(batch_size=1, max_calls=1))
    assert limited.calls == 1 and limited.call_limit_reached and len(limited.answers) == 1
    assert ask_definitions(items, _articles(), "es", llm=None, settings=s).answers == {}


def test_an_article_out_of_the_list_is_invalid_and_not_asked_again(tmp_path):
    llm = LawLLM(article=9)
    cache = tmp_path / "cache.jsonl"
    r = ask_definitions([(_source("Erasure"), [4])], _articles(), "es", llm=llm,
                        cache_path=cache)
    assert r.answers[EX + "Erasure"] == {"status": INVALID, "definition": None, "article": 9}
    again = ask_definitions([(_source("Erasure"), [4])], _articles(), "es", llm=llm,
                            cache_path=cache)
    assert again.calls == 0 and again.cached == 1


def test_a_quota_error_stops_and_a_bad_answer_is_reported():
    items = [(_source("Erasure"), [4]), (_source("Pizza"), [1])]
    waits = []
    quota = ask_definitions(items, _articles(), "es", llm=LawLLM(error="429 RESOURCE_EXHAUSTED"),
                            settings=DefinitionSettings(batch_size=1, pause_seconds=0),
                            sleep_fn=waits.append)
    assert quota.quota_exhausted and quota.calls == 1 and quota.answers == {}
    assert waits == [60]                                      # one retry after a minute

    class Broken(LawLLM):
        def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
            self.calls += 1
            return "sin json"

    bad = ask_definitions(items, _articles(), "es", llm=Broken(),
                          settings=DefinitionSettings(pause_seconds=0))
    assert bad.calls == 1 and bad.answers == {} and "definiciones en JSON" in bad.errors[0]["error"]


# ------------------------------------------------------------------ the whole mode

def test_the_law_completes_the_definitions(env, tmp_path):
    llm = LawLLM()
    run = _align(env, llm)
    assert run.seen == LABELS
    ws, m = run.ws, run.manifest
    assert all(s["status"] == DONE for s in ws.progress())
    entries = _by_name(ws)
    assert {n: e["status"] for n, e in entries.items()} == {
        "Consent": LAW_CONCEPT, "DataController": LAW_CONCEPT, "Erasure": ARTICLES,
        "Pizza": NOT_APPLICABLE}
    assert entries["DataController"]["definition"] == EXTRACTED[2][0]["definition"]
    assert entries["DataController"]["law_concept"]["method"] == "same_name"
    assert (entries["Erasure"]["definition"], entries["Erasure"]["article"]) == (
        DEFINED["Erasure"], 4)
    assert entries["Erasure"]["for_decider"] and not entries["Consent"]["for_decider"]
    assert llm.asked == ["Pizza", "Erasure"]                 # the law concepts need no AI
    d = m["law_definitions"]
    assert (d["concepts"], d["with_own"], d["missing"], d["added"]) == (5, 1, 4, 3)
    assert (d[LAW_CONCEPT], d[ARTICLES], d[NOT_APPLICABLE], d[PENDING]) == (2, 1, 1, 0)
    assert d["law"] == "ley.txt" and d["language"] == "es" and d["skipped"] is None
    assert d["law_concepts"]["summary"]["accepted"] == 2
    assert set(m["settings"]["prompts"]) == {"decide", "labels", "define", "extract"}
    assert m["settings"]["prompts"]["define"]["sha256"] == ld.PROMPT.sha256
    assert ws.step("definitions")["detail"] == (
        "3 definiciones de la ley (2 de conceptos de la ley y 1 escrita por la IA con los "
        "artículos) para 4 conceptos sin definición")
    decided = "\n".join(c for c in llm.contexts if "Candidatos DPV" in c)
    assert f"Definicion: {EXTRACTED[2][0]['definition']}" in decided   # the decider sees it
    assert f"Definicion: {DEFINED['Erasure']}" in decided
    assert m["modules"]["definitions"] == "http://example.org/gdpr-definitions"
    assert m["files"]["definitions"] == "output/mi-gdpr-definitions.ttl"
    note = [n for n in m["notes"] if n.startswith("El módulo de definiciones")][0]
    assert note == ("El módulo de definiciones agrega 3 definiciones que le faltaban a la "
                    "ontología, tomadas de ley.txt (2 de conceptos de la ley y 1 escrita por la "
                    "IA con sus artículos); cada una cita su artículo y ninguna definición "
                    "original cambia. El decisor usó las de 2 conceptos que no heredaron su "
                    "alineación.")
    assert d["to_decider"] == 2

    rows = _csv(ws)
    assert rows["DataController"]["law_definition"] == EXTRACTED[2][0]["definition"]
    assert rows["DataController"]["law_definition_source"] == (
        "ley.txt, art. 2 · concepto de la ley DataController (mismo nombre)")
    assert rows["Erasure"]["law_definition_source"] == "ley.txt, art. 4 · escrita por la IA"
    assert rows["Pizza"]["law_definition_source"] == "no aplica: la ley no lo define"
    assert rows["PersonalData"]["law_definition"] == ""
    assert rows["DataController"]["law_evidence"] == "art. 2 (citado)"   # what the AI cited
    assert rows["Erasure"]["law_evidence"].startswith("art. 4 (0,00)")
    assert m["law_evidence"]["cited_by_ai"] == 2

    folder = tmp_path / "z"
    zipfile.ZipFile(run.zip).extractall(folder)
    g = _resolve(folder, m["modules"]["alignment"])
    controller = URIRef(EX + "DataController")
    text = Literal(EXTRACTED[2][0]["definition"], lang="es")
    assert (controller, SKOS.definition, text) in g
    axiom = next(g.subjects(OWL.annotatedTarget, text))
    assert (axiom, RDF.type, OWL.Axiom) in g and (axiom, OWL.annotatedSource, controller) in g
    assert (axiom, DCTERMS.source, Literal("ley.txt, art. 2")) in g
    readme = (folder / "LEEME.txt").read_text(encoding="utf-8")
    assert ("  - mi-gdpr-definitions.ttl: 3 definiciones que le faltaban a tu ontología, "
            "tomadas de ley.txt (2 de conceptos de la ley y 1 escrita por la IA con sus "
            "artículos); cada una cita su artículo. Tus definiciones no cambian.") in readme
    assert ("    En 2 de ellos el artículo es el que citó la IA al sacar su definición de la "
            "ley (citado).\n") in readme
    lines = render_console(ws, m, run.zip).splitlines()
    assert ("  Definiciones de ley.txt: a 4 de 5 conceptos les faltaba; 2 de conceptos de la "
            "ley, 1 escrita por la IA con los artículos, 1 no aplica (módulo "
            f"output/mi-gdpr-definitions.ttl, {m['triples']['definitions']} tripletas)") in lines
    assert ("  IA de las definiciones: extracción 1 llamadas; definiciones 1 llamadas, 0 desde "
            "la caché") in lines

    calls = llm.calls
    again = _align(env, llm)
    assert llm.calls == calls and again.seen == LABELS[:2] + LABELS[-2:]   # nothing asked again
    assert not again.manifest["recomputed"]["definitions"]
    assert any(line.startswith("  Definiciones de ley.txt") and
               line.endswith("(de la corrida anterior)")
               for line in render_console(again.ws, again.manifest).splitlines())


def test_few_missing_definitions_spend_no_ai(env):
    ttl = TTL.replace("ex:Pizza a owl:Class .", "") + "".join(
        f'ex:Term{n} a owl:Class ; rdfs:comment "Definición {n}."@es .\n' for n in range(30))
    llm = LawLLM()
    run = _align(env, llm, ttl=ttl)
    assert run.seen == [s for s in LABELS if s not in (
        "Extraer los conceptos de la ley con la IA", "Aceptar los conceptos por regla",
        "Completar las definiciones con la ley")]
    assert not llm.asked and not any("Extrae" in c for c in llm.contexts)
    assert not (run.ws.work_dir / "law-concepts.json").exists()
    d = run.manifest["law_definitions"]
    assert d["skipped"] == "few_missing" and (d["missing"], d["concepts"]) == (3, 34)
    assert run.ws.step("definitions")["detail"] == (
        "No hizo falta: a 3 de 34 conceptos les falta definición (menos del 10 %)")
    assert "definitions" not in run.manifest["modules"]
    assert "define" not in run.manifest["settings"]["prompts"]
    assert "  Definiciones de ley.txt: no hizo falta (a 3 de 34 conceptos les falta definición)" \
        in render_console(run.ws, run.manifest).splitlines()


def test_turned_off_it_does_nothing(env):
    env.cfg = {**env.cfg, "law_definitions": {"enabled": False}}
    llm = LawLLM()
    run = _align(env, llm)
    assert "Completar las definiciones con la ley" not in run.seen and not llm.asked
    assert run.manifest["law_definitions"]["skipped"] == "disabled"
    assert run.ws.step("definitions")["status"] == DONE
    assert all(r["law_definition"] == "" for r in _csv(run.ws).values())
    assert "  Definiciones de ley.txt: apagadas en config.yaml" in \
        render_console(run.ws, run.manifest).splitlines()


def test_each_source_can_be_turned_off(env):
    env.cfg = {**env.cfg, "law_definitions": {**CFG["law_definitions"],
                                              "from_law_concepts": False}}
    llm = LawLLM(defined={**DEFINED, "DataController": "Quien decide el tratamiento."})
    run = _align(env, llm)
    assert "Extraer los conceptos de la ley con la IA" not in run.seen
    assert {n: e["status"] for n, e in _by_name(run.ws).items()} == {
        "Consent": NOT_APPLICABLE, "DataController": ARTICLES, "Erasure": ARTICLES,
        "Pizza": NOT_APPLICABLE}
    assert run.manifest["law_definitions"]["law_concepts"] is None
    assert "extract" not in run.manifest["settings"]["prompts"]

    env.cfg = {**env.cfg, "law_definitions": {**CFG["law_definitions"], "from_articles": False}}
    llm = LawLLM()
    run = _align(env, llm, ttl=TTL + "ex:Retention a owl:Class .\n")
    assert not llm.asked
    assert {n: e["status"] for n, e in _by_name(run.ws).items()} == {
        "Consent": LAW_CONCEPT, "DataController": LAW_CONCEPT, "Erasure": NO_SOURCE,
        "Pizza": NO_SOURCE, "Retention": NO_SOURCE}
    assert _csv(run.ws)["Pizza"]["law_definition_source"] == ""


def test_without_ai_they_wait_and_the_next_upload_completes_them(env, monkeypatch):
    run = _align(env, LawLLM(), use_llm=False)
    d = run.manifest["law_definitions"]
    assert (d[PENDING], d["added"], d["law_concepts"]["summary"]["accepted"]) == (4, 0, 0)
    assert "definitions" not in run.manifest["modules"]
    assert _csv(run.ws)["Pizza"]["law_definition_source"] == "pendiente"
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    v = svc.view(run.ws.id)
    assert v["definitions_note"] == (
        "La IA no estuvo disponible: 4 conceptos quedaron sin su definición de la ley. Vuelve a "
        "subir los archivos cuando la IA esté disponible para completarlas.")
    assert v["definitions_note"] in svc.notes(v)
    assert v["extraction_note"].startswith("La IA no estuvo disponible: la IA leyó 0 de 5 "
                                           "artículos de la ley")
    llm = LawLLM()
    again = _align(env, llm)
    assert "Completar las definiciones con la ley" in again.seen
    assert again.manifest["recomputed"]["definitions"] and again.manifest["recomputed"]["decider"]
    assert again.manifest["law_definitions"]["added"] == 3
    assert svc.view(again.ws.id)["definitions_note"] is None


def test_the_call_limit_leaves_the_rest_for_the_next_upload(env):
    env.cfg = {**env.cfg, "law_definitions": {**CFG["law_definitions"], "batch_size": 1,
                                              "max_calls": 3, "from_law_concepts": False}}
    llm = LawLLM(defined={**DEFINED, "DataController": "Quien decide el tratamiento."})
    run = _align(env, llm)
    d = run.manifest["law_definitions"]
    assert sorted(llm.asked) == ["DataController", "Erasure", "Pizza"]   # the decider's first
    assert (d[ARTICLES], d[NOT_APPLICABLE], d[PENDING]) == (2, 1, 1)
    assert d["ai"]["call_limit_reached"]
    assert _by_name(run.ws)["Consent"]["status"] == PENDING      # inherited: it can wait
    assert "límite de llamadas" in render_console(run.ws, run.manifest)
    again = _align(env, llm)
    assert llm.asked[3:] == ["Consent"] and again.manifest["law_definitions"][PENDING] == 0


def test_only_new_definitions_redo_the_decider_and_the_evidence(env):
    run = _align(env, LawLLM())
    same = {**env.cfg, "law_definitions": {**CFG["law_definitions"], "max_tokens": 4096}}
    llm, seen = LawLLM(), []
    _, m, _ = align_workspace(run.ws, **{**_kw(env), "cfg": same, "llm": llm,
                                         "progress": seen.append})
    assert "Completar las definiciones con la ley" in seen and llm.calls == 0   # from the cache
    assert m["recomputed"]["definitions"]
    assert not m["recomputed"]["decider"] and not m["recomputed"]["evidence"]
    other = {**env.cfg, "law_definitions": {**CFG["law_definitions"], "from_law_concepts": False}}
    llm = LawLLM(defined={**DEFINED, "DataController": "Quien decide el tratamiento."})
    _, m, _ = align_workspace(run.ws, **{**_kw(env), "cfg": other, "llm": llm})
    assert m["recomputed"]["decider"] and m["recomputed"]["evidence"]
    assert llm.asked == ["DataController", "Consent"]            # Erasure and Pizza: cached
    assert any("Definicion: Quien decide el tratamiento." in c for c in llm.contexts
               if "Candidatos DPV" in c)
    assert _csv(run.ws)["DataController"]["law_evidence"].startswith("art. 2 (0,00)")


def test_another_law_concepts_result_redoes_the_definitions(env):
    run = _align(env, LawLLM())
    from src.core.law_concepts import RESULT_FILE
    path = run.ws.work_dir / RESULT_FILE
    law = json.loads(path.read_text(encoding="utf-8"))
    result = ld.load_result(run.ws)
    assert ld.is_current(result, run.ws, env.cfg)
    law["entries"][0]["definition"] = "Una definición que cambió en la ley extraída."
    path.write_text(json.dumps(law), encoding="utf-8")
    assert not ld.is_current(result, run.ws, env.cfg)
    assert not ld.is_current({**result, "method_version": 0}, run.ws, env.cfg)
    assert not ld.is_current(None, run.ws, env.cfg)


# ------------------------------------------------------------------ the page

def test_the_page_tells_what_the_law_defined(env, monkeypatch):
    run = _align(env, LawLLM())
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    v = svc.view(run.ws.id)
    assert v["definitions"] == (
        "Definiciones de ley.txt: a 4 de 5 conceptos les faltaba; 2 salieron de conceptos de la "
        "ley y 1 la escribió la IA con sus artículos (1 no aplica: la ley no lo define). Van "
        "en un módulo aparte, cada una con su artículo, y el decisor usó 2 para alinear lo que "
        "no heredaste de la base.")                      # Consent inherited: not the decider
    assert v["definitions"] in svc.summary_captions(v) and v["definitions_note"] is None
    assert list(v["rows"][0])[-2:] == ["Definición de la ley", "Artículos de la ley"]
    controller = [r for r in v["rows"] if r["Concepto"] == "DataController"][0]
    assert controller["Definición de la ley"] == EXTRACTED[2][0]["definition"]
    assert v["evidence"].endswith("En 2 conceptos el artículo es el que citó la IA al sacar su "
                                  "definición de la ley (citado).")
    review = {r["Concepto"]: r for r in svc.evidence_rows(v["rows"])}
    assert list(review["Pizza"]) == ["Concepto", "Estado", "Definición de la ley",
                                     "Artículos de la ley"]
    assert review["Erasure"]["Definición de la ley"] == DEFINED["Erasure"]
    names = svc.zip_names(svc.download(run.ws.id, "turtle")["data"])
    assert "mi-gdpr-definitions.ttl" in names


def test_definitions_caption_and_note():
    assert svc.definitions_caption(None) is None and svc.definitions_note(None) is None
    base = {"law": "gdpr-en.pdf", "concepts": 235, "with_own": 231, "missing": 4, "added": 0,
            "law_concept": 0, "articles": 0, "not_applicable": 0, "pending": 0, "ai": {}}
    assert svc.definitions_caption({**base, "skipped": "disabled"}) is None
    assert svc.definitions_caption({**base, "skipped": "few_missing"}) == (
        "Definiciones: tu ontología ya las tenía casi todas (231 de 235), así que no hizo falta "
        "buscarlas en gdpr-en.pdf.")
    many = {**base, "skipped": None, "with_own": 35, "missing": 200, "added": 150,
            "law_concept": 40, "articles": 110, "not_applicable": 50, "to_decider": 1}
    assert svc.definitions_caption(many) == (
        "Definiciones de gdpr-en.pdf: a 200 de 235 conceptos les faltaba; 40 salieron de "
        "conceptos de la ley y 110 las escribió la IA con sus artículos (50 no aplican: la ley "
        "no los define). Van en un módulo aparte, cada una con su artículo, y el decisor usó "
        "una para alinear lo que no heredaste de la base.")
    only_law = {**many, "articles": 0, "added": 40, "to_decider": 0}
    assert svc.definitions_caption(only_law).startswith(
        "Definiciones de gdpr-en.pdf: a 200 de 235 conceptos les faltaba; 40 salieron de "
        "conceptos de la ley (50 no aplican")
    assert svc.definitions_caption(only_law).endswith("cada una con su artículo.")
    nothing = {**many, "law_concept": 0, "articles": 0, "added": 0, "to_decider": 0}
    assert "les faltaba; la ley no dio ninguna (50 no aplican" in svc.definitions_caption(nothing)
    left = {**many, "pending": 7, "ai": {"available": True, "quota_exhausted": True}}
    assert svc.definitions_note(left).startswith("Se agotó la cuota diaria de la IA: 7 conceptos")
    limit = {**left, "ai": {"available": True, "call_limit_reached": True}}
    assert svc.definitions_note(limit).startswith("Se llegó al límite de llamadas")
    assert svc.definitions_note({**left, "ai": {"available": True}}).startswith(
        "La IA falló en algunos lotes: 7 conceptos")


def test_the_upload_and_data_texts_name_the_definitions():
    assert "definiciones" in svc.MODE_HELP[svc.MODE_BOTH]
    assert "definiciones" in svc.MODE_WAIT[svc.MODE_BOTH]
    assert "las definiciones van en módulos aparte" in svc.DATA_NOTE
    assert "Si subes una ley, también le envía su texto" in svc.DATA_NOTE


def test_the_workspace_of_a_law_alone_has_no_definitions_step(env):
    w = open_upload({KIND_ONTOLOGY: ("o.ttl", TTL.encode("utf-8"))}, cfg=env.cfg)
    assert "definitions" not in [s["step"] for s in w.progress()]
    both = open_upload({KIND_ONTOLOGY: ("o.ttl", TTL.encode("utf-8")),
                        KIND_LAW: ("ley.txt", LAW.encode("utf-8"))}, cfg=env.cfg)
    assert [s["label"] for s in both.progress()][3] == "Completar las definiciones con la ley"


def test_off_for_the_decider_they_stay_in_the_module_and_the_evidence(env):
    run = _align(env, LawLLM())
    off = {**env.cfg, "law_definitions": {**CFG["law_definitions"], "to_decider": False}}
    llm, seen = LawLLM(), []
    _, m, _ = align_workspace(run.ws, **{**_kw(env), "cfg": off, "llm": llm,
                                         "progress": seen.append})
    assert not m["recomputed"]["definitions"]                 # the same definitions
    assert m["recomputed"]["decider"] and not m["recomputed"]["evidence"]
    decided = "\n".join(c for c in llm.contexts if "Candidatos DPV" in c)
    assert decided and DEFINED["Erasure"] not in decided
    assert EXTRACTED[2][0]["definition"] not in decided
    d = m["law_definitions"]
    assert d["sent_to_decider"] is False and d["to_decider"] == 0 and d["added"] == 3
    assert m["modules"]["definitions"] and ld.load_result(run.ws)["summary"]["to_decider"] == 2
    note = [n for n in m["notes"] if n.startswith("El módulo de definiciones")][0]
    assert note.endswith("El decisor no las recibe: en las 552 decisiones no mejoraron la "
                         "alineación.")
    assert _csv(run.ws)["DataController"]["law_evidence"] == "art. 2 (citado)"
    from src.alignment.propose import load_result
    assert load_result(run.ws)["law_definitions"] is None
    calls = llm.calls
    align_workspace(run.ws, **{**_kw(env), "cfg": off, "llm": llm})
    assert llm.calls == calls                                  # current again: nothing asked
    monkeypatched = svc.definitions_caption(d)
    assert "el decisor usó" not in monkeypatched