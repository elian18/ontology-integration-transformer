"""S7-T09: the concepts of an uploaded law, extracted by the AI (prompt extract.v2) and accepted
by the fixed rule, in the workspace of the law.

Fast tests: a fake LLM that answers per article of the batch (like tests/test_extract.py) and a
small law in .txt written to a temporary workspace. The slow test makes one real call."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from src.core import law_concepts as lc
from src.core.acceptance import ACCEPTED, REPEATED
from src.core.law_concepts import (ExtractionSettings, accept_for_workspace,
                                   extract_for_workspace, extraction_settings, is_current,
                                   load_result, prepare_law_concepts, render_console)
from src.workspace.store import DONE, ERROR, open_law

ROOT = Path(__file__).resolve().parents[1]
LAW = ("Art. 1.- Objeto.- La presente ley protege los datos personales del titular.\n"
       "Art. 2.- Titular.- El titular es la persona natural dueña de sus datos.\n"
       "Art. 3.- Consentimiento.- El consentimiento es la manifestación de la voluntad.\n"
       "Art. 4.- Responsable.- El responsable decide sobre el tratamiento de los datos.\n")
BY_ARTICLE = {
    1: [{"name": "PersonalData", "label": "datos personales", "type": "class",
         "definition": "Información de una persona natural identificada.", "article": 1}],
    2: [{"name": "DataSubject", "label": "titular", "type": "class",
         "definition": "Persona natural dueña de sus datos personales.", "article": 2}],
    3: [{"name": "Consent", "label": "consentimiento", "type": "class",
         "definition": "Manifestación de la voluntad del titular.", "article": 3},
        {"name": "Holder", "label": "Titular", "type": "class",
         "definition": "Otra forma de nombrar al titular de los datos.", "article": 3}],
    4: [{"name": "DataController", "label": "responsable", "type": "class",
         "definition": "corta", "article": 4},
        {"name": "hasController", "label": "tiene responsable", "type": "property",
         "definition": "Relaciona un tratamiento con su responsable.", "article": 9}],
}
CFG = {"extraction": {"batch_size": 2, "pause_seconds": 0}, "auto": {"use_llm": True}}


class ExtractLLM:
    """Answers the extraction prompt for the articles present in the batch."""

    def __init__(self, quota_on=(), error_on=(), model="modelo-de-prueba"):
        self.quota_on, self.error_on, self.model = set(quota_on), set(error_on), model
        self.calls, self.batches, self.prompts = 0, [], []

    def ask(self, prompt, context="", max_tokens=1024):
        self.calls += 1
        self.prompts.append((prompt, max_tokens))
        present = [int(n) for n in re.findall(r"=== ARTICULO (\d+) ", context)]
        self.batches.append(present)
        if set(present) & self.quota_on:
            raise Exception("Error code: 429 - RESOURCE_EXHAUSTED quota")
        if set(present) & self.error_on:
            raise Exception("400 bad request")
        return "\n".join(json.dumps(c, ensure_ascii=False)
                         for n in present for c in BY_ARTICLE.get(n, []))


@pytest.fixture
def cfg(tmp_path):
    return {**CFG, "workspaces": {"dir": str(tmp_path / "ws")}}


@pytest.fixture
def ws(cfg):
    return open_law("Ley de prueba.txt", LAW.encode("utf-8"), cfg=cfg)


# ------------------------------------------------------------------ settings and prompt

def test_settings_defaults_config_and_validation():
    assert extraction_settings({}) == ExtractionSettings() == ExtractionSettings(8, 8192, 5.0, 30)
    assert extraction_settings() == ExtractionSettings()                # config.yaml = defaults
    block = {"batch_size": 3, "max_tokens": 1024, "pause_seconds": 1, "max_calls": 2}
    assert extraction_settings({"extraction": block}) == ExtractionSettings(3, 1024, 1.0, 2)
    for bad in ({"batch_size": 0}, {"max_tokens": 255}, {"pause_seconds": -1},
                {"max_calls": 0}):
        with pytest.raises(ValueError, match="bloque extraction"):
            extraction_settings({"extraction": bad})


def test_the_prompt_v2_serves_any_law():
    assert (lc.PROMPT.name, lc.PROMPT.version) == ("extract", 2)
    for words in ("La ley esta en {idioma}: escribe 'label' y 'definition' en {idioma}.",
                  "UpperCamelCase", "lowerCamelCase empezando con un verbo",
                  "una frase corta con verbo", "sin el signo de grado ni literales",
                  "No inventes conceptos"):
        assert words in lc.PROMPT.text
    english = lc.prompt_text("en")
    assert "La ley esta en ingles: escribe 'label' y 'definition' en ingles." in english
    assert "{idioma}" not in english and "{idioma}" not in lc.prompt_text("es")
    assert "en espanol: escribe" in lc.prompt_text("es")


# ------------------------------------------------------------------ the two steps

def test_extract_and_accept(ws, cfg):
    llm, seen = ExtractLLM(), []
    result = prepare_law_concepts(ws, llm=llm, cfg=cfg, sleep_fn=lambda s: None,
                                  progress=seen.append)
    assert seen == ["Extraer los conceptos de la ley con la IA",
                    "Aceptar los conceptos por regla"]
    assert llm.batches == [[1, 2], [3, 4]]
    assert {p for p, _ in llm.prompts} == {lc.prompt_text("es")}         # a law in Spanish
    assert {m for _, m in llm.prompts} == {8192}
    by = {e["name"]: e for e in result["entries"]}
    assert by["PersonalData"]["status"] == ACCEPTED and by["PersonalData"]["kind"] == "class"
    assert by["DataSubject"]["joined"] == ["Holder"] and by["DataSubject"]["articles"] == [2, 3]
    assert by["Holder"]["status"] == REPEATED
    assert by["DataController"]["reasons"] == ["short_definition"]
    assert by["hasController"]["reasons"] == ["no_article"]    # art. 9 is not in the batch
    assert result["summary"] == {"concepts": 6, "accepted": 3, "repeated": 1, "rejected": 2,
                                 "classes": 3, "properties": 0,
                                 "reasons": {"no_article": 1, "short_definition": 1}}
    assert result["law"] == {"file": "Ley de prueba.txt", "language": "es", "articles": 4,
                             "segmenter_version": 2}
    x = result["extraction"]
    assert (x["articles_total"], x["articles_processed"], x["complete"], x["concepts"]) == (
        4, 4, True, 6)
    assert x["pending_articles"] == [] and x["model"] == "modelo-de-prueba"
    assert result["ai"] == {"available": True, "model": "modelo-de-prueba", "calls": 2,
                            "quota_exhausted": False, "call_limit_reached": False,
                            "errors": []}
    assert result["prompt"] == lc.PROMPT.manifest() and result["rule_version"] == 1
    assert load_result(ws) == result
    steps = {s["step"]: s for s in ws.progress()}
    assert steps["extract"]["status"] == DONE and steps["extract"]["complete"] is True
    assert steps["extract"]["detail"] == "6 conceptos de 4 de 4 artículos"
    assert steps["accept"]["detail"] == "3 aceptados, 1 repetidos y 2 rechazados por la regla"
    assert (ws.work_dir / lc.EXTRACTION_FILE).exists()
    again = prepare_law_concepts(ws, llm=llm, cfg=cfg)
    assert again == result and llm.calls == 2                         # current: nothing redone


def test_an_english_law_gets_the_prompt_in_english(cfg):
    law = ("Article 1\nSubject-matter\nThis Regulation protects the personal data of the data "
           "subject and the rights of the data subject.\nArticle 2\nScope\nThis Regulation "
           "applies to the processing of personal data by the controller.\n")
    ws = open_law("law.txt", law.encode("utf-8"), cfg=cfg)
    llm = ExtractLLM()
    r = prepare_law_concepts(ws, llm=llm, cfg=cfg, sleep_fn=lambda s: None)
    assert r["law"]["language"] == "en" and llm.prompts[0][0] == lc.prompt_text("en")


def test_the_pause_and_the_call_limit(ws, cfg):
    pauses = []
    cfg1 = {**cfg, "extraction": {"batch_size": 1, "pause_seconds": 3, "max_calls": 3}}
    llm = ExtractLLM()
    r = prepare_law_concepts(ws, llm=llm, cfg=cfg1, sleep_fn=pauses.append)
    assert (llm.calls, pauses) == (3, [3, 3])
    assert r["ai"]["call_limit_reached"] is True and r["extraction"]["pending_articles"] == [4]
    assert not is_current(r, cfg1)
    r2 = prepare_law_concepts(ws, llm=llm, cfg=cfg1, sleep_fn=pauses.append)
    assert llm.batches[-1] == [4] and r2["extraction"]["complete"]      # continues
    assert r2["summary"]["concepts"] == 6


def test_without_sleep_fn_it_uses_time_sleep(ws, cfg, monkeypatch):
    pauses = []
    monkeypatch.setattr(lc.time, "sleep", pauses.append)
    prepare_law_concepts(ws, llm=ExtractLLM(), cfg={**cfg, "extraction": {"pause_seconds": 2,
                                                                          "batch_size": 2}})
    assert pauses == [2]


def test_quota_stops_and_the_next_run_continues(ws, cfg):
    llm = ExtractLLM(quota_on={3})
    r = prepare_law_concepts(ws, llm=llm, cfg=cfg, sleep_fn=lambda s: None)
    assert r["ai"]["quota_exhausted"] is True and r["extraction"]["pending_articles"] == [3, 4]
    assert "RESOURCE_EXHAUSTED" in r["ai"]["errors"][0]
    assert r["summary"]["concepts"] == 2
    text = render_console(r)
    assert "(cuota agotada: vuelve a correr mañana para seguir)" in text
    assert "    Faltan 2 artículos: vuelve a correr para seguir (lo ya extraído no se repite)." \
        in text
    llm2 = ExtractLLM()
    r2 = prepare_law_concepts(ws, llm=llm2, cfg=cfg, sleep_fn=lambda s: None)
    assert llm2.batches == [[3, 4]] and r2["extraction"]["complete"]


def test_another_prompt_or_model_starts_again(ws, cfg):
    prepare_law_concepts(ws, llm=ExtractLLM(quota_on={3}), cfg=cfg, sleep_fn=lambda s: None)
    other = ExtractLLM(model="otro-modelo")
    prepare_law_concepts(ws, llm=other, cfg=cfg, sleep_fn=lambda s: None)
    assert other.batches == [[1, 2], [3, 4]]
    path = ws.work_dir / lc.EXTRACTION_FILE
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["prompt"]["sha256"] = "otro"
    saved["processed_articles"] = [1, 2]
    path.write_text(json.dumps(saved), encoding="utf-8")
    third = ExtractLLM(model="otro-modelo")
    extract_for_workspace(ws, llm=third, cfg=cfg, sleep_fn=lambda s: None)
    assert third.batches == [[1, 2], [3, 4]]                          # another prompt
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["source"] == "Ley de prueba.txt (segmentador v2)"
    saved["source"], saved["processed_articles"] = "Ley de prueba.txt (segmentador v1)", [1, 2]
    path.write_text(json.dumps(saved), encoding="utf-8")
    fourth = ExtractLLM(model="otro-modelo")
    extract_for_workspace(ws, llm=fourth, cfg=cfg, sleep_fn=lambda s: None)
    assert fourth.batches == [[1, 2], [3, 4]]                         # another segmentation


def test_a_concept_citing_an_article_the_law_lacks_is_rejected(ws, cfg):
    llm = ExtractLLM()
    prepare_law_concepts(ws, llm=llm, cfg=cfg, sleep_fn=lambda s: None)
    path = ws.work_dir / lc.EXTRACTION_FILE
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["concepts"].append({"name": "Ghost", "label": "fantasma", "type": "class",
                              "definition": "Un concepto que cita un artículo ajeno.",
                              "articles": [9]})
    path.write_text(json.dumps(saved), encoding="utf-8")
    r = accept_for_workspace(ws, extract_for_workspace(ws, llm=llm, cfg=cfg))
    assert llm.calls == 2                                             # nothing left to extract
    ghost = next(e for e in r["entries"] if e["name"] == "Ghost")
    assert ghost["status"] == "rejected" and ghost["reasons"] == ["unknown_article"]


def test_a_failed_batch_is_left_for_the_next_run(ws, cfg):
    r = prepare_law_concepts(ws, llm=ExtractLLM(error_on={1}), cfg=cfg, sleep_fn=lambda s: None)
    assert r["extraction"]["pending_articles"] == [1, 2]
    assert r["extraction"]["batches_with_errors"] == 1 and r["ai"]["quota_exhausted"] is False
    assert render_console(r).count("1 lotes con error; error: 400 bad request") == 1


def test_without_ai(ws, cfg):
    off = {**cfg, "auto": {"use_llm": False}}
    r = prepare_law_concepts(ws, cfg=off)
    assert r["ai"] == {"available": False, "error": "IA desactivada (--sin-ia)"}
    assert r["summary"]["concepts"] == 0 and r["extraction"]["complete"] is False
    assert is_current(r, off) and not is_current(r, cfg)
    assert "IA: no usada (IA desactivada (--sin-ia))" in render_console(r)
    partial = prepare_law_concepts(ws, llm=ExtractLLM(quota_on={3}), cfg=cfg,
                                   sleep_fn=lambda s: None)           # extracts 1 and 2
    assert prepare_law_concepts(ws, cfg=off) == partial               # without AI: it stays
    again = extract_for_workspace(ws, cfg=off)                        # the saved extraction
    assert again["extraction"]["articles_processed"] == 2 and len(again["concepts"]) == 2
    path = ws.work_dir / lc.EXTRACTION_FILE
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["prompt"]["sha256"] = "otro"
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert extract_for_workspace(ws, cfg=off)["concepts"] == []      # another prompt: none


def test_errors_mark_the_steps(ws, cfg, monkeypatch):
    import src.core.extract as ex
    monkeypatch.setattr(ex, "propose_profile_concepts", lambda *a, **kw: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        extract_for_workspace(ws, llm=ExtractLLM(), cfg=cfg)
    assert ws.step("extract")["status"] == ERROR and "division" in ws.step("extract")["detail"]
    monkeypatch.setattr(lc, "apply_rule", lambda *a, **kw: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        accept_for_workspace(ws, {"concepts": [], "extraction": {}, "ai": {}})
    assert ws.step("accept")["status"] == ERROR


def test_is_current(ws, cfg):
    assert not is_current(None, cfg)
    r = prepare_law_concepts(ws, llm=ExtractLLM(), cfg=cfg, sleep_fn=lambda s: None)
    assert is_current(r, cfg) and is_current(r, cfg, use_llm=False)
    for change in ({"method_version": 0}, {"rule_version": 0},
                   {"prompt": {**r["prompt"], "sha256": "x"}},
                   {"law": {**r["law"], "segmenter_version": 1}}):
        assert not is_current({**r, **change}, cfg)
    pending = {**r, "extraction": {**r["extraction"], "complete": False}}
    assert not is_current(pending, cfg) and is_current(pending, cfg, use_llm=False)


# ------------------------------------------------------------------ console and CLI

def test_console(ws, cfg):
    r = prepare_law_concepts(ws, llm=ExtractLLM(), cfg=cfg, sleep_fn=lambda s: None)
    text = render_console(r)
    assert text.splitlines()[:3] == [
        "Conceptos de la ley (S7-T09)",
        "  Ley: Ley de prueba.txt · español · 4 artículos",
        "  Extracción con la IA: 4 de 4 artículos · IA: 2 llamadas en esta corrida, 0 lotes "
        "con error"]
    assert "Faltan" not in text and "  Conceptos extraídos: 6" in text
    assert ("    aceptados: 3 (3 clases, 0 propiedades) · repetidos: 1 · rechazados: 2"
            in text)
    assert ("    rechazos: no cita un artículo de la ley (1); definición de menos de 20 "
            "caracteres (1)") in text
    assert '    PersonalData "datos personales" [class] <- art. 1' in text
    assert '    DataSubject "titular" [class] <- art. 2, 3' in text
    assert '    repetido: Holder se une a DataSubject ("Titular")' in text
    assert "    rechazado: DataController (definición de menos de 20 caracteres)" in text
    assert len(render_console(r, examples=1).splitlines()) == len(text.splitlines()) - 2
    limited = {**r, "ai": {**r["ai"], "call_limit_reached": True}}
    assert "(límite de llamadas: vuelve a correr para seguir)" in render_console(limited)


def test_cli(tmp_path, cfg, monkeypatch, capsys):
    monkeypatch.setattr("src.config.load_config", lambda: cfg)
    law = tmp_path / "ley.txt"
    law.write_text(LAW, encoding="utf-8")
    assert lc._run([str(law), "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Conceptos de la ley (S7-T09)" in out and lc.RESULT_FILE in out
    assert "IA: no usada (IA desactivada (--sin-ia))" in out
    assert lc._run([str(tmp_path / "no.txt")]) == 1
    empty = tmp_path / "vacia.txt"
    empty.write_text("Sin artículos numerados.\n", encoding="utf-8")
    assert lc._run([str(empty), "--sin-ia"]) == 1
    assert "No se encontraron artículos numerados" in capsys.readouterr().out


# ------------------------------------------------------------------ real LLM

@pytest.mark.slow
def test_real_llm_extracts_two_gdpr_articles():
    """One real call (spends quota) on articles 6 and 7 of the GDPR, in English."""
    if not os.environ.get("LLM_API_KEY"):
        from src import config  # noqa: F401  (loads .env)
    if not os.environ.get("LLM_API_KEY"):
        pytest.skip("Falta LLM_API_KEY en .env")
    gdpr = ROOT / "data" / "input" / "laws" / "gdpr-en.pdf"
    if not gdpr.exists():
        pytest.skip("Falta data/input/laws/gdpr-en.pdf")
    from src.core.acceptance import apply_rule, summarize
    from src.core.extract import propose_profile_concepts
    from src.ingest.law_text import clean_law_file
    from src.ingest.legal_segmenter import segment_articles
    text, _ = clean_law_file(gdpr)
    articles = [a for a in segment_articles(text)["articles"] if a["number"] in (6, 7)]
    report = propose_profile_concepts(articles, source="gdpr", prompt=lc.PROMPT,
                                      prompt_text=lc.prompt_text("en"), max_tokens=8192,
                                      batch_size=8)
    entries = apply_rule([c.as_dict() for c in report.concepts], {6, 7})
    print("\n" + "\n".join(f"  {e['status']}: {e['name']} \"{e['label']}\" <- {e['articles']}"
                           for e in entries))
    assert report.calls == 1 and not report.errors and report.processed_articles == [6, 7]
    s = summarize(entries)
    assert s["accepted"] >= 2 and s["accepted"] >= s["rejected"]
    assert any("consent" in e["label"].lower() for e in entries if e["status"] == ACCEPTED)
    from src.ingest.legal_segmenter import detect_language
    english = [detect_language(e["definition"]) == "en" for e in entries]
    assert sum(english) >= 0.8 * len(english)                     # definitions in English