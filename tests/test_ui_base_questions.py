"""Tests for the page "Preguntas de la base" (S9-T10).

A small report and bridge written in a folder of the test check each part of the page, the
commands it shows when a result is missing, and that it looks the same with and without
EDITABLE_BASE; a guarded test opens it with the real results of data/output."""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from app.services import base_questions as svc
from src.alignment import gdpr_bridge as gb
from src.validation import question_catalog as qc
from src.validation import question_explainer as qe
from src.validation import questions as qs

ROOT = Path(__file__).resolve().parents[1]
CATALOG = qc.load_catalog()


def _result(qid, color, **kw):
    q = next(x for x in CATALOG.questions if x.id == qid)
    base = {"id": qid, "group": q.group, "question": q.question, "kind": q.kind,
            "criterion": qc.criterion_text(q), "color": color, "answer": [], "error": None}
    if q.kind == qc.VALUE:
        base.update(values=["3"], answer=["notification001 3 days"])
    elif q.kind == qc.AT_LEAST:
        base.update(count=2, answer=["Consent Consent", "Right_of_access DataSubjectRight"])
    else:
        base.update(found=[{"name": "Libre", "how": "nombre"}], missing=["Inequívoca"])
    return {**base, **kw}


def _report(**kw) -> dict:
    results = [_result("P01", qs.GREEN),
               _result("P02", qs.YELLOW, explanation="Aunque una fila dice 'consent_Unequivocal', "
                                                    "la regla no lo vio.", explained_by=qe.AI),
               _result("L03", qs.YELLOW, explanation="Ninguna fila nombra a Aplicación "
                                                    "favorable.", explained_by=qe.FIXED),
               _result("G01", qs.GREEN)]
    report = {"date": "2026-11-08T00:37:49+00:00",
              "engine": {"name": "jena", "version": "6.2.0", "java": "24.0.2"},
              "seconds": 9.91, "catalog": {"file": "src/validation/questions.yaml", "version": 1,
                                           "sha256": CATALOG.sha256},
              "files": ["data/output/ontopriv-core.rdf", "vocab/dpv.ttl"], "triples": 20142,
              "results": results, "summary": qs.summary(results),
              "explanations": {"prompt": qe.PROMPTS[2].manifest(), "questions": 2,
                               "available": True, "calls": 0, "cached": 2, "invalid": 0,
                               "errors": []},
              "original": {"files": ["data/input/ontopriv.rdf", "vocab/dpv.ttl"],
                           "differences": [], "colors": {"P01": "verde", "P02": "amarillo"}},
              "formats": {"owl": [], "turtle": [], "jsonld": []}}
    return {**report, **kw}


def _bridge(**kw) -> dict:
    rows = [{"ontopriv": "Consent", "gdprtext": "GivenConsent", "dpv_terms": "dpv:Consent",
             "ontopriv_relation": "exactMatch", "gdprtext_relation": "exactMatch",
             "gdpr_articles": "article6-1-a; recital32", "verdict": "coherente",
             "ai_relation": "exactMatch", "explanation": "Ambos son el consentimiento."},
            {"ontopriv": "Processing", "gdprtext": "InfoAboutProcessing",
             "dpv_terms": "dpv:Processing", "ontopriv_relation": "exactMatch",
             "gdprtext_relation": "exactMatch", "gdpr_articles": None,
             "verdict": "incoherente", "ai_relation": "relatedMatch",
             "explanation": "Uno es el tratamiento y el otro un derecho."}]
    validation = {"shapes": {"engine": {"version": "0.40.1"}, "new_violations": 0,
                             "new_warnings": 0, "inherited_violations": 1,
                             "inherited_warnings": 1605},
                  "logic": {"engine": {"name": "jena", "version": "6.2.0", "java": "24.0.2",
                                       "reasoner": "owl-micro"},
                            "new_errors": 0, "new_warnings": 0, "inherited_errors": 5,
                            "inherited_warnings": 1303}}
    out = {"created": "2026-11-08T21:05:00+00:00", "pairs": 2, "terms": 2, "loose_pairs": 9,
           "validation": validation, "prompt": gb.PROMPT.manifest(),
           "summary": {"coherente": 1, "dudosa": 0, "incoherente": 1, "inválida": 0,
                       "pending": 0}, "rows": rows}
    return {**out, **kw}


def _write(folder: Path, report=None, bridge=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if report is not None:
        (folder / qs.REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False),
                                             encoding="utf-8")
    if bridge is not None:
        (folder / gb.RESULT_FILE).write_text(json.dumps(bridge, ensure_ascii=False),
                                             encoding="utf-8")
        (folder / gb.CSV_FILE).write_text("ontopriv,gdprtext\n", encoding="utf-8")
    return folder


@pytest.fixture
def page(monkeypatch):
    """The page reading a folder of the test (or data/output with cfg None)."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    app_dir = str(ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    view_svc = importlib.import_module("services.base_questions")      # what the page imports

    def open_page(folder: Path | None):
        cfg = None if folder is None else {"outputs": {"dir": str(folder)}}
        monkeypatch.setattr(view_svc, "base_cfg", lambda: cfg)
        at = AppTest.from_file(str(ROOT / "app/views/base_questions.py"), default_timeout=60)
        at.run()
        assert not at.exception
        return at
    return open_page


def _shown(at) -> dict:
    """Everything the page shows, to compare two runs."""
    return {"markdown": [m.value for m in at.markdown], "caption": [c.value for c in at.caption],
            "info": [i.value for i in at.info], "warning": [w.value for w in at.warning],
            "code": [c.value for c in at.code], "tabs": [t.label for t in at.tabs],
            "tables": [t.value.to_dict() for t in at.table],
            "frames": [f.value.to_dict() for f in at.dataframe],
            "expanders": [e.label for e in at.expander],
            "downloads": [d.proto.label for d in at.get("download_button")],
            "buttons": [b.label for b in at.button]}


_QUESTION = re.compile(r":\w+-badge\[(Responde|No responde|No aplica|No corrió)\] \*\*")


def _questions(shown: dict) -> list[str]:
    return [m for m in shown["markdown"] if _QUESTION.match(m)]


# ------------------------------------------------------------------ the service

def test_each_question_as_the_page_shows_it():
    rows = svc.question_rows(_report(), CATALOG, qc.BASE)
    assert [r["id"] for r in rows] == ["P01", "P02", "L03"]
    p01, p02, l03 = rows
    assert p01["badge"] == ":green-badge[Responde]" and p01["why"] is None
    assert p01["answer"] == "Valores que trae: 3 · notification001 3 days"
    assert p01["origin"].startswith("Artículo de OntoPriv · Question 1")
    assert p01["expected"].startswith("A más tardar tres días")
    assert (p02["badge"], p02["why_label"]) == (":yellow-badge[No responde]", "Por qué (IA)")
    assert p02["why"] == "Aunque una fila dice 'consent\\_Unequivocal', la regla no lo vio."
    assert l03["why_label"] == "Por qué" and l03["origin"] == "LOPDP · art. 10"
    assert p01["query"] == qs.query_text("P01")
    assert [r["id"] for r in svc.question_rows(_report(), CATALOG, qc.GENERIC)] == ["G01"]
    assert svc.group_caption(_report(), qc.BASE) == "1 en verde, 2 en amarillo y 0 en gris de 3."
    assert svc.origin_text(None) is None


def test_the_header_of_the_run():
    lines = svc.header_lines(_report(), CATALOG)
    assert lines[0] == ("Corrida del 2026-11-08 00:37 UTC · Apache Jena 6.2.0, SPARQL sin "
                        "inferencia (Java 24.0.2) · 9.91 s · 20142 tripletas "
                        "(ontopriv-core.rdf, dpv.ttl)")
    assert lines[1] == f"Catálogo src/validation/questions.yaml v1 (sha256 {CATALOG.sha256[:12]})"
    assert lines[2].startswith("IA que explica las amarillas (prompt explain.v2): ")
    quiet = _report(explanations={**_report()["explanations"], "questions": 0})
    assert len(svc.header_lines(quiet, CATALOG)) == 2                # nothing to explain
    assert not svc.stale(_report(), CATALOG)
    old = _report(catalog={**_report()["catalog"], "sha256": "0" * 64})
    assert svc.stale(old, CATALOG)


def test_the_original_ontopriv_and_the_formats():
    same = svc.original_view(_report())
    assert same["message"].startswith("Las 2 preguntas del artículo dan las mismas filas")
    assert same["rows"][1] == {"Pregunta": "P02", "OntoPriv original": "Amarillo",
                               "Base del sistema": "Amarillo", "Mismas filas": "sí"}
    other = svc.original_view(_report(original={**_report()["original"], "differences": [
        {"id": "P01", "only_in_base": 1, "only_in_original": 0}]}))
    assert other["message"].startswith("1 pregunta da otras filas en el OntoPriv original")
    assert other["message"].endswith(": P01.") and other["rows"][0]["Mismas filas"] == "no"
    assert svc.original_view(_report(original=None)) is None
    formats = svc.formats_view(_report())
    assert formats["message"] == ("Los cuatro formatos (RDF/XML, OWL (RDF/XML), Turtle, JSON-LD) "
                                  "dan las mismas filas en las 4 preguntas.")
    assert formats["rows"][2] == {"Formato": "JSON-LD",
                                  "Preguntas con otras filas que en RDF/XML": "ninguna"}
    broken = svc.formats_view(_report(formats={"owl": [], "turtle": [{"id": "G03"}],
                                               "jsonld": []}))
    assert broken["message"].startswith("Algún formato no da las mismas filas")
    assert broken["rows"][1]["Preguntas con otras filas que en RDF/XML"] == "G03"
    assert svc.formats_view(_report(formats=None)) is None


def test_the_bridge_puts_the_doubtful_first(tmp_path):
    view = svc.bridge_view(_bridge(), _write(tmp_path, bridge=_bridge()))
    assert view["lines"][0].startswith("2 pares OntoPriv ↔ GDPRtEXT que llegan al mismo término")
    assert view["lines"][1].startswith("Forma (pySHACL 0.40.1): 0 violaciones nuevas")
    assert view["lines"][3] == ("Sin hallazgos nuevos: el puente no rompe ninguna regla ni "
                                "agrega contradicciones.")
    assert view["lines"][-1] == ("IA revisora (bridge.v1): 1 coherente, 0 dudosas, "
                                 "1 incoherente; solo explica, no cambia los pares.")
    assert [r["OntoPriv"] for r in view["rows"]] == ["Processing", "Consent"]
    assert list(view["rows"][0])[:4] == ["IA revisora", "OntoPriv", "GDPRtEXT", "Explicación"]
    assert view["flagged"] == [{
        "title": ":red-badge[Incoherente] **Processing** ↔ **InfoAboutProcessing** · por "
                 "dpv:Processing", "explanation": "Uno es el tratamiento y el otro un derecho."}]
    assert view["rows"][1]["Artículos del GDPR"] == "article6-1-a; recital32"
    assert view["rows"][0]["Artículos del GDPR"] == ""
    assert [d["file_name"] for d in view["downloads"]] == [gb.CSV_FILE]  # no module written
    (tmp_path / f"{gb.MODULE}.ttl").write_text("@prefix x: <y#> .", encoding="utf-8")
    view = svc.bridge_view(_bridge(), tmp_path)
    assert [d["file_name"] for d in view["downloads"]] == [gb.CSV_FILE, f"{gb.MODULE}.ttl"]
    waiting = svc.bridge_view(_bridge(summary={**_bridge()["summary"], "pending": 2}), tmp_path)
    assert ", 2 sin revisar; solo explica" in waiting["lines"][-1]


# ------------------------------------------------------------------ the page

def test_the_page_says_which_command_makes_what_is_missing(page, tmp_path):
    at = page(tmp_path / "vacía")
    shown = _shown(at)
    assert shown["tabs"] == ["Preguntas", "OntoPriv original", "Formatos", "Puente con el GDPR"]
    assert shown["info"] == [
        "Todavía no hay respuestas de las preguntas de la base. Se genera con:",
        "Todavía no hay la comparación con el OntoPriv original. Se genera con:",
        "Todavía no hay la comparación de los cuatro formatos. Se genera con:",
        "Todavía no hay puente con el GDPR. Se genera con:"]
    assert shown["code"] == [svc.RUN_QUESTIONS] * 3 + [svc.RUN_BRIDGE]
    assert shown["buttons"] == []


def test_the_page_shows_the_questions_the_original_the_formats_and_the_bridge(page, tmp_path):
    at = page(_write(tmp_path, _report(), _bridge()))
    shown = _shown(at)
    assert _questions(shown) == [
        f":green-badge[Responde] **P01** · {svc.question_rows(_report(), CATALOG, 'base')[0]['question']}",
        ":yellow-badge[No responde] **P02** · ¿Qué condiciones específicas hacen válido el "
        "consentimiento?",
        f":yellow-badge[No responde] **L03** · {next(q.question for q in CATALOG.questions if q.id == 'L03')}",
        f":green-badge[Responde] **G01** · {next(q.question for q in CATALOG.questions if q.id == 'G01')}"]
    assert "**Esperada:** A más tardar tres días desde que el responsable conoce el riesgo " \
           "(LOPDP, art. 46)." in shown["markdown"]
    assert "**Por qué (IA):** Aunque una fila dice 'consent\\_Unequivocal', la regla no lo vio." \
        in shown["markdown"]
    assert "**Por qué:** Ninguna fila nombra a Aplicación favorable." in shown["markdown"]
    assert shown["expanders"].count("Ver la consulta SPARQL") == 4
    assert ("Genéricas (base y subidas) (1): las mismas que se responden en cada subida"
            in shown["expanders"])
    assert shown["code"][0].strip() == qs.query_text("P01").strip()
    assert "1 en verde, 2 en amarillo y 0 en gris de 3." in shown["caption"]
    assert "LOPDP · art. 10" in shown["caption"]                       # where L03 comes from
    assert shown["caption"][0].startswith("Corrida del 2026-11-08 00:37 UTC · Apache Jena")
    assert [list(t["Pregunta"].values()) for t in shown["tables"][:1]] == [["P01", "P02"]]
    assert list(shown["tables"][1]["Formato"].values()) == ["OWL (RDF/XML)", "Turtle", "JSON-LD"]
    assert list(shown["frames"][0]["OntoPriv"].values()) == ["Processing", "Consent"]
    assert (":red-badge[Incoherente] **Processing** ↔ **InfoAboutProcessing** · por "
            "dpv:Processing") in shown["markdown"]
    assert "Uno es el tratamiento y el otro un derecho." in shown["markdown"]
    assert shown["downloads"] == ["Descargar las explicaciones (CSV)"]
    assert shown["info"] == [] and shown["warning"] == [] and shown["buttons"] == []


def test_an_old_run_asks_to_run_again(page, tmp_path):
    old = _report(catalog={**_report()["catalog"], "sha256": "0" * 64})
    shown = _shown(page(_write(tmp_path, old)))
    assert shown["warning"] == ["El catálogo de preguntas cambió después de esta corrida. "
                                "Vuelve a correrla:"]
    assert shown["code"][0] == svc.RUN_QUESTIONS


def test_the_page_is_the_same_with_and_without_editable_base(page, tmp_path, monkeypatch):
    folder = _write(tmp_path, _report(), _bridge())
    monkeypatch.delenv("EDITABLE_BASE", raising=False)
    closed = _shown(page(folder))
    monkeypatch.setenv("EDITABLE_BASE", "1")
    opened = _shown(page(folder))
    assert closed == opened and closed["buttons"] == []
    text = (ROOT / "app/views/base_questions.py").read_text(encoding="utf-8")
    assert "st.button" not in text and "editable" not in text


REAL = [ROOT / "data" / "output" / qs.REPORT_FILE, ROOT / "data" / "output" / gb.RESULT_FILE]


@pytest.mark.skipif(not all(p.exists() for p in REAL),
                    reason="Faltan las preguntas de la base o el puente en data/output")
def test_the_page_with_the_real_results_of_the_base(page):
    at = page(None)
    shown = _shown(at)
    questions = _questions(shown)
    assert [m.split("**")[1] for m in questions[:15]] == [
        f"P0{n}" for n in range(1, 6)] + [f"L{n:02d}" for n in range(1, 11)]
    assert len(questions) == 23 and shown["expanders"].count("Ver la consulta SPARQL") == 23
    assert len(shown["tables"]) == 2 and len(shown["frames"]) == 1
    real = json.loads(REAL[1].read_text(encoding="utf-8"))
    assert len(next(iter(shown["frames"][0].values()))) == real["pairs"]
    flagged = sum(r["verdict"] in ("incoherente", "dudosa") for r in real["rows"])
    assert sum("-badge[Incoherente]" in m or "-badge[Dudosa]" in m
               for m in shown["markdown"]) == flagged
    assert shown["info"] == [] and shown["buttons"] == []
