from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import align_ontology as svc
from src.validation import ai_auditor as au
from src.validation import question_explainer as qe
from src.validation import questions as qs
from src.validation import upload_validation as uv
from src.validation.shape_inspector import GREEN, RED, YELLOW
from src.workspace.store import load_workspace
from tests.test_alignment_propose import TTL, DeciderLLM, Embed, env, targets  # noqa: F401
from tests.test_ui_align_ontology import _boxes, cfg, page  # noqa: F401
from tests.test_validation_upload import FlaggingLLM, _real_dpv


def _align(env, targets, llm=None):  # noqa: F811
    return svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), base=env.base, targets=targets,
                     embed_fn=Embed(), llm=llm or DeciderLLM(), chroma_path=env.chroma,
                     sleep_fn=lambda s: None)


@pytest.fixture
def yellow(env, targets, monkeypatch):  # noqa: F811
    """An upload whose validation is yellow: 3 concepts without definition, PersonalData
    marked by the AI reviewer."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    return _align(_real_dpv(env), targets, FlaggingLLM())


@pytest.fixture
def red(env, targets, monkeypatch):  # noqa: F811
    """The small DPV of the tests has no rdfs:isDefinedBy: its terms "are not in the DPV"."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    return _align(env, targets)


# ------------------------------------------------------------------ the texts

def _audit(**kw):
    summary = {"accepted": 4, au.COHERENT: 2, au.DOUBTFUL: 1, au.INCOHERENT: 1, au.INVALID: 0,
               "pending": 0, **kw.pop("summary", {})}
    return {"status": YELLOW, "summary": summary, "ai": {"available": True},
            "prompt": au.PROMPT.manifest(), "flagged": [{}, {}], **kw}


def test_the_badge_of_each_status():
    assert [svc.badge(s) for s in (GREEN, YELLOW, RED, uv.NOT_RUN, uv.FAILED)] == [
        ":green-badge[Verde]", ":yellow-badge[Amarillo]", ":red-badge[Rojo]",
        ":gray-badge[Sin revisar]", ":gray-badge[No corrió]"]


def test_the_text_of_each_card():
    shapes = {"status": YELLOW, "engine": "pySHACL 0.40.1", "new": 0, "inherited": 232,
              "warnings": 4}
    assert svc.card_text(uv.SHAPES, shapes) == ("Reglas rotas: 0 nuevas · 232 heredadas  \n"
                                                "Avisos: 4")
    assert svc.card_text(uv.LOGIC, {**shapes, "inherited": 255}).startswith(
        "Contradicciones: 0 nuevas · 255 heredadas")
    assert svc.card_text(uv.SHAPES, {**shapes, "new": 1, "inherited": 1}).startswith(
        "Reglas rotas: 1 nueva · 1 heredada  \n")
    assert svc.card_caption(uv.SHAPES, shapes) == "pySHACL 0.40.1"
    assert svc.card_text(uv.LOGIC, {"status": uv.FAILED, "error": "Java no responde"}) == (
        "No pudo correr  \nJava no responde")
    assert svc.card_text(uv.AUDIT, _audit()) == ("Incoherentes: 1 · dudosas: 1  \n"
                                                 "De 4 aceptadas por la regla")
    assert svc.card_text(uv.AUDIT, _audit(summary={"pending": 2})).endswith(" · sin revisar: 2")
    assert svc.card_text(uv.AUDIT, {"status": uv.NOT_RUN}) == "No se hizo la revisión"
    assert svc.card_text(uv.AUDIT, _audit(summary={"accepted": 0})).startswith(
        "Nada que revisar")
    off = _audit(status=uv.NOT_RUN, ai={"available": False, "error": "IA desactivada (--sin-ia)"})
    assert svc.card_text(uv.AUDIT, off) == "No revisó las 4 aceptadas  \nIA desactivada (--sin-ia)"
    failed = _audit(status=uv.NOT_RUN, ai={"available": True, "errors": ["429 cuota"]})
    assert svc.card_text(uv.AUDIT, failed).endswith("\n429 cuota")
    assert svc.card_caption(uv.AUDIT, _audit()) == "prompt audit.v1 · solo explica"
    assert svc.card_caption(uv.AUDIT, {"status": uv.NOT_RUN}) == "solo explica"


def test_the_headline_says_what_to_look_at():
    shapes = {"status": YELLOW, "new": 0, "inherited": 232, "warnings": 4}
    logic = {"status": GREEN, "new": 0, "inherited": 0, "warnings": 0}
    result = {"status": YELLOW, "inspectors": {uv.SHAPES: shapes, uv.LOGIC: logic,
                                              uv.AUDIT: _audit()}}
    assert svc.validation_headline(result) == (
        "Nada de lo que escribió el sistema rompe una regla. Para mirar: 232 hallazgos "
        "heredados (ya estaban en tu ontología o en la base del sistema); 4 avisos "
        "(recomendaciones, como conceptos sin definición); 2 correspondencias que la IA "
        "revisora marcó para que las mires.")
    one = {**result, "inspectors": {uv.SHAPES: {**shapes, "inherited": 1, "warnings": 0},
                                    uv.LOGIC: {"status": uv.FAILED, "error": "x"},
                                    uv.AUDIT: {"status": uv.NOT_RUN}}}
    assert svc.validation_headline(one).endswith(
        "Para mirar: 1 hallazgo heredado (ya estaban en tu ontología o en la base del sistema); "
        "no pudo correr: Lógica (razonador).")
    green = {"status": GREEN, "inspectors": {uv.SHAPES: logic, uv.LOGIC: logic,
                                             uv.AUDIT: _audit(status=GREEN)}}
    assert svc.validation_headline(green) == ("Ninguna regla se rompe y el razonador no "
                                              "encontró contradicciones. La IA revisora no "
                                              "marcó nada.")
    unreviewed = {**green, "inspectors": {**green["inspectors"],
                                          uv.AUDIT: _audit(status=uv.NOT_RUN)}}
    assert svc.validation_headline(unreviewed).endswith(
        "La IA no revisó las correspondencias aceptadas.")
    nothing = {**green, "inspectors": {**green["inspectors"],
                                       uv.AUDIT: _audit(status=GREEN, summary={"accepted": 0})}}
    assert svc.validation_headline(nothing).endswith("no encontró contradicciones.")
    assert svc.validation_headline({**result, "status": RED}).startswith(
        "El sistema escribió algo que rompe una regla")


# ------------------------------------------------------------------ the panel of an upload

def test_the_panel_of_an_upload(yellow):
    v = svc.view(yellow)
    val = v["validation"]
    assert (val["status"], val["badge"]) == (YELLOW, ":yellow-badge[Amarillo]")
    assert [(c["criterion"], c["title"], c["badge"]) for c in val["cards"]] == [
        ("Interoperabilidad", "Forma (SHACL)", ":yellow-badge[Amarillo]"),
        ("Coherencia", "Lógica (razonador)", ":green-badge[Verde]"),
        ("Completitud", "Preguntas (SPARQL)", ":yellow-badge[Amarillo]"),
        ("Coherencia", "IA revisora", ":yellow-badge[Amarillo]")]
    assert val["cards"][2]["text"] == ("Responden: 2 · sin responder: 5  \n"
                                       "De 7 preguntas que aplican · no aplica: 1")
    assert val["cards"][2]["caption"].startswith("rdflib ")
    assert val["cards"][2]["caption"].endswith(" · prompt explain.v1")
    q = val["questions"]
    assert q["caption"].startswith("8 preguntas, las mismas para cualquier ontología de privacidad")
    assert q["caption"].endswith(": 2 en verde, 5 en amarillo y 0 en gris de las 7 que aplican; "
                                 "1 no aplica.")
    rows = {r["id"]: r for r in q["rows"]}
    assert list(rows) == [f"G0{n}" for n in range(1, 9)]
    assert rows["G01"]["badge"] == ":green-badge[Responde]" and rows["G01"]["why"] is None
    assert (rows["G02"]["badge"], rows["G02"]["why_label"], rows["G02"]["why"]) == (
        ":yellow-badge[No responde]", "Por qué (IA)", "Explicación de prueba para G02.")
    assert rows["G08"]["badge"] == ":gray-badge[No aplica]"
    assert rows["G08"]["answer"].startswith("No aplica a «Mi ontología»")
    assert rows["G03"]["query"] == qs.query_text("G03") and "dpv:LegalBasis" in rows["G03"]["query"]
    assert rows["G01"]["criterion"] == ("Verde si la respuesta trae al menos 1 concepto; "
                                        "amarillo si no trae ninguno.")
    assert val["rules"] == [{"Regla": "Cada concepto tiene definición", "Tipo": "aviso",
                             "Nuevas": 0, "Heredadas": 3}]
    assert val["kinds"] == []
    assert val["flagged"] == [{"Concepto": "PersonalData", "Término del DPV": "PersonalData",
                               "Relación": "exactMatch", "Veredicto": au.INCOHERENT,
                               "Explicación": "Trata de otra cosa."}]
    assert list(val["flagged"][0]) == svc.FLAGGED_COLUMNS
    review = {r["Concepto"]: r["Revisión de la IA"] for r in v["rows"]}
    assert review["PersonalData"] == au.INCOHERENT and review["DataController"] == ""
    log = svc.validation_log(yellow, svc.workspace_cfg()).decode("utf-8")
    assert log.startswith(f"Validación del espacio {yellow}: amarillo")


def test_an_upload_validated_in_the_sprint_8_shows_yellow(yellow):
    ws = load_workspace(yellow, cfg=svc.workspace_cfg())
    path = ws.output_dir / uv.RESULT_FILE
    _set_questions(yellow, None)                               # the Sprint 8 had no questions
    path.write_text(path.read_text(encoding="utf-8").replace(f'"{YELLOW}"', '"ámbar"'),
                    encoding="utf-8")                          # as the Sprint 8 saved it
    assert '"ámbar"' in path.read_text(encoding="utf-8")
    val = svc.view(yellow)["validation"]
    assert (val["status"], val["badge"]) == (YELLOW, ":yellow-badge[Amarillo]")
    assert [c["badge"] for c in val["cards"]] == [":yellow-badge[Amarillo]",
                                                   ":green-badge[Verde]",
                                                   ":gray-badge[Sin revisar]",
                                                   ":yellow-badge[Amarillo]"]
    assert svc.badge("ámbar") == ":yellow-badge[Amarillo]"
    assert svc.question_badge({"color": "ámbar"}) == ":yellow-badge[No responde]"


def test_the_colors_are_a_traffic_light():
    help_text = svc.VALIDATION_HELP["Colores"]
    assert [w for w in ("rojo:", "amarillo:", "verde:", "gris:") if w in help_text] == [
        "rojo:", "amarillo:", "verde:", "gris:"]
    assert "ámbar" not in help_text and "ámbar" not in str(svc.LIGHTS)
    assert "preguntas que tu ontología no responde" in help_text
    assert "la pregunta no aplica" in help_text


# ------------------------------------------------------------------ the questions (S9-T07)

def _summary(green=7, yellow=0, gray=0, na=1):
    return {GREEN: green, YELLOW: yellow, qs.GRAY: gray, qs.NOT_APPLICABLE: na}


def _ai(questions=0, version=2):
    return {"prompt": qe.PROMPTS[version].manifest(), "questions": questions}


def test_each_card_names_its_criterion():
    assert svc.CRITERIA == {uv.SHAPES: "Interoperabilidad", uv.LOGIC: "Coherencia",
                            uv.QUESTIONS: "Completitud", uv.AUDIT: "Coherencia"}
    assert list(svc.CRITERIA) == list(svc.CARD_TITLES)
    criteria = svc.VALIDATION_HELP["Criterios"]
    assert [w for w in ("coherencia:", "completitud:", "interoperabilidad:") if w in criteria] \
        == ["coherencia:", "completitud:", "interoperabilidad:"]
    assert "(Preguntas)" in criteria and "Preguntas (SPARQL)" in svc.VALIDATION_HELP


def test_the_card_of_the_questions():
    block = {"status": GREEN, "engine": "Apache Jena 6.2.0", "summary": _summary(), "ai": _ai()}
    assert svc.card_text(uv.QUESTIONS, block) == ("Responden: 7 · sin responder: 0  \n"
                                                  "De 7 preguntas que aplican · no aplica: 1")
    assert svc.card_caption(uv.QUESTIONS, block) == "Apache Jena 6.2.0"    # nothing explained
    yellow = {**block, "status": YELLOW, "summary": _summary(5, 1, 2, 0), "ai": _ai(1)}
    assert svc.card_text(uv.QUESTIONS, yellow) == (
        "Responden: 5 · sin responder: 1 · no corrieron: 2  \nDe 8 preguntas que aplican")
    assert svc.card_caption(uv.QUESTIONS, yellow) == "Apache Jena 6.2.0 · prompt explain.v2"
    few = {**block, "summary": _summary(1, 0, 0, 2)}
    assert svc.card_text(uv.QUESTIONS, few).endswith("De 1 pregunta que aplica · no aplican: 2")
    failed = {"status": uv.FAILED, "error": "Java no responde"}
    assert svc.card_text(uv.QUESTIONS, failed) == "No pudo correr  \nJava no responde"
    assert svc.card_caption(uv.QUESTIONS, failed) == ""
    assert svc.card_text(uv.QUESTIONS, {"status": uv.NOT_RUN}) == svc.NO_QUESTIONS


def test_the_badge_of_each_question():
    assert [svc.question_badge(r) for r in (
        {"color": GREEN}, {"color": YELLOW}, {"color": qs.GRAY, "applies": False},
        {"color": qs.GRAY, "applies": True}, {"color": qs.GRAY})] == [
        ":green-badge[Responde]", ":yellow-badge[No responde]", ":gray-badge[No aplica]",
        ":gray-badge[No corrió]", ":gray-badge[No corrió]"]


def test_what_the_page_writes_is_shown_as_it_is():
    assert svc.plain("Data_Owner *x* $5 [a](b) <i> #1 `c` a|b ~s \\") == (
        "Data\\_Owner \\*x\\* \\$5 \\[a\\](b) \\<i\\> \\#1 \\`c\\` a\\|b \\~s \\\\")
    assert svc.plain("¿Qué derechos tiene el titular?") == "¿Qué derechos tiene el titular?"


def _row(qid, color, **kw):
    return {"id": qid, "question": f"¿Pregunta {qid}?", "criterion": "Verde si trae algo.",
            "color": color, "applies": True, "answer": f"Respuesta de {qid}", **kw}


def test_the_tab_of_the_questions():
    block = {"status": YELLOW, "engine": "rdflib 7.6.0", "seconds": 0.5,
             "summary": _summary(1, 2, 0, 0), "ai": _ai(2),
             "results": [_row("G01", GREEN, answer="1 concepto (p. ej. Data_Owner DataSubject)"),
                         _row("G02", YELLOW, explanation="La IA dice _por qué_.",
                              explained_by=qe.AI),
                         _row("G03", YELLOW, explanation="Frase fija.", explained_by=qe.FIXED)]}
    view = svc.questions_view(block)
    assert view["caption"] == ("3 preguntas, las mismas para cualquier ontología de privacidad, "
                               "respondidas con SPARQL sobre tu ontología, sus módulos y el DPV "
                               "(rdflib 7.6.0, 0.5 s): 1 en verde, 2 en amarillo y 0 en gris "
                               "de 3.")
    assert [(r["id"], r["why_label"], r["why"]) for r in view["rows"]] == [
        ("G01", "Por qué", None), ("G02", "Por qué (IA)", "La IA dice \\_por qué\\_."),
        ("G03", "Por qué", "Frase fija.")]
    assert view["rows"][0]["query"] == qs.query_text("G01")
    assert view["rows"][0]["answer"] == "1 concepto (p. ej. Data\\_Owner DataSubject)"
    assert view["rows"][0]["question"] == "¿Pregunta G01?"
    assert svc.questions_view(None) is None
    assert svc.questions_view({"status": uv.NOT_RUN}) is None


def test_a_query_that_left_the_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(qs, "QUERIES_DIR", tmp_path)
    assert svc._query("G01") is None
    (tmp_path / "G01.rq").write_text("SELECT * WHERE { ?s ?p ?o }", encoding="utf-8")
    assert svc._query("G01") == "SELECT * WHERE { ?s ?p ?o }"


def test_the_headline_of_a_green_panel_counts_the_answered_questions():
    clean = {"status": GREEN, "new": 0, "inherited": 0, "warnings": 0}
    audit = _audit(status=GREEN, summary={"accepted": 0})
    result = {"status": GREEN, "inspectors": {
        uv.SHAPES: clean, uv.LOGIC: clean, uv.AUDIT: audit,
        uv.QUESTIONS: {"status": GREEN, "summary": _summary()}}}
    assert svc.validation_headline(result) == (
        "Ninguna regla se rompe y el razonador no encontró contradicciones. Tu ontología "
        "responde las 7 preguntas de competencia que aplican.")
    one = {**result, "inspectors": {**result["inspectors"],
                                    uv.QUESTIONS: {"status": GREEN,
                                                   "summary": _summary(1, 0, 0, 7)}}}
    assert svc.validation_headline(one).endswith(
        "Tu ontología responde la pregunta de competencia que aplica.")


def _set_questions(ws_id: str, block: dict | None) -> None:
    """Rewrite the questions of an upload as another run would have left them."""
    path = load_workspace(ws_id, cfg=svc.workspace_cfg()).output_dir / uv.RESULT_FILE
    result = json.loads(path.read_text(encoding="utf-8"))
    if block is None:
        result["inspectors"].pop(uv.QUESTIONS)
    else:
        result["inspectors"][uv.QUESTIONS] = block
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def _green(block: dict) -> dict:
    """The same questions with every one that applies answered, as GDPRtEXT has them."""
    results = [{k: v for k, v in r.items() if k not in ("explanation", "explained_by")}
               | ({"color": GREEN, "answer": "3 conceptos"} if r["applies"] else {})
               for r in block["results"]]
    return {**block, "status": GREEN, "summary": qs.summary(results),
            "ai": {**block["ai"], "questions": 0}, "results": results}


def test_an_upload_from_before_the_questions(page, yellow):
    _set_questions(yellow, None)
    val = svc.view(yellow)["validation"]
    card = val["cards"][2]
    assert (card["title"], card["badge"], card["text"]) == (
        "Preguntas (SPARQL)", ":gray-badge[Sin revisar]", svc.NO_QUESTIONS)
    assert val["questions"] is None
    at = page()
    at.session_state["align_ws"] = yellow
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Resumen", "Conceptos", "Para revisar"]
    assert len([d for d in at.download_button if d.key != "align_dl_log"]) == 4


def test_a_red_panel_still_lets_the_person_download(red):
    val = svc.view(red)["validation"]
    assert val["status"] == RED and val["cards"][0]["badge"] == ":red-badge[Rojo]"
    assert val["rules"][0] == {"Regla": "El término del DPV existe", "Tipo": "regla",
                               "Nuevas": 3, "Heredadas": 0}
    assert len(svc.download_options(red)) == 4


def test_the_incoherent_ones_come_first():
    def f(name, verdict):
        return {"name": name, "dpv_name": "X", "relation": None, "verdict": verdict,
                "explanation": None}
    result = {"inspectors": {uv.AUDIT: {"flagged": [f("b", au.DOUBTFUL), f("C", au.INCOHERENT),
                                                    f("a", au.INCOHERENT)]}}}
    rows = svc.flagged_rows(result)
    assert [(r["Concepto"], r["Veredicto"]) for r in rows] == [
        ("a", au.INCOHERENT), ("C", au.INCOHERENT), ("b", au.DOUBTFUL)]
    assert rows[0]["Relación"] == "" and rows[0]["Explicación"] == ""
    assert svc.flagged_rows(None) == []


def test_an_upload_from_before_the_validation(yellow):
    ws = load_workspace(yellow, cfg=svc.workspace_cfg())
    (ws.output_dir / uv.RESULT_FILE).unlink()
    assert svc.view(yellow)["validation"] is None


def test_the_detail_counts_what_comes_from_the_base(monkeypatch):
    from src.validation import logic_inspector as li, shape_inspector as si
    rule = {"name": "Ningún texto rdf:langString sin idioma", "severity": "violación",
            "nuevo": 0, "ontología": 2, "base": 1}
    kind = {"label": "Un individuo está en dos clases disjuntas", "errors": 3, "warnings": 0,
            "nuevo": 1, "ontología": 0, "base": 2}
    monkeypatch.setattr(si, "load_result", lambda ws: {"summary": {"rules": {"R": rule}}})
    monkeypatch.setattr(li, "load_result", lambda ws: {"summary": {"kinds": {"K": kind}}})
    ok = {"status": YELLOW}
    assert svc._rule_rows(None, ok) == [{"Regla": rule["name"], "Tipo": "regla", "Nuevas": 0,
                                         "Heredadas": 3}]
    assert svc._kind_rows(None, ok) == [{"Tipo": kind["label"], "Contradicciones": 3,
                                         "Avisos": 0, "Nuevas": 1, "Heredadas": 2}]


def test_a_failed_inspector_has_no_stale_detail(yellow):
    ws = load_workspace(yellow, cfg=svc.workspace_cfg())
    block = {"status": uv.FAILED, "error": "x"}
    assert svc._rule_rows(ws, block) == [] and svc._kind_rows(ws, block) == []


# ------------------------------------------------------------------ the page (AppTest)

def test_the_page_shows_the_panel_and_what_the_ai_marked(page, yellow):
    at = page()
    at.session_state["align_ws"] = yellow
    at.run()
    assert not at.exception
    text = [m.value for m in at.markdown]
    assert "**Validación**" in text
    assert any(t.startswith(":yellow-badge[Amarillo] Nada de lo que escribió el sistema")
               for t in text)
    assert "**Forma (SHACL)** :yellow-badge[Amarillo]" in text
    assert "**Lógica (razonador)** :green-badge[Verde]" in text
    assert "Contradicciones: 0 nuevas · 0 heredadas  \nAvisos: 0" in text
    assert {"¿Qué revisa cada uno?", "Ver el detalle"} <= set(_boxes(at))
    assert any(t.startswith("**Colores.** rojo:") for t in text)
    log = [d for d in at.download_button if d.key == "align_dl_log"]
    assert len(log) == 1 and log[0].label == "Descargar el detalle (validation-log.txt)"
    assert log[0].proto.deferred_file_id and log[0].proto.ignore_rerun   # built when clicked
    assert "**Marcadas por la IA revisora (1)**" in text                 # "Para revisar"
    frames = [list(d.value.columns) for d in at.dataframe]
    assert svc.FLAGGED_COLUMNS in frames and ["Regla", "Tipo", "Nuevas", "Heredadas"] in frames
    assert [t.label for t in at.tabs] == ["Resumen", "Conceptos", "Para revisar",
                                          svc.QUESTIONS_TAB]
    assert not at.warning and not at.error                  # the panel uses no alerts


def test_the_page_of_a_red_upload(page, red):
    at = page()
    at.session_state["align_ws"] = red
    at.run()
    assert not at.exception
    text = [m.value for m in at.markdown]
    assert any(t.startswith(":red-badge[Rojo] El sistema escribió algo que rompe una regla")
               for t in text)
    assert len([d for d in at.download_button if d.key != "align_dl_log"]) == 4


def test_the_page_of_an_upload_from_before_the_validation(page, yellow):
    ws = load_workspace(yellow, cfg=svc.workspace_cfg())
    (ws.output_dir / uv.RESULT_FILE).unlink()
    at = page()
    at.session_state["align_ws"] = yellow
    at.run()
    assert not at.exception
    assert svc.NO_VALIDATION in [c.value for c in at.caption]
    assert not [d for d in at.download_button if d.key == "align_dl_log"]


def _questions_page(at) -> dict:
    """What the page shows of the questions: card, criteria, rows of the tab, queries."""
    text = [m.value for m in at.markdown]
    return {"card": next(t for t in text if t.startswith("**Preguntas (SPARQL)**")),
            "criteria": [c.value for c in at.caption if c.value.startswith("Criterio: ")],
            "rows": [t for t in text if t.startswith(":") and "-badge[" in t and "** · " in t],
            "why": [t for t in text if t.startswith("**Por qué")],
            "queries": [e.label for e in at.expander].count("Ver la consulta SPARQL"),
            "code": [c.value for c in at.code],
            "tabs": [t.label for t in at.tabs],
            "zips": len([d for d in at.download_button if d.key != "align_dl_log"])}


def test_the_page_shows_the_questions_in_green_yellow_and_gray(page, yellow):
    path = load_workspace(yellow, cfg=svc.workspace_cfg()).output_dir / uv.RESULT_FILE
    block = json.loads(path.read_text(encoding="utf-8"))["inspectors"][uv.QUESTIONS]
    seen = {}
    for name, state in (("yellow", block), ("green", _green(block)),
                        ("gray", {"status": uv.FAILED, "error": "Java no responde"})):
        _set_questions(yellow, state)
        at = page()
        at.session_state["align_ws"] = yellow
        at.run()
        assert not at.exception
        seen[name] = _questions_page(at)
    for shown in seen.values():                       # the criteria and the download, always
        assert shown["criteria"] == ["Criterio: Interoperabilidad", "Criterio: Coherencia",
                                     "Criterio: Completitud", "Criterio: Coherencia"]
        assert shown["zips"] == 4 and svc.QUESTIONS_TAB in shown["tabs"]
    yellow_page, green_page, gray_page = seen["yellow"], seen["green"], seen["gray"]
    assert yellow_page["card"] == "**Preguntas (SPARQL)** :yellow-badge[Amarillo]"
    assert yellow_page["rows"][:2] == [
        ":green-badge[Responde] **G01** · ¿Qué conceptos de la ontología están alineados con el "
        "DPV?", ":yellow-badge[No responde] **G02** · ¿Qué derechos del titular representa la "
        "ontología?"]
    assert yellow_page["rows"][-1].startswith(":gray-badge[No aplica] **G08** · ")
    assert yellow_page["why"][0] == "**Por qué (IA):** Explicación de prueba para G02."
    assert len(yellow_page["why"]) == 5
    assert yellow_page["queries"] == 8
    assert yellow_page["code"][0].strip() == qs.query_text("G01").strip()
    assert green_page["card"] == "**Preguntas (SPARQL)** :green-badge[Verde]"
    assert len(green_page["rows"]) == 8 and green_page["why"] == []
    assert sum(r.startswith(":green-badge[Responde]") for r in green_page["rows"]) == 7
    assert gray_page["card"] == "**Preguntas (SPARQL)** :gray-badge[No corrió]"
    assert gray_page["rows"] == [] and gray_page["queries"] == 0


ROOT = Path(__file__).resolve().parents[1]
REAL = {"gdprtext-ontology-b863b1da": "Mi ontología y una ley",
        "ley-1581-2012-colombia-e98f0354": "Una ley"}


@pytest.mark.skipif(not all((ROOT / "data" / "workspaces" / w / "output" / uv.RESULT_FILE)
                            .exists() for w in REAL),
                    reason="Faltan las subidas de GDPRtEXT y de la Ley de Colombia")
def test_the_page_with_gdprtext_and_the_law_of_colombia(monkeypatch):
    """The two real uploads, only read: the panel, its criteria and the eight questions."""
    pytest.importorskip("streamlit.testing.v1")
    import importlib
    import sys
    from streamlit.testing.v1 import AppTest
    app_dir = str(ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    view_svc = importlib.import_module("services.align_ontology")
    monkeypatch.setattr(view_svc, "workspace_cfg", lambda: None)       # config.yaml
    for ws_id, mode in REAL.items():
        if not uv.load_result(load_workspace(ws_id)).get("inspectors", {}).get(uv.QUESTIONS):
            pytest.skip(f"{ws_id} se validó antes de las preguntas")
        at = AppTest.from_file(str(ROOT / "app/views/align_ontology.py"), default_timeout=60)
        at.session_state["align_ws"] = ws_id
        at.run()
        assert not at.exception
        shown = _questions_page(at)
        assert shown["criteria"] == ["Criterio: Interoperabilidad", "Criterio: Coherencia",
                                     "Criterio: Completitud", "Criterio: Coherencia"]
        assert shown["zips"] == 4 and svc.QUESTIONS_TAB in shown["tabs"]
        assert len(shown["rows"]) == 8 and shown["queries"] == 8
        assert any(c.value.startswith(f"{mode} · ") for c in at.caption)
