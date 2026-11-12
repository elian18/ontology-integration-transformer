"""S10-T05: the questions as one more step of each upload and in the console. They never block
the download; an answer becomes the changes of S10-T03 and a new version; what was answered is
not asked again. The LLM is a fake: no test calls Gemini."""
from __future__ import annotations

import json

import pytest

from src.alignment import decisions as dl
from src.alignment import user_alignment as ua
from src.alignment import user_decisions as ud
from src.alignment import user_questions as uq
from src.workspace.store import load_workspace
from tests.test_alignment_propose import DeciderLLM, env, targets  # noqa: F401
from tests.test_user_decisions import console  # noqa: F401
from tests.test_user_questions import FakeLLM, _align as _ask_align, _q
from tests.test_validation_upload import _align, _real_dpv

EX = "http://example.org/gdpr#"
DPV = "https://w3id.org/dpv#"


def _doubtful(w):
    """Two doubts on the test upload: the AI reviewer marks PersonalData (accepted by the rule)
    as incoherent, and DataController becomes a closeMatch suggestion."""
    path = w.work_dir / "alignment-decisions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for e in data["entries"]:
        if e["name"] == "DataController":
            e["relation"] = "skos:closeMatch"
    path.write_text(json.dumps(data), encoding="utf-8")
    path = w.work_dir / "ai-audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    for e in audit["entries"]:
        if e["name"] == "PersonalData":
            e.update(verdict="incoherente", ai_relation="relatedMatch",
                     explanation="Son datos, no la categoría.")
    path.write_text(json.dumps(audit), encoding="utf-8")
    return w


@pytest.fixture
def aligned(env, targets):  # noqa: F811
    w, _, _ = _align(_real_dpv(env), targets)
    return w


@pytest.fixture
def doubtful(aligned):
    return _doubtful(aligned)


def test_every_upload_writes_its_questions(aligned, env, targets):  # noqa: F811
    data = uq.load_questions(aligned)                         # the upload already did it
    assert (data["version"], data["doubts"], data["questions"]) == (1, 0, [])
    assert aligned.manifest["questions"] == {"version": 1, "asked": 0, "doubts": 0}
    assert uq.render_questions(data, aligned.id) == [
        "  Para revisar: ninguna pregunta; no quedó nada en duda."]
    _doubtful(aligned)
    data = uq.refresh_questions(aligned, env.cfg, targets=targets)
    assert [(q["n"], q["names"], [o["letter"] for o in q["options"]], q["leave"])
            for q in data["questions"]] == [(1, ["PersonalData"], ["a", "b"], "c"),
                                            (2, ["DataController"], ["a"], "b")]
    assert data["questions"][0]["options"][0]["apply"] == [{
        "concept": EX + "PersonalData", "action": ud.SET, "relation": "skos:relatedMatch",
        "dpv": DPV + "PersonalData", "replaces": None}]
    assert load_workspace(aligned.id, cfg=env.cfg).manifest["questions"] == {
        "version": 1, "asked": 2, "doubts": 2}
    assert uq.render_questions(data, aligned.id) == [
        "  Para revisar (opcional): 2 preguntas sobre 2 dudas de la versión 1",
        "    1. ¿Qué hacemos con «PersonalData»? La regla aceptó que es lo mismo que "
        "«PersonalData» y la IA revisora la marcó como incoherente.",
        "       a) Cambiar a: solo está relacionado con PersonalData (lo que propone la IA "
        "revisora)",
        "       b) Quitar la correspondencia",
        "       c) Dejar como está",
        "       Por qué se pregunta: La IA revisora: «Son datos, no la categoría».",
        "    2. ¿«DataController» es casi lo mismo que «DataController» del DPV?",
        "       a) Sí, es casi lo mismo que DataController",
        "       b) Dejar como está",
        "       Por qué se pregunta: La IA lo propuso: «Decisión de prueba para "
        "DataController». La regla no lo aceptó: no pasó la regla.",
        f'  Para responder: py -m src.alignment.user_alignment --espacio {aligned.id} '
        f'--responder "1=a,2=a"']


def test_answers_make_a_new_version_and_are_not_asked_again(doubtful, env, targets):  # noqa: F811
    w = doubtful
    uq.refresh_questions(w, env.cfg, targets=targets)
    result = uq.answer(w, "1=b, 2=b", cfg=env.cfg, targets=targets, llm=DeciderLLM())
    assert result["applied"] and result["version"] == 2 and result["left"] == 1
    [record] = ud.read(w)
    assert (record.concept_key, record.decision, record.reviewer) == (
        EX + "PersonalData", dl.DECISION_NO_MATCH, "usuario")
    assert record.note == ("¿Qué hacemos con «PersonalData»? La regla aceptó que es lo mismo "
                           "que «PersonalData» y la IA revisora la marcó como incoherente. → "
                           "Quitar la correspondencia")
    assert uq.kept(w) == {EX + "DataController"}
    row = json.loads(uq.kept_path(w).read_text(encoding="utf-8"))
    assert (row["version"], row["question"]) == (1, "¿«DataController» es casi lo mismo que "
                                                    "«DataController» del DPV?")
    data = uq.load_questions(w)
    assert (data["version"], data["questions"], data["kept"]) == (2, [], 1)
    assert result["questions"] == data and (w.path / ud.HISTORY_DIR / "v1").exists()
    assert uq.render_questions(data, w.id) == [
        "  Para revisar: ninguna pregunta; no quedó nada en duda (dejaste 1 concepto como "
        "estaban)."]
    llm = DeciderLLM()                                    # a normal run asks nothing again
    w, manifest, _ = ua.align_workspace(w, llm=llm, base=env.base, targets=targets, cfg=env.cfg,
                                        chroma_path=env.chroma, sleep_fn=lambda s: None)
    assert llm.calls == 0 and uq.load_questions(w)["questions"] == []
    assert manifest["alignment_version"] == 2


def test_leaving_everything_keeps_the_version(doubtful, env, targets):  # noqa: F811
    w = doubtful
    uq.refresh_questions(w, env.cfg, targets=targets)
    result = uq.answer(w, "2=b", cfg=env.cfg, targets=targets)
    assert not result["applied"] and result["version"] == 1 and result["left"] == 1
    assert not ud.log_path(w).exists() and not (w.path / ud.HISTORY_DIR).exists()
    assert [q["names"] for q in result["questions"]["questions"]] == [["PersonalData"]]


def test_what_the_answers_cannot_be(doubtful, env, targets):  # noqa: F811
    w = doubtful
    with pytest.raises(ValueError, match="no tiene preguntas pendientes"):
        uq.answer(w, "1=a", cfg=env.cfg, targets=targets)        # the edits came after it
    data = uq.refresh_questions(w, env.cfg, targets=targets)
    for text, error in [
            ("1:a", "No entiendo «1:a». Escribe las respuestas como «1=a,3=b»."),
            ("3=a", "No hay pregunta 3: hay 2 preguntas."),
            ("2=c", "La pregunta 2 no tiene la opción «c»: elige a o b."),
            ("1=d", "La pregunta 1 no tiene la opción «d»: elige a, b o c."),
            ("1=a,1=b", "Respondiste dos veces la pregunta 1."),
            (" , ", "No hay respuestas. Escríbelas como «1=a,3=b».")]:
        with pytest.raises(ValueError) as exc:
            uq.parse_answers(text, data)
        assert str(exc.value) == error
    assert uq.parse_answers("2=B , 1 = a", data) == {2: "b", 1: "a"}
    w.manifest["alignment_version"] = 2                       # someone changed it meanwhile
    with pytest.raises(ValueError, match="Las preguntas son de la versión 1 y la ontología ya "
                                         "va en la 2: vuelve a verlas con --espacio"):
        uq.answer(w, "1=a", cfg=env.cfg, targets=targets)


def test_the_ai_asks_only_when_it_is_on(doubtful, env, targets, monkeypatch):  # noqa: F811
    monkeypatch.setattr(uq, "ask_settings", lambda cfg=None: uq.AskSettings(ai=True,
                                                                             pause_seconds=0))
    answer = {"preguntas": [_q(["D2"], ("Sí", [_ask_align("D2", "closeMatch",
                                                          "DataController")]))],
              "sin_pregunta": [{"duda": "D1", "motivo": "Está bien."}]}
    llm = FakeLLM(answer)
    data = uq.refresh_questions(doubtful, env.cfg, targets=targets, llm=llm)
    assert llm.calls == 1 and data["source"] == "ask" and data["prompt"]["version"] == 1
    assert [(q["names"], q["recommended"]) for q in data["questions"]] == [
        (["DataController"], 0)]
    assert (doubtful.path.parent / "_cache" / uq.CACHE_FILE).exists()       # shared cache
    fixed = uq.refresh_questions(doubtful, env.cfg, targets=targets, use_llm=False)
    assert fixed["source"] == "preguntas fijas" and len(fixed["questions"]) == 2


# ------------------------------------------------------------------ console

def test_the_console_shows_them_and_answers_them(doubtful, console, capsys):  # noqa: F811
    w = doubtful
    assert ua._run(["--espacio", w.id, "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "  Para revisar (opcional): 2 preguntas sobre 2 dudas de la versión 1\n" in out
    assert out.rstrip().endswith(f'--espacio {w.id} --responder "1=a,2=a"')
    assert ua._run(["--espacio", w.id, "--responder", "1=a,2=b", "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Tus respuestas\n  1. ¿Qué hacemos con «PersonalData»?")
    assert "     a) Cambiar a: solo está relacionado con PersonalData (lo que propone la IA " \
           "revisora)\n" in out
    assert "     b) Dejar como está\n" in out
    assert "    decididas por el usuario: 1 correspondencia en 1 concepto\n" in out
    assert ("  Para revisar: ninguna pregunta; no quedó nada en duda (dejaste 1 concepto como "
            "estaban).\n") in out
    assert out.rstrip().endswith("La versión 1 quedó en history/v1/ con sus cuatro descargas.")
    assert ua._run(["--espacio", w.id, "--responder", "1=a"]) == 1
    assert capsys.readouterr().out == "Error: Este espacio no tiene preguntas pendientes.\n"


def test_the_console_when_nothing_changes_and_its_errors(doubtful, console, capsys):  # noqa: F811
    w = doubtful
    assert ua._run(["--espacio", w.id, "--sin-ia"]) == 0
    capsys.readouterr()
    assert ua._run(["--espacio", w.id, "--responder", "2=b"]) == 0
    out = capsys.readouterr().out
    assert ("  Nada que cambiar: la versión 1 sigue igual. Lo que dejaste como está no se "
            "vuelve a preguntar.\n") in out
    assert "  Para revisar (opcional): 1 pregunta sobre 1 duda de la versión 1\n" in out
    for argv, error in [
            (["--responder", "1=a"], "Error: indica un archivo o --espacio."),
            (["x.ttl", "--responder", "1=a"],
             "Error: --responder va con --espacio y sin --cambio."),
            (["--espacio", w.id, "--responder", "1=a", "--cambio", "Pizza quitar"],
             "Error: --responder va con --espacio y sin --cambio."),
            (["--espacio", w.id, "--responder", "1=z"],
             "Error: La pregunta 1 no tiene la opción «z»: elige a, b o c.")]:
        assert ua._run(argv) == 1
        assert capsys.readouterr().out == error + "\n"
