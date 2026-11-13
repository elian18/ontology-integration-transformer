"""S10-T14: "Tú decides" asks the questions of the upload one at a time (with why, and the whole
record only if it is opened), lets the person adjust another concept, and turns the answers into
one new version: "Recalculando tu ontología…" and "¡Lista con los cambios que aprobaste!".
"Mantener todo" makes no version. The LLM and the embedder are fakes."""
from __future__ import annotations

import pytest
from rdflib import Graph

from app.services import align_ontology as svc
from src.alignment import user_alignment as ua
from src.alignment import user_decisions as ud
from src.alignment import user_questions as uq
from src.provenance import upload as up
from src.workspace.store import WorkspaceError, load_workspace
from tests.test_alignment_propose import env, targets  # noqa: F401
from tests.test_ui_align_ontology import _titles, _zips, cfg, page  # noqa: F401
from tests.test_upload_questions import aligned, doubtful  # noqa: F401

EX = "http://example.org/gdpr#"
DPV = "https://w3id.org/dpv#"


@pytest.fixture
def asking(doubtful, env, targets, monkeypatch):  # noqa: F811
    """The test upload with two questions: PersonalData (the AI reviewer doubts it) and
    DataController (a suggestion)."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    uq.refresh_questions(doubtful, env.cfg, targets=targets)
    return doubtful


def test_the_questions_as_the_web_asks_them(asking):
    d = svc.decisions_view(asking.id)
    assert d["version"] == 1 and [q["n"] for q in d["questions"]] == [1, 2]
    first, second = d["questions"]
    assert first == {
        "n": 1, "concept": EX + "PersonalData", "leave": "c",
        "text": "¿Qué hacemos con «PersonalData»? La regla aceptó que es lo mismo que "
                "«PersonalData» y la IA revisora la marcó como incoherente.",
        "why": "La IA revisora: «Son datos, no la categoría».",
        "options": {"a": "Cambiar a: solo está relacionado con PersonalData (lo que propone la "
                         "IA revisora)", "b": "Quitar la correspondencia",
                    "c": "Dejar como está"}}
    assert second["options"] == {"a": "Sí, es casi lo mismo que DataController",
                                 "b": "Dejar como está"}


def test_a_recommended_option_says_so(asking, monkeypatch):
    real = uq.load_questions

    def recommended(ws):
        data = real(ws)
        data["questions"][0]["recommended"] = 1
        return data
    monkeypatch.setattr(uq, "load_questions", recommended)
    options = svc.decisions_view(asking.id)["questions"][0]["options"]
    assert options["b"] == "Quitar la correspondencia (recomendada)"
    assert options["a"].endswith("revisora)")


def test_the_answers_make_one_version(asking, env, targets):  # noqa: F811
    out = svc.apply_answers(asking.id, {1: "a", 2: "b"},
                            [{"concept": EX + "Consent", "action": ud.REMOVE, "note": "x"}],
                            targets=targets)
    assert out == {"applied": True, "version": 2, "left": 1, "refused": [],
                   "changes": ["PersonalData: ahora está relacionado con PersonalData",
                               "Consent: ahora no tiene equivalente"]}
    ws = load_workspace(asking.id, cfg=env.cfg)
    assert ud.current_version(ws) == 2 and uq.kept(ws) == {EX + "DataController"}
    assert svc.decisions_view(asking.id)["questions"] == []            # nothing asked again
    prov = Graph().parse(ws.path / ua.load_manifest(ws)["provenance"]["file"])
    rows = up.records_of(prov, EX + "PersonalData")
    assert [(r["decidedBy"], r["version"], r["current"]) for r in rows if r["decidedBy"]] == [
        ("regla", "1", "false"), ("usuario", "2", "true")]


def test_leaving_everything_makes_no_version(asking, env, targets):  # noqa: F811
    out = svc.apply_answers(asking.id, {1: "c", 2: "b"}, targets=targets)
    assert out == {"applied": False, "version": 1, "left": 2, "changes": [], "refused": []}
    ws = load_workspace(asking.id, cfg=env.cfg)
    assert ud.current_version(ws) == 1 and not (ws.path / ud.HISTORY_DIR).exists()
    assert svc.decisions_view(asking.id)["questions"] == []            # not asked again
    assert svc.apply_answers(asking.id, {}, targets=targets)["applied"] is False


def test_what_cannot_be_applied(asking, env, targets):  # noqa: F811
    out = svc.apply_answers(asking.id, {1: "b"}, [{"concept": EX + "PersonalData",
                                                   "action": ud.REMOVE}], targets=targets)
    assert out["applied"] and out["changes"] == ["PersonalData: ahora no tiene equivalente"]
    assert len(out["refused"]) == 1 and out["refused"][0].startswith("PersonalData: ")
    ws = load_workspace(asking.id, cfg=env.cfg)
    (ws.work_dir / uq.QUESTIONS_FILE).write_text(
        (ws.work_dir / uq.QUESTIONS_FILE).read_text(encoding="utf-8").replace(
            '"version": 2', '"version": 1'), encoding="utf-8")
    assert [q["names"] for q in uq.load_questions(ws)["questions"]] == [["DataController"]]
    with pytest.raises(WorkspaceError, match="ya va en la 2: vuelve a mirarlas"):
        svc.apply_answers(asking.id, {1: "a"}, targets=targets)


def test_another_concept_and_its_options(asking, targets, monkeypatch):  # noqa: F811
    assert svc.adjust_options(asking.id, EX + "PersonalData", targets=targets) == [
        {"text": "Es lo mismo que SensitivePersonalData",
         "change": {"action": "set", "relation": "skos:exactMatch",
                    "dpv": DPV + "SensitivePersonalData", "replaces": DPV + "PersonalData"}},
        {"text": "Es lo mismo que DataSubject",
         "change": {"action": "set", "relation": "skos:exactMatch",
                    "dpv": DPV + "DataSubject", "replaces": DPV + "PersonalData"}},
        {"text": "No tiene equivalente (quitar la correspondencia)",
         "change": {"action": "remove", "relation": None, "dpv": None, "replaces": None}}]
    assert [o["text"] for o in svc.adjust_options(asking.id, EX + "Consent",
                                                  targets=targets)] == [
        "No tiene equivalente (quitar la correspondencia)"]                # inherited
    monkeypatch.setattr(svc, "ADJUST_MAX", 1)
    assert [o["text"] for o in svc.adjust_options(asking.id, EX + "PersonalData",
                                                  targets=targets)] == [
        "Es lo mismo que SensitivePersonalData",                     # not PersonalData again
        "No tiene equivalente (quitar la correspondencia)"]
    monkeypatch.setattr(svc, "ADJUST_MAX", 3)
    pizza = [o["text"] for o in svc.adjust_options(asking.id, EX + "Pizza", targets=targets)]
    assert pizza == ["Es lo mismo que PersonalData", "Es lo mismo que SensitivePersonalData",
                     "Es lo mismo que DataSubject"]                      # no correspondence yet
    assert svc.adjust_options(asking.id, EX + "Nada", targets=targets) == []


# ------------------------------------------------------------------ the page

@pytest.fixture
def deciding(page, asking, targets, monkeypatch):  # noqa: F811
    """The page on "Tú decides" of the test upload; the targets are the small DPV."""
    import streamlit
    from src.alignment import propose as prop
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)
    streamlit.cache_resource.clear()
    at = page()
    at.session_state["align_ws"], at.session_state["align_step"] = asking.id, 4
    at.run()
    yield at
    streamlit.cache_resource.clear()


def test_the_page_asks_one_question_at_a_time(deciding):
    at = deciding
    assert not at.exception and _titles(at) == [svc.DECIDE_TITLE]
    assert at.markdown[0].value == svc.DECIDE_LEAD.format(n="2 detalles")
    assert at.get("progress")[0].proto.text == "Pregunta 1 de 2"
    radio = at.radio(key="align_answer_1_1")
    assert radio.options == ["Cambiar a: solo está relacionado con PersonalData (lo que propone "
                             "la IA revisora)", "Quitar la correspondencia", "Dejar como está"]
    assert radio.value == "c"                                           # nothing until chosen
    assert not [b for b in at.button if b.key == "align_q_prev"]       # the first one
    assert "¿Por qué te lo preguntamos? La IA revisora: «Son datos, no la categoría»." in [
        c.value for c in at.caption]
    assert [e.label for e in at.expander] == ["Ver el registro completo",
                                              "Ajustar otro concepto"]
    assert not [b for b in at.button if b.key == "align_apply"]
    assert at.button(key="align_keep_all").proto.type == "primary"
    radio.set_value("a").run()
    assert at.session_state["align_answers"] == {1: "a"}
    assert at.button(key="align_apply").label == "Aplicar mis respuestas"
    at.button(key="align_q_next").click().run()
    assert at.get("progress")[0].proto.text == "Pregunta 2 de 2"
    assert not [b for b in at.button if b.key == "align_q_next"]
    at.button(key="align_q_prev").click().run()
    assert at.radio(key="align_answer_1_1").value == "a"                # still chosen


def test_the_page_applies_and_shows_the_new_version(deciding, env):  # noqa: F811
    at = deciding
    at.radio(key="align_answer_1_1").set_value("a").run()
    at.button(key="align_apply").click().run()
    assert not at.exception and _titles(at) == [svc.APPLIED_TITLE]
    assert at.markdown[0].value == "- PersonalData: ahora está relacionado con PersonalData"
    assert [e for e in at.get("component_instance")]                    # the new preview
    assert "align_answers" not in at.session_state
    at.button(key="align_decide_done").click().run()
    assert _zips(at) == ["RDF/XML", "OWL", "Turtle", "JSON-LD"]
    ws = load_workspace(at.session_state["align_ws"], cfg=env.cfg)
    assert ud.current_version(ws) == 2


def test_keep_everything_makes_no_version(deciding, env):  # noqa: F811
    at = deciding
    at.radio(key="align_answer_1_1").set_value("b").run()
    at.button(key="align_keep_all").click().run()                        # the answer is dropped
    assert _zips(at) and "align_answers" not in at.session_state
    ws = load_workspace(at.session_state["align_ws"], cfg=env.cfg)
    assert ud.current_version(ws) == 1 and not uq.kept(ws)


def test_the_page_adjusts_another_concept(deciding, env):  # noqa: F811
    at = deciding
    at.selectbox(key="align_adjust_concept").set_value(EX + "Consent").run()
    assert at.radio(key="align_adjust_choice").options == [
        "No tiene equivalente (quitar la correspondencia)"]
    at.button(key="align_adjust_add").click().run()
    at.button(key="align_adjust_add").click().run()                    # once per concept
    assert [e["concept"] for e in at.session_state["align_extra"]] == [EX + "Consent"]
    assert any(m.value == "- Consent: No tiene equivalente (quitar la correspondencia)"
               for m in at.markdown)
    at.button(key=f"align_extra_drop_{EX}Consent").click().run()
    assert at.session_state["align_extra"] == []
    at.button(key="align_adjust_add").click().run()
    at.button(key="align_apply").click().run()
    assert _titles(at) == [svc.APPLIED_TITLE]
    assert at.markdown[0].value == "- Consent: ahora no tiene equivalente"


def test_leaving_the_answers_says_nothing_changed(deciding, env):  # noqa: F811
    at = deciding
    at.radio(key="align_answer_1_1").set_value("a").run()
    at.radio(key="align_answer_1_1").set_value("c").run()               # chosen: leave it
    at.button(key="align_apply").click().run()
    assert _titles(at) == [svc.DECIDE_TITLE] and at.info[0].value == svc.NOTHING_APPLIED
    assert not at.get("component_instance")
    ws = load_workspace(at.session_state["align_ws"], cfg=env.cfg)
    assert ud.current_version(ws) == 1 and uq.kept(ws) == {EX + "PersonalData"}


def test_a_page_without_questions(page, aligned, env, monkeypatch):  # noqa: F811
    at = page()
    at.session_state["align_ws"], at.session_state["align_step"] = aligned.id, 4
    at.run()
    assert not at.exception and at.success[0].value == svc.NOTHING_TO_DECIDE
    assert not at.get("progress") and not at.radio
    at.button(key="align_keep_all").click().run()
    assert _zips(at)
