"""S10-T13: "Revisa tu ontología" first tells, in three short screens and plain words, what was
done: what was found, how it was linked with the DPV and how it was checked. No SHACL, SPARQL,
Jena or IRIs. The LLM and the embedder are fakes."""
from __future__ import annotations

import json

from app.services import align_ontology as svc
from src.validation import upload_validation as uv
from tests.test_alignment_propose import env, targets  # noqa: F401
from tests.test_ui_align_ontology import (_titles, _zips, aligned, both_aligned, cfg,  # noqa: F401
                                          law_aligned, law_cfg, page)

TECHNICAL = ("SHACL", "SPARQL", "Jena", "razonador", "http://", "https://", "skos:", "owl:")


def _all_text(steps) -> str:
    return json.dumps(steps, ensure_ascii=False)


def test_an_ontology_step_by_step(aligned):  # noqa: F811
    found, linked, checked = svc.walkthrough(aligned[0])
    assert [found["title"], linked["title"], checked["title"]] == list(svc.WALK_TITLES)
    assert found["lead"] == "Tu ontología (Mi GDPR.ttl) tiene 4 conceptos."
    assert found["points"] == ["La IA completó los nombres en español e inglés de 4 conceptos."]
    assert found["examples"] == ["**Consentimiento** (Consent)", "**Dato personal** (PersonalData)",
                                 "**Pizza**", "**Responsable del tratamiento** (DataController)"]
    assert linked["lead"] == svc.DPV_LEAD
    assert linked["points"] == [
        "1 ya estaba revisado por una persona en la base del sistema.",
        "La IA eligió el equivalente de 1 entre los candidatos del DPV y la regla del sistema lo "
        "aceptó.",
        "En 1, la IA propuso un equivalente que la regla no aceptó: te lo preguntaremos.",
        "1 no tiene un equivalente en el DPV (no todo concepto lo tiene)."]
    assert linked["examples"] == ["**Consentimiento** (Consent) es igual a *Consent*",
                                  "**Dato personal** (PersonalData) es igual a *PersonalData*"]
    assert checked["lead"] == svc.REVIEW_LEADS["rojo"]
    assert checked["points"] == [
        "Tiene 3 errores de forma nuevos.", "No tiene contradicciones propias.",
        "Responde 2 de 7 preguntas básicas, como «¿Qué conceptos de la ontología están "
        "alineados con el DPV?».",
        "La IA volvió a revisar el enlace que eligió: 1 está bien, 0 son dudosos y 0 parecen "
        "errados."]
    assert not any(word in _all_text([found, linked, checked]) for word in TECHNICAL)


def test_a_law_step_by_step(law_aligned, monkeypatch):  # noqa: F811
    found, linked, _ = svc.walkthrough(law_aligned[0])
    assert found["lead"] == ("Leímos Ley de prueba.txt (4 artículos) y la IA encontró 5 "
                             "conceptos de protección de datos.")
    assert found["points"][1] == "2 ya estaban en la base del sistema y se reutilizan tal cual."
    assert found["examples"][:3] == [
        "**Consent** · sale del art. 1",
        "**consentimiento otorgado** (ConsentGiven) · sale del art. 1",
        "**volumen de negocio** (BusinessVolume) · sale del art. 2"]
    real = svc._law_concepts
    monkeypatch.setattr(svc, "_law_concepts", lambda ws: {"entries": [
        {"name": "Lost", "status": "accepted", "articles": [9, 7]},
        {"name": "Out", "status": "rejected", "articles": [1]}, *real(ws)["entries"]]})
    monkeypatch.setattr(svc, "WALK_EXAMPLES", 10)
    examples = svc.walkthrough(law_aligned[0])[0]["examples"]
    lost = examples.index("**Lost** · sale de los arts. 7 y 9")   # first in the file, but
    assert lost > examples.index("**Consent** · sale del art. 1")   # not in the table: after
    assert not [e for e in examples if "Out" in e]
    assert linked["points"] == [
        "3 ya estaban revisados por una persona en la base del sistema.",
        "2 no tienen un equivalente en el DPV (no todo concepto lo tiene)."]
    assert "**evaluación de impacto** (ImpactAssessment) es igual a *DPIA*" in linked["examples"]


def test_an_ontology_with_a_law_step_by_step(both_aligned, monkeypatch):  # noqa: F811
    found = svc.walkthrough(both_aligned[0])[0]
    assert found["points"][0] == ("Buscamos en ley.txt los artículos más cercanos a cada "
                                  "concepto: 4 de 4 tienen alguno.")
    real = svc.ua.load_manifest
    for skipped, first in ((None, "A 3 les faltaba la definición; tu ley (ley.txt) dio 2."),
                           ("few_missing", found["points"][0])):
        defined = {"missing": 3, "law_concept": 1, "articles": 1, "law": "ley.txt",
                   "skipped": skipped}
        monkeypatch.setattr(svc.ua, "load_manifest",
                            lambda ws, d=defined: {**real(ws), "law_definitions": d})
        assert svc.walkthrough(both_aligned[0])[0]["points"][0] == first
    assert found["examples"][0] == ("**Consentimiento** (Consent) · cerca de los arts. 1, 2 y 3 "
                                    "de tu ley")


def test_plain_words():
    assert svc._arts([25, 4, 89, 4]) == "de los arts. 4, 25 y 89"
    assert svc._arts([4]) == "del art. 4"
    assert svc._name({"concept_name": "Data_Owner", "label_es": "Titular"}) == (
        "**Titular** (Data\\_Owner)")
    assert svc._name({"concept_name": "Consent", "label_es": "consent"}) == "**Consent**"
    assert set(svc.RELATION_VERBS) == set(svc.RELATION_WORDS)
    codes = [c for codes, _ in svc.WALK_STATES for c in codes]
    assert len(codes) == len(set(codes)) == len(svc.STATE_LABELS)       # every state, once


def test_what_the_review_says_in_each_case(aligned, monkeypatch):  # noqa: F811
    real = uv.load_result

    def result(ws, **change):
        r = real(ws)
        for key, values in change.items():
            r["inspectors"][key].update(values)
        return r
    monkeypatch.setattr(uv, "load_result", lambda ws: result(
        ws, shapes={"new": 0, "inherited": 1}, logic={"new": 1, "inherited": 5}))
    points = svc.walkthrough(aligned[0])[2]["points"]
    assert points[:2] == [
        "Está bien formada: cumple las reglas de lo que se entrega; el único problema de forma "
        "que aparece ya estaba en tu archivo o en la base del sistema: no lo causó la "
        "alineación.",
        "Tiene 1 contradicción nueva; las 5 contradicciones que aparecen ya estaban en tu "
        "archivo o en la base del sistema: no las causó la alineación."]
    monkeypatch.setattr(uv, "load_result", lambda ws: result(
        ws, shapes={"status": uv.FAILED}, audit={"status": uv.NOT_RUN}))
    points = svc.walkthrough(aligned[0])[2]["points"]
    assert points[0] == "No tiene contradicciones propias."              # no line for the form
    assert points[-1] == "La IA no pudo volver a revisar los enlaces que eligió."
    monkeypatch.setattr(uv, "load_result", lambda ws: None)
    checked = svc.walkthrough(aligned[0])[2]
    assert checked["lead"] == svc.NO_REVIEW and not checked["points"]


def test_the_page_walks_the_three_steps(page, aligned):  # noqa: F811
    at = page()
    at.session_state["align_ws"] = aligned[0]
    at.run()
    at.button(key="align_review_open").click().run()
    for step, title in enumerate(svc.WALK_TITLES, start=1):
        assert not at.exception and _titles(at) == [title]
        assert at.get("progress")[0].proto.text == f"Paso {step} de 3"
        text = " ".join(m.value for m in at.markdown)
        assert not any(word in text for word in TECHNICAL) and not at.download_button
        nxt = at.button(key="align_walk_next")
        assert nxt.label == ("Continuar" if step == 3 else "Siguiente")
        if step == 1:
            assert "Algunos de tus conceptos" in text
        if step == 2:
            at.button(key="align_walk_back").click().run()               # "Atrás": step 1
            assert _titles(at) == [svc.WALK_TITLES[0]]
            at.button(key="align_walk_next").click().run()
            assert _titles(at) == [svc.WALK_TITLES[1]]
            nxt = at.button(key="align_walk_next")
        nxt.click().run()
    assert _titles(at) == [svc.DECIDE_TITLE]                              # what comes after
    at.button(key="align_decide_back").click().run()                      # back to step 3
    assert _titles(at) == [svc.WALK_TITLES[2]]
    at.button(key="align_walk_back").click().run()
    at.button(key="align_walk_back").click().run()
    at.button(key="align_walk_back").click().run()                        # "¡Lista!"
    assert _titles(at) == [svc.READY_TITLE] and "align_step" not in at.session_state


def test_a_broken_workspace_in_the_steps(page):  # noqa: F811
    at = page()
    at.session_state["align_ws"], at.session_state["align_step"] = "no-existe-00000000", 2
    at.run()
    assert not at.exception and not at.get("progress")
    assert any("No existe el espacio de trabajo" in e.value for e in at.error)
