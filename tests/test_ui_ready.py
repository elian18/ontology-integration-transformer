"""S10-T12: after the run, "¡Tu ontología está lista!" shows a small interactive graph of the
ontology (streamlit-agraph) and one button to review it; what step 3 showed (downloads,
validation, tabs) waits behind that button. The LLM and the embedder are fakes."""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.services import align_ontology as svc
from tests.test_alignment_propose import EX, env, targets  # noqa: F401
from tests.test_ui_align_ontology import _titles, _zips, aligned, cfg, page  # noqa: F401

CONSENT, PERSONAL, CONTROLLER, PIZZA = (f"{EX}{n}" for n in
                                        ("Consent", "PersonalData", "DataController", "Pizza"))
DPV = "https://w3id.org/dpv#"


def test_the_preview_of_an_upload(aligned, monkeypatch):  # noqa: F811
    p = svc.preview(aligned[0])
    assert p["nodes"][0] == {"id": svc.PREVIEW_ROOT, "label": "Mi GDPR.ttl", "kind": "root",
                             "title": "Mi GDPR.ttl"}
    assert [(n["label"], n["kind"]) for n in p["nodes"][1:]] == [
        ("Consent", "aligned"), ("PersonalData", "aligned"), ("Pizza", "none"),
        ("DataController", "none"), ("Consent", "dpv"), ("ConsentRecord", "dpv"),
        ("PersonalData", "dpv")]
    assert p["nodes"][1]["title"] == "Consent: Validada en la base"
    assert p["nodes"][4]["title"] == "DataController: Sugerida"       # not in the ontology
    assert p["nodes"][-1] == {"id": DPV + "PersonalData", "label": "PersonalData",
                              "kind": "dpv", "title": "DPV: PersonalData"}
    assert p["edges"] == [
        {"source": CONSENT, "target": DPV + "Consent", "label": "igual a", "kind": "dpv"},
        {"source": CONSENT, "target": DPV + "ConsentRecord", "label": "relacionado con",
         "kind": "dpv"},
        {"source": PERSONAL, "target": DPV + "PersonalData", "label": "igual a", "kind": "dpv"},
        *({"source": iri, "target": svc.PREVIEW_ROOT, "label": "", "kind": "root"}
          for iri in (CONSENT, PERSONAL, PIZZA, CONTROLLER))]
    assert (p["shown"], p["total"]) == (4, 4)
    assert svc.preview_caption(p) == svc.PREVIEW_CAPTION                    # all of them


def test_which_concepts_are_drawn(aligned, monkeypatch):  # noqa: F811
    monkeypatch.setattr(svc, "PREVIEW_CONCEPTS", 3)
    monkeypatch.setattr(svc, "PREVIEW_ALIGNED", 1)
    monkeypatch.setattr(svc, "_subclasses", lambda ws, manifest, names: [
        (CONTROLLER, CONSENT), (PIZZA, PERSONAL)])
    p = svc.preview(aligned[0])                        # aligned, its neighbour, then the rest
    assert [n["id"] for n in p["nodes"] if n["kind"] in ("aligned", "none")] == [
        CONSENT, CONTROLLER, PIZZA]                    # the rest: without a term first
    assert [n["label"] for n in p["nodes"] if n["kind"] == "dpv"] == ["Consent",
                                                                      "ConsentRecord"]
    assert p["edges"][2:] == [
        {"source": CONTROLLER, "target": CONSENT, "label": "es un tipo de", "kind": "subclass"},
        {"source": CONSENT, "target": "root", "label": "", "kind": "root"},
        {"source": PIZZA, "target": "root", "label": "", "kind": "root"}]   # one graph
    assert svc.preview_caption(p) == "Se ven 3 de tus 4 conceptos. " + svc.PREVIEW_CAPTION
    monkeypatch.setattr(svc, "_subclasses", lambda ws, manifest, names: [])
    p = svc.preview(aligned[0])
    assert [n["id"] for n in p["nodes"] if n["kind"] in ("aligned", "none")] == [
        CONSENT, PIZZA, CONTROLLER]


def test_the_subclasses_come_from_the_ontology(tmp_path):
    (tmp_path / "o.ttl").write_text(f"""
@prefix ex: <{EX}> . @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:Pizza rdfs:subClassOf ex:Consent , ex:Food , [ a ex:Restriction ] .
ex:Consent rdfs:subClassOf ex:Consent .
ex:DataController rdfs:subClassOf ex:PersonalData .
""", encoding="utf-8")
    ws = SimpleNamespace(path=tmp_path)
    names = {CONSENT: "", PIZZA: "", CONTROLLER: "", PERSONAL: ""}
    assert svc._subclasses(ws, {"files": {"ontology": "o.ttl"}}, names) == [
        (CONTROLLER, PERSONAL), (PIZZA, CONSENT)]
    assert svc._subclasses(ws, {"files": {"ontology": "otro.ttl"}}, names) == []
    assert svc._subclasses(ws, {}, names) == []


def test_the_relations_in_plain_words():
    assert set(svc.RELATION_WORDS) == {"exactMatch", "closeMatch", "broadMatch", "narrowMatch",
                                       "relatedMatch", "seeAlso"}
    assert svc.RELATION_WORDS["broadMatch"] == svc.SUBCLASS == "es un tipo de"


def _graph(at) -> dict:
    found = [e for e in at.get("component_instance")
             if e.proto.component_name == "streamlit_agraph.agraph"]
    return json.loads(json.loads(found[0].proto.json_args)["data"]) if found else {}


def test_the_ready_screen(page, aligned):  # noqa: F811
    at = page()
    at.session_state["align_ws"] = aligned[0]
    at.session_state["align_mode"] = "ontology"
    at.run()
    assert not at.exception
    assert _titles(at) == [svc.READY_TITLE] and at.caption[0].value == "Mi GDPR.ttl"
    graph = _graph(at)
    assert [n["label"] for n in graph["nodes"]] == [
        "Mi GDPR.ttl", "Consent", "PersonalData", "Pizza", "DataController", "Consent",
        "ConsentRecord", "PersonalData"]
    assert [n["color"] for n in graph["nodes"]][:6] == ["#31333F", "#4C78A8", "#4C78A8",
                                                        "#BAB0AC", "#BAB0AC", "#F58518"]
    assert [n["shape"] for n in graph["nodes"]][:6] == ["box", "dot", "dot", "dot", "dot",
                                                        "diamond"]
    assert [e["label"] for e in graph["edges"]][:3] == ["igual a", "relacionado con", "igual a"]
    assert [e["color"] for e in graph["edges"]][2:4] == ["#F58518", "#D5DAE5"]
    assert at.caption[1].value == svc.PREVIEW_CAPTION
    button = at.button(key="align_review_open")
    assert button.label == svc.REVIEW_BUTTON and button.proto.type == "primary"
    assert at.caption[2].value == svc.REVIEW_HINT
    assert not at.download_button and not at.success and not at.tabs and not at.dataframe
    assert not [m for m in at.markdown if "concepto" in m.value]          # no figures
    button.click().run()                                     # "Lo que hicimos" (S10-T13)
    assert not at.exception and _titles(at) == [svc.WALK_TITLES[0]] and not at.download_button
    at.button(key="align_walk_back").click().run()                        # back to "lista"
    assert _titles(at) == [svc.READY_TITLE] and not at.download_button
    at.button(key="align_result_back").click().run()                      # back to the files
    assert _titles(at) == ["Mi ontología"] and "align_review" not in at.session_state


def test_a_broken_workspace_on_the_ready_screen(page):  # noqa: F811
    at = page()
    at.session_state["align_ws"] = "no-existe-00000000"
    at.run()
    assert not at.exception and not _graph(at)
    assert any("No existe el espacio de trabajo" in e.value for e in at.error)
    assert at.button(key="align_again")
