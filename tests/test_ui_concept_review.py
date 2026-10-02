"""S5-T03: the "Aprobar conceptos IA" page (service logic + the page itself with AppTest)."""
import importlib
import json
import sys
from pathlib import Path

import pytest

from app.services import concept_review as cr
from src.alignment.decisions import read_log, review_settings, DECISIONS_FILE, DECISION_PENDING

ROOT = Path(__file__).resolve().parents[1]
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
CFG = {"review": {"source_id": "ontopriv+lopdp", "reviewer": "Elian"}}


def _row(key, origin, name, **kw):
    base = {"concept_key": key, "origin": origin, "concept_name": name, "concept_label": None,
            "concept_kinds": ["class"], "concept_definition": None, "concept_articles": [],
            "duplicate_status": None, "duplicate_of": None, "duplicate_of_name": None,
            "duplicate_score": None, "duplicate_reason": None, "rank": 1,
            "dpv_iri": "https://w3id.org/dpv#X", "dpv_name": "X", "score": 0.5,
            "proposed_relation": None}
    base.update(kw)
    return base


def _files(tmp_path):
    rows = [
        _row(ONTO + "Consent", "ontology", "Consent", concept_family="Principles"),
        _row(ONTO + "Right_to_portability", "ontology", "Right_to_portability",
             concept_family="Rights"),
        _row("ai:DataBreachNotice", "ai", "DataBreachNotice", duplicate_status="new",
             concept_label="Notificacion de vulneracion", concept_articles=[46],
             concept_definition="Aviso a la autoridad."),
        _row("ai:PortabilityRight", "ai", "PortabilityRight",
             duplicate_status="possible_duplicate", duplicate_of=ONTO + "Right_to_portability",
             duplicate_of_name="Right_to_portability", duplicate_score=0.82,
             duplicate_reason="score", concept_label="Derecho a la portabilidad",
             concept_articles=[17]),
        _row("ai:Consent", "ai", "Consent", duplicate_status="possible_duplicate",
             duplicate_of=ONTO + "Consent", duplicate_of_name="Consent", duplicate_score=1.0,
             duplicate_reason="same_name", concept_articles=[8]),
    ]
    cand = tmp_path / "alignment-candidates.json"
    cand.write_text(json.dumps({"metadata": {"embedding_model": "m"}, "rows": rows}),
                    encoding="utf-8")
    return cand, tmp_path / "review" / DECISIONS_FILE


# ---------------------------------------------------------------- service
def test_load_returns_none_without_candidates(tmp_path):
    assert cr.load(candidates_path=tmp_path / "no.json", log_path=tmp_path / "l.jsonl",
                   cfg=CFG) is None


def test_load_lists_only_ai_concepts_as_pending(tmp_path):
    cand, log = _files(tmp_path)
    state = cr.load(candidates_path=cand, log_path=log, cfg=CFG)
    assert [r["name"] for r in state["rows"]] == ["DataBreachNotice", "PortabilityRight", "Consent"]
    assert state["progress"]["pending"] == 3
    assert all(r["status"] == DECISION_PENDING for r in state["rows"])
    families = {r["name"]: r["duplicate_of_family"] for r in state["rows"]}
    assert families == {"DataBreachNotice": None, "PortabilityRight": "Rights",
                        "Consent": "Principles"}


def test_apply_writes_the_log_and_reload_shows_it(tmp_path):
    cand, log = _files(tmp_path)
    state = cr.load(candidates_path=cand, log_path=log, cfg=CFG)
    ok, msg = cr.apply(state, "ai:DataBreachNotice", "approve", "Elian", entity_kind="class")
    assert ok and "aprobado como concepto nuevo" in msg
    ok, msg = cr.apply(state, "ai:PortabilityRight", "duplicate", "Elian")
    assert ok
    state = cr.load(candidates_path=cand, log_path=log, cfg=CFG)
    rows = {r["name"]: r for r in state["rows"]}
    assert rows["DataBreachNotice"]["status_label"] == "aprobado"
    assert rows["PortabilityRight"]["status_label"] == "ya existe en OntoPriv"
    assert state["progress"]["pending"] == 1
    assert len(read_log(log)) == 2


def test_apply_returns_spanish_errors_instead_of_raising(tmp_path):
    cand, log = _files(tmp_path)
    state = cr.load(candidates_path=cand, log_path=log, cfg=CFG)
    ok, msg = cr.apply(state, "ai:Consent", "approve", "Elian", entity_kind="class")
    assert not ok and "Ya existe en OntoPriv" in msg
    ok, msg = cr.apply(state, "ai:DataBreachNotice", "reject", "  ")
    assert not ok and "revisor" in msg
    assert not log.exists()


def test_new_concept_marked_as_existing_entity(tmp_path):
    cand, log = _files(tmp_path)
    state = cr.load(candidates_path=cand, log_path=log, cfg=CFG)
    assert [e["name"] for e in state["ontology"]] == ["Consent", "Right_to_portability"]
    assert cr.entity_label(state["ontology"][1]) == \
        "Right_to_portability  (Rights · clase · ley-organica-proteccion-datos-personales)"

    ok, msg = cr.apply(state, "ai:DataBreachNotice", "duplicate", "Elian")
    assert not ok and "Elige la entidad de OntoPriv" in msg
    ok, _ = cr.apply(state, "ai:DataBreachNotice", "duplicate", "Elian", same_as=ONTO + "Consent")
    assert ok
    state = cr.load(candidates_path=cand, log_path=log, cfg=CFG)
    row = next(r for r in state["rows"] if r["name"] == "DataBreachNotice")
    assert row["status_label"] == "ya existe en OntoPriv" and row["same_as_name"] == "Consent"


def test_same_name_entities_are_told_apart_in_the_picker():
    """OntoPriv has two 'Confidentiality': a class (principle) and a datatype property."""
    a = {"name": "Confidentiality", "family": "Principles", "kinds": ["class"],
         "module": "ley-organica-proteccion-datos-personales"}
    b = {"name": "Confidentiality", "family": "Principles", "kinds": ["property"],
         "module": "OntologiaLOPDP"}
    assert cr.entity_label(a) != cr.entity_label(b)
    assert cr.entity_label(b) == "Confidentiality  (Principles · propiedad · OntologiaLOPDP)"
    assert cr.entity_label({"name": "X"}) == "X"


def test_filters_and_next_pending(tmp_path):
    cand, log = _files(tmp_path)
    state = cr.load(candidates_path=cand, log_path=log, cfg=CFG)
    rows = state["rows"]
    assert [r["name"] for r in cr.filter_rows(rows, mark="possible_duplicate")] == \
        ["PortabilityRight", "Consent"]
    assert [r["name"] for r in cr.filter_rows(rows, status=None, query="portabilidad")] == \
        ["PortabilityRight"]
    assert [r["name"] for r in cr.filter_rows(rows, status=None, query="46")] == \
        ["DataBreachNotice"]
    assert cr.next_pending(rows) == "ai:DataBreachNotice"
    assert cr.next_pending(rows, after_key="ai:Consent") == "ai:DataBreachNotice"   # wraps
    rows[1]["status"] = "approved"
    assert cr.next_pending(rows, after_key="ai:DataBreachNotice") == "ai:Consent"


def test_labels_for_the_page():
    assert cr.kind_options() == {"clase": "class", "propiedad de objeto": "object_property",
                                 "propiedad de datos": "datatype_property"}
    row = {"name": "PortabilityRight", "label": "Derecho a la portabilidad",
           "status_label": "por validar"}
    assert cr.option_label(row) == "PortabilityRight · Derecho a la portabilidad  [por validar]"


def test_default_paths_follow_the_config():
    paths = cr.default_paths()
    assert paths["log"] == ROOT / "data/review" / DECISIONS_FILE
    assert paths["candidates"].name == "alignment-candidates.json"


def test_page_is_registered_in_the_navigation():
    text = (ROOT / "app/app.py").read_text(encoding="utf-8")
    assert 'st.Page("views/concept_review.py", title="Aprobar conceptos IA"' in text
    assert text.index("views/concept_review.py") < text.index("views/alignment.py")


# ---------------------------------------------------------------- the page (AppTest)
@pytest.fixture
def page(tmp_path, monkeypatch):
    """Run the real view with the service pointed at temporary files."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    app_dir = str(ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    # The view imports "services.concept_review" (app/ on sys.path, as app/app.py does), which is
    # a different module object from "app.services.concept_review"; patch the one the view uses.
    # Imported by name at run time because app/ is only on the path while the tests run.
    view_service = importlib.import_module("services.concept_review")
    cand, log = _files(tmp_path)
    monkeypatch.setattr(view_service, "default_paths",
                        lambda: {"candidates": cand, "log": log})
    at = AppTest.from_file(str(ROOT / "app/views/concept_review.py"), default_timeout=30)
    return at, log


def test_page_shows_the_first_pending_concept(page):
    at, _ = page
    at.run()
    assert not at.exception
    assert at.header[0].value == "Aprobar conceptos propuestos por la IA"
    assert at.subheader[0].value == "DataBreachNotice"
    assert at.metric[0].value == "3"                           # por validar
    # the page pre-fills the reviewer configured in config.yaml (review.reviewer)
    assert at.text_input(key="reviewer").value == review_settings()["reviewer"]


def test_approving_from_the_page_writes_the_log_and_moves_on(page):
    at, log = page
    at.run()
    at.button[1].click().run()                                 # "Aprobar como nuevo"
    assert not at.exception
    records = read_log(log)
    assert [(d.concept_key, d.decision, d.entity_kind, d.reviewer) for d in records] == \
        [("ai:DataBreachNotice", "approved", "class", review_settings()["reviewer"])]
    assert at.subheader[0].value == "PortabilityRight"         # next pending
    assert "familia Rights" in at.info[0].value
    assert "aprobado como concepto nuevo" in at.success[0].value
    assert at.metric[0].value == "2"


def test_same_name_concept_has_approve_disabled(page):
    at, _ = page
    at.run()
    at.selectbox[2].set_value("ai:Consent").run()
    assert at.subheader[0].value == "Consent"
    approve = at.button[1]
    assert approve.label == "Aprobar como nuevo" and approve.disabled
    assert not at.button[3].disabled                            # "Ya existe en OntoPriv"
    assert at.selectbox(key="same_ai:Consent").value == ONTO + "Consent"   # preset


def test_new_concept_needs_an_entity_before_marking_it_existing(page):
    at, log = page
    at.run()
    assert at.subheader[0].value == "DataBreachNotice"
    assert at.button[3].label == "Ya existe en OntoPriv" and at.button[3].disabled
    at.selectbox(key="same_ai:DataBreachNotice").set_value(ONTO + "Consent").run()
    assert not at.button[3].disabled
    at.button[3].click().run()
    assert not at.exception
    d = read_log(log)[0]
    assert (d.concept_key, d.decision, d.same_as) == ("ai:DataBreachNotice", "duplicate",
                                                      ONTO + "Consent")