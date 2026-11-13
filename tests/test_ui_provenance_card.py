"""S10-T08: in step 3, choosing a concept under "Conceptos" shows its card: the article it comes
from, how it was produced, who decided its correspondence and its versions, read with the fixed
query of its provenance module. A concept without correspondence says why. The LLM is a fake."""
from __future__ import annotations

import json
import os

import pytest

from app.services import align_ontology as svc
from src.alignment import user_alignment as ua
from src.provenance import upload as up
from tests.test_alignment_propose import env, targets  # noqa: F401
from tests.test_provenance_upload import (CORE, ONTO, PEOPLE, _inputs,  # noqa: F401
                                          _removed_and_aligned_again)
from tests.test_ui_align_ontology import cfg, page  # noqa: F401
from tests.test_user_decisions import EX, _apply, aligned  # noqa: F401


@pytest.fixture(scope="module")
def toy():
    """Version 3 of the upload of test_provenance_upload (PersonalData changed in v2, Consent
    taken out and Pizza aligned in v3)."""
    g, _ = up.build_graph(_inputs(), PEOPLE)
    return g


def _card(g, name, rows=None, **kw):
    return svc.card(up.records_of(g, ONTO + name), ONTO + name, rows or [],
                    **{"version": 3, "file": "Mi ontología.ttl", "mode": "ontology", **kw})


def _row(name, status):
    return {"concept_iri": ONTO + name, "concept_name": name, "status": status}


# ------------------------------------------------------------------ the card

def test_a_correspondence_the_person_changed_shows_both_versions(toy):
    c = _card(toy, "PersonalData", [_row("PersonalData", ua.USER_ALIGNED)])
    assert (c["missing"], c["name"], c["iri"], c["state"]) == (
        None, "PersonalData", ONTO + "PersonalData", "Decidida por el usuario")
    assert c["headline"] == ("**Correspondencia en la versión 3:**  \n"
                             "skos:closeMatch dpv:PersonalData · decidida por **usuario** en la "
                             "versión 2")
    assert c["versions"] == [
        {"Versión": 1, "Correspondencia": "skos:exactMatch dpv:PersonalData",
         "Decidida por": "regla", "Cuándo": "2026-11-01 00:00 UTC",
         "Estado": "reemplazada el 2026-11-02 10:00 UTC"},
        {"Versión": 2, "Correspondencia": "skos:closeMatch dpv:PersonalData",
         "Decidida por": "usuario", "Cuándo": "2026-11-02 10:00 UTC", "Estado": "vigente"}]
    old, new = c["steps"]
    assert old["badge"] == ":gray-badge[Reemplazada]" and new["badge"] == ":green-badge[Vigente]"
    assert old["meta"] == ("Decisor: candidatos del DPV, IA y regla · "
                           "ontology-integration-transformer · 2026-11-01 00:00 UTC")
    assert old["extra"][0].startswith("IA revisora: PersonalData dpv:PersonalData, coherente")
    assert new["meta"] == "Cambios del usuario (versión 2) · 2026-11-02 10:00 UTC"
    assert new["extra"] == ["Revisa: PersonalData skos:exactMatch dpv:PersonalData (regla, "
                            "versión 1)"]
    assert c["origin"] == ("**De qué artículo sale:** ninguno: viene de tu ontología (Mi "
                           "ontología.ttl), no de un artículo de una ley.")


def test_an_inherited_one_says_who_decided_it_in_the_base(toy):
    c = _card(toy, "Registry")
    assert c["headline"].endswith("skos:broadMatch dpv:ProtectionOfNationalSecurity · decidida "
                                  "por **base (Elian Gavilanes)** en la versión 1")
    assert [v["Decidida por"] for v in c["versions"]] == ["base (Elian Gavilanes)"]
    assert c["steps"][-1]["text"].startswith("Heredada de NationalRegistry de la base (muy "
                                             "parecido, puntaje 0.7512): la decidió Elian "
                                             "Gavilanes el 2026-10-02 en la base.")
    assert c["name"] == "Registry" and c["state"] is None
    fee = _card(toy, "Fee")
    assert fee["headline"].endswith("rdfs:seeAlso dpv:Fee · decidida por **regla sin IA** en la "
                                    "versión 1")


def test_a_concept_without_correspondence_says_why(toy):
    pizza = _card(toy, "Lost")
    assert pizza["headline"] == ("**Sin correspondencia en la versión 3.** La IA no estuvo "
                                 "disponible y la regla sin IA no encontró un candidato con "
                                 "puntaje ≥ 0.75.")
    assert pizza["versions"] == [] and pizza["steps"][0]["badge"] == ":green-badge[Vigente]"
    rule = _card(toy, "Rule")
    assert rule["headline"] == ("**Sin correspondencia en la versión 3.** Heredada de Rule de la "
                                "base (mismo IRI): en la base no tiene correspondencia con el "
                                "DPV.")
    thing = _card(toy, "Thing")
    assert thing["headline"] == (
        "**Sin correspondencia en la versión 3.** La IA propuso skos:relatedMatch dpv:Thing con "
        "confianza alta, pero la regla no la acepta (no acepta skos:relatedMatch; puntaje 0.4 "
        "\\< 0.5): no entra a la ontología.")
    assert thing["origin"] == (
        "**De qué artículo sale:** ninguno: viene de tu ontología (Mi ontología.ttl), no de un "
        "artículo de una ley; su definición sale de ley.txt, art. 3.")


def test_what_the_person_took_out_and_what_came_back(toy):
    consent = _card(toy, "Consent", mode="law", file="Perfil de ley.txt")
    assert consent["headline"] == ("**Sin correspondencia en la versión 3.** El usuario quitó "
                                   "exactMatch Consent (base, versión 1).")
    assert consent["steps"][-1]["text"].endswith("(base, versión 1). Pedido: Consent quitar")
    assert consent["origin"] == ("**De qué artículo sale:** ninguno: viene de Perfil de ley.txt, "
                                 "sin un artículo citado.")
    law = svc.card(up.records_of(toy, CORE + "Consent"), CORE + "Consent", [])
    assert law["origin"] == "**De qué artículo sale:** ley.txt, arts. 1 y 2"
    assert law["steps"][0]["title"] == "consentimiento (Consent)" and not law["versions"]
    assert [(v["Versión"], v["Correspondencia"], v["Decidida por"], v["Estado"])
            for v in consent["versions"]] == [
        (1, "skos:exactMatch dpv:Consent", "base (Elian Gavilanes)",
         "reemplazada el 2026-11-03 10:00 UTC"),
        (3, "ninguna (quitada)", "usuario", "vigente")]
    pizza = _card(toy, "Pizza")
    assert pizza["headline"].endswith("skos:exactMatch dpv:Food · decidida por **usuario** en la "
                                      "versión 3")
    why = next(s for s in pizza["steps"] if "sin correspondencia" in s["title"])
    assert why["badge"] == ":gray-badge[Reemplazada]"
    g, _ = up.build_graph(_removed_and_aligned_again(), PEOPLE)
    back = _card(g, "Consent")
    assert back["headline"].endswith("skos:closeMatch dpv:Policy · decidida por **usuario** en "
                                     "la versión 3")
    assert [v["Estado"] for v in back["versions"]] == [
        "reemplazada el 2026-11-02 10:00 UTC", "reemplazada el 2026-11-03 10:00 UTC", "vigente"]


def _rec(local, label, comment=None, **kw):
    return {"record": f"http://example.org/mi-provenance#{local}", "label": label,
            "comment": comment, **kw}


def test_the_latest_reason_holds_while_nothing_is_in_force():
    """Modules written before S10-T08 did not replace a removal the person undid later."""
    why = [_rec("decider-1", "x: sin propuesta", "La IA dudó."),
           _rec("user-removal-2", "x: quitada", "El usuario quitó todo.", version="2")]
    assert svc.card(why, ONTO + "x", [])["headline"].endswith("** El usuario quitó todo.")
    law = [_rec("decider-9", "x", "Según el art. 4 de la ley. Los arts. 5 y 6. Más.")]
    assert svc.card(law, ONTO + "x", [])["headline"].endswith("** Según el art. 4 de la ley.")
    back = _rec("correspondence-3", "x skos:exactMatch dpv:X (usuario, versión 3)",
                decidedBy="usuario", who="usuario", version="3", current="true",
                property="http://www.w3.org/2004/02/skos/core#exactMatch",
                target="https://w3id.org/dpv#X")
    c = svc.card([*why, back], ONTO + "x", [])
    assert c["headline"].endswith("skos:exactMatch dpv:X · decidida por **usuario** en la "
                                  "versión 3")
    assert [s["badge"] for s in c["steps"]] == [":gray-badge[Reemplazada]"] * 2 + [
        ":green-badge[Vigente]"]
    assert [v["Estado"] for v in c["versions"]] == ["reemplazada", "vigente"]


def test_what_the_provenance_does_not_mention():
    c = svc.card([], ONTO + "alice", [_row("alice", ua.NOT_ALIGNABLE)], version=1, file="a.ttl")
    assert c["headline"] == ("**Sin correspondencia en la versión 1.** No se alinea. Individuos "
                             "(ejemplos de datos): no se alinean, igual que en la base.")
    assert c["steps"] == [] and c["versions"] == [] and c["state"] == "No se alinea"
    assert svc.card([], ONTO + "x_y", [])["headline"] == (
        "**Sin correspondencia en la versión 1.** La procedencia no dice por qué.")
    assert svc.card([], ONTO + "x_y", [])["name"] == "x\\_y"                   # markdown
    named = svc.card([], ONTO + "c-12", [{"concept_name": "Algo", "status": ua.ACCEPTED}])
    assert named["name"] == "Algo"


def test_the_articles_are_grouped_by_law():
    assert svc._articles(["ley.pdf, art. 25", "ley.pdf, art. 21", "otra.txt, art. 3",
                          "ley.pdf, art. 21", "anexo", "anexo"]) == (
        "ley.pdf, arts. 21 y 25; otra.txt, art. 3; anexo")
    assert svc._articles(["ley.pdf, art. 9", "ley.pdf, art. 10", "ley.pdf, art. 2"]) == (
        "ley.pdf, arts. 2, 9 y 10")


def test_the_concepts_to_choose_from():
    raw = [_row("Pizza", ua.RULE_NO_MATCH), _row("consent", ua.INHERITED_ALIGNED),
           _row("consent", ua.ACCEPTED), {"concept_iri": "", "concept_name": "x"},
           {"concept_iri": ONTO + "Bare", "concept_name": "", "status": "otro"}]
    assert svc.concept_options(raw) == {ONTO + "Bare": "Bare · otro",
                                        ONTO + "consent": "consent · Validada en la base",
                                        ONTO + "Pizza": "Pizza · Sin correspondencia"}


# ------------------------------------------------------------------ from the workspace

def test_the_card_is_read_from_the_module_of_the_upload(aligned, env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    v = svc.view(aligned.id)
    assert list(v["concept_options"].values()) == [
        "Consent · Validada en la base", "DataController · Sugerida",
        "PersonalData · Aceptada automáticamente", "Pizza · Sin correspondencia"]
    iri = next(i for i, label in v["concept_options"].items() if label.startswith("Personal"))
    c = svc.provenance_card(aligned.id, iri)
    assert c["headline"].endswith("skos:exactMatch dpv:PersonalData · decidida por **regla** en "
                                  "la versión 1")
    assert c["state"] == "Aceptada automáticamente"
    pizza = next(i for i in v["concept_options"] if i.endswith("Pizza"))
    assert svc.provenance_card(aligned.id, pizza)["headline"].startswith(
        "**Sin correspondencia en la versión 1.** La IA dijo que ningún candidato del DPV "
        "corresponde.")


def test_after_a_change_the_card_has_both_versions(aligned, env, targets,  # noqa: F811
                                                    monkeypatch):
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    iri = next(i for i in svc.view(aligned.id)["concept_options"] if i.endswith("PersonalData"))
    before = svc.provenance_card(aligned.id, iri)
    _apply(aligned, env, targets, "PersonalData broadMatch PersonalData")
    path = aligned.path / ua.load_manifest(aligned)["provenance"]["file"]
    os.utime(path, ns=(1, 1))                     # same size, another time: read it again
    after = svc.provenance_card(aligned.id, iri)
    assert before["versions"][0]["Estado"] == "vigente"
    assert [(v["Versión"], v["Decidida por"], v["Estado"][:11]) for v in after["versions"]] == [
        (1, "regla", "reemplazada"), (2, "usuario", "vigente")]
    assert after["headline"].startswith("**Correspondencia en la versión 2:**  \nskos:broadMatch")


def test_an_alignment_older_than_the_provenance(aligned, env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    path = aligned.output_dir / ua.MANIFEST_FILE
    m = json.loads(path.read_text(encoding="utf-8"))
    entry = m.pop("provenance")
    path.write_text(json.dumps(m), encoding="utf-8")
    assert svc.provenance_card(aligned.id, ONTO + "x") == {"missing": svc.NO_PROVENANCE}
    m["provenance"] = entry
    path.write_text(json.dumps(m), encoding="utf-8")
    (aligned.path / entry["file"]).unlink()
    assert svc.provenance_card(aligned.id, ONTO + "x")["missing"] == svc.NO_PROVENANCE


# ------------------------------------------------------------------ the page

def _card_text(at) -> list[str]:
    return [m.value for m in at.markdown]


def test_the_page_shows_the_card_of_the_chosen_concept(page, aligned, env, targets):  # noqa: F811
    _apply(aligned, env, targets, "PersonalData broadMatch PersonalData")
    at = page()
    at.session_state["align_ws"] = aligned.id
    at.session_state["align_review"] = True              # the review (S10-T12)
    at.run()
    box = at.selectbox(key="align_concept")
    assert box.value is None and box.label == "Concepto" and box.help == svc.PROVENANCE_HELP
    assert box.options == ["Consent · Validada en la base", "DataController · Sugerida",
                           "PersonalData · Decidida por el usuario",
                           "Pizza · Sin correspondencia"]
    assert f"**{svc.PROVENANCE_TITLE}**" in _card_text(at)
    assert not any(m.startswith("#### ") for m in _card_text(at))            # nothing chosen
    iri = str(EX.PersonalData)
    box.set_value(iri).run()
    assert not at.exception
    text = _card_text(at)
    assert "#### PersonalData :gray-badge[Decidida por el usuario]" in text
    assert ("**Correspondencia en la versión 2:**  \nskos:broadMatch dpv:PersonalData · decidida "
            "por **usuario** en la versión 2") in text
    assert any(m.startswith("**De qué artículo sale:** ninguno") for m in text)
    step = next(m for m in text if m.startswith(":green-badge[Vigente] **PersonalData skos:"))
    assert step.splitlines() == [
        ":green-badge[Vigente] **PersonalData skos:broadMatch dpv:PersonalData (usuario, versión "
        "2)**  ", "Decidida por el usuario (versión 2). Antes: exactMatch PersonalData (regla, "
        "versión 1). Pedido: PersonalData broadMatch PersonalData  ",
        "Revisa: PersonalData skos:exactMatch dpv:PersonalData (regla, versión 1)"]
    assert "Cambios del usuario (versión 2) · " in " ".join(c.value for c in at.caption)
    versions = next(d.value for d in at.dataframe if "Versión" in d.value.columns)
    assert list(versions.columns) == svc.VERSION_COLUMNS
    assert list(versions["Decidida por"]) == ["regla", "usuario"]
    assert iri in [c.value for c in at.caption]


def test_the_page_says_why_a_concept_has_no_correspondence(page, aligned):  # noqa: F811
    at = page()
    at.session_state["align_ws"] = aligned.id
    at.session_state["align_review"] = True              # the review (S10-T12)
    at.run()
    at.selectbox(key="align_concept").set_value(str(EX.Pizza)).run()
    assert not at.exception
    assert any(m.startswith("**Sin correspondencia en la versión 1.** La IA dijo que ningún "
                            "candidato del DPV corresponde.") for m in _card_text(at))
    assert "Ninguna versión le dio una correspondencia." in [c.value for c in at.caption]


def test_the_page_of_an_alignment_older_than_the_provenance(page, aligned):  # noqa: F811
    path = aligned.output_dir / ua.MANIFEST_FILE
    m = json.loads(path.read_text(encoding="utf-8"))
    m.pop("provenance")
    path.write_text(json.dumps(m), encoding="utf-8")
    at = page()
    at.session_state["align_ws"] = aligned.id
    at.session_state["align_review"] = True              # the review (S10-T12)
    at.run()
    at.selectbox(key="align_concept").set_value(str(EX.Consent)).run()
    assert not at.exception
    assert svc.NO_PROVENANCE in [c.value for c in at.caption]
