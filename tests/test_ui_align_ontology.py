"""S6-T08 + S7-T06 + S7-T08 + S7-T12: the "Alinear con el DPV" page (service + the page with
AppTest), in its three modes: "Mi ontología", "Una ley" and "Mi ontología y una ley".

AppTest cannot drive a file uploader, so the upload itself is tested through the service
(``align`` and ``run``) and the page is run with the workspace id already in the session,
exactly what the page stores after an upload."""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from rdflib import Graph

from app.services import align_ontology as svc
from src.core.emit import CORE_FILE
from src.workspace.store import WorkspaceError
from tests.test_alignment_propose import TTL, DeciderLLM, Embed, env, targets  # noqa: F401
from tests.test_law_evidence import LAW_E2E
from tests.test_law_profile import CORE_TTL, LAW, Embed as LawEmbed, LawLLM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(env):
    return env.cfg


@pytest.fixture
def aligned(env, targets, monkeypatch):
    """An upload already aligned (what the page has after the user drops a file)."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    seen = []
    ws_id = svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), progress=seen.append, base=env.base,
                      targets=targets, embed_fn=Embed(), llm=DeciderLLM(),
                      chroma_path=env.chroma, sleep_fn=lambda s: None)
    return ws_id, seen


# ------------------------------------------------------------------ service

@pytest.fixture
def law_cfg(env, monkeypatch):  # noqa: F811
    """The config of ``env`` plus the small OntoPriv-Core the "Una ley" mode imports."""
    out = Path(env.cfg["outputs"]["dir"])
    Graph().parse(data=CORE_TTL, format="turtle").serialize(out / CORE_FILE, format="xml")
    cfg = {**env.cfg, "extraction": {"pause_seconds": 0}}
    monkeypatch.setattr(svc, "workspace_cfg", lambda: cfg)
    return cfg


def _run(env, targets, mode, files, embed_fn, llm, seen=None):  # noqa: F811
    return svc.run(mode, files, progress=(seen if seen is not None else []).append,
                   base=env.base, targets=targets, embed_fn=embed_fn, llm=llm,
                   chroma_path=env.chroma, sleep_fn=lambda s: None)


@pytest.fixture
def law_aligned(env, targets, law_cfg):  # noqa: F811
    """A law already turned into an aligned ontology ("Una ley")."""
    seen = []
    ws_id = _run(env, targets, svc.MODE_LAW, {"law": ("Ley de prueba.txt", LAW.encode("utf-8"))},
                 LawEmbed(), LawLLM(), seen)
    return ws_id, seen


@pytest.fixture
def both_aligned(env, targets, monkeypatch):  # noqa: F811
    """An ontology aligned with the articles of a law to review ("Mi ontología y una ley")."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    seen = []
    files = {"ontology": ("Mi GDPR.ttl", TTL.encode("utf-8")),
             "law": ("ley.txt", LAW_E2E.encode("utf-8"))}
    return _run(env, targets, svc.MODE_BOTH, files, Embed(), DeciderLLM(), seen), seen


def test_align_runs_every_step_and_names_them(aligned):
    ws_id, seen = aligned
    assert ws_id.startswith("mi-gdpr-")
    assert seen == ["Heredar lo validado", "Buscar candidatos del DPV",
                    "Completar las etiquetas en español e inglés",
                    "Escribir la ontología alineada", "Preparar la descarga"]


def test_view_has_the_numbers_and_the_table(aligned):
    v = svc.view(aligned[0])
    assert v["file"] == "Mi GDPR.ttl" and v["concepts"] == 4 and v["version"] == 1
    assert v["metrics"] == {"Validadas en la base": 1, "Aceptadas automáticamente": 1,
                            "Sugeridas": 1, "Sin correspondencia": 1, "Sin propuesta": 0}
    assert (v["mappings"], v["validated_mappings"], v["accepted_mappings"], v["suggested"]) == (
        3, 2, 1, 1)
    assert v["rule"].startswith("confianza alta de la IA") and v["ai_note"] is None
    states = [r["Estado"] for r in v["rows"]]
    assert states == ["Validada en la base", "Validada en la base", "Aceptada automáticamente",
                      "Sugerida", "Sin correspondencia"]
    consent = [r for r in v["rows"] if r["Concepto"] == "Consent"]
    assert len(consent) == 2 and consent[0]["Decidido por"] == "base · mismo nombre · Consent"
    assert (consent[0]["Etiqueta (es)"], consent[0]["Etiqueta (en)"]) == ("Consentimiento",
                                                                          "Consent")
    assert list(consent[0])[:4] == ["Concepto", "Etiqueta (es)", "Etiqueta (en)", "Estado"]
    assert v["labels"] == ("Etiquetas en español e inglés: 0 conceptos ya las tenían; la IA "
                           "completó las de 4 (4 en español y 4 en inglés) en un módulo aparte, "
                           "sin cambiar tus etiquetas.")
    assert v["labels_note"] is None
    accepted = svc.filter_rows(v["rows"], "Aceptada automáticamente")[0]
    assert (accepted["Concepto"], accepted["Decidido por"], accepted["Confianza de la IA"]) == (
        "PersonalData", "regla", "alta")
    assert accepted["Justificación"] == "Decisión de prueba para PersonalData."
    assert svc.filter_rows(v["rows"], "Sugerida")[0]["Relación"] == "skos:relatedMatch"
    assert svc.filter_rows(v["rows"], "Todos") == v["rows"]
    assert (v["mode"], v["mode_label"]) == ("ontology", "Mi ontología")
    assert v["profile"] is None and v["evidence"] is None and v["extraction_note"] is None
    assert not v["core"] and "Artículos de la ley" not in v["rows"][0]


# ------------------------------------------------------------------ the three modes (S7-T12)

def test_the_modes_and_what_each_one_needs():
    assert svc.MODES == {"ontology": "Mi ontología", "law": "Una ley",
                         "ontology+law": "Mi ontología y una ley"}
    assert set(svc.MODE_HELP) == set(svc.MODE_WAIT) == set(svc.MODE_RUNNING) == set(svc.MODES)
    one = ("a.ttl", b"x")
    assert svc.ready("ontology", {"ontology": one}) and not svc.ready("ontology", {})
    assert svc.ready("law", {"law": one}) and not svc.ready("law", {"ontology": one})
    assert not svc.ready("ontology+law", {"ontology": one, "law": None})
    assert svc.ready("ontology+law", {"ontology": one, "law": one})
    assert svc.missing_text("law", {"law": None}) == svc.MODE_WAIT["law"]
    assert svc.missing_text("ontology+law", {"ontology": None, "law": None}) == \
        svc.MODE_WAIT["ontology+law"]
    assert svc.missing_text("ontology+law", {"ontology": one, "law": None}) == \
        "Falta tu ley: la alineación arranca cuando estén las dos."
    assert svc.missing_text("ontology+law", {"ontology": None, "law": one}) == \
        "Falta tu ontología: la alineación arranca cuando estén las dos."
    assert svc.LAW_TYPES == ["txt", "pdf"]


def test_run_checks_the_mode_and_the_files(cfg, monkeypatch):
    monkeypatch.setattr(svc, "workspace_cfg", lambda: cfg)
    with pytest.raises(WorkspaceError, match="Modo desconocido: otro"):
        svc.run("otro", {})
    with pytest.raises(WorkspaceError, match="^Falta subir tu ley.$"):
        svc.run("ontology+law", {"ontology": ("a.ttl", b"x")})
    with pytest.raises(WorkspaceError, match="^Falta subir tu ontología y tu ley.$"):
        svc.run("ontology+law", {})
    with pytest.raises(WorkspaceError, match="^Falta subir tu ontología.$"):
        svc.run("ontology", {"law": ("l.txt", b"x")})
    with pytest.raises(WorkspaceError, match="texto de ley debe ser .txt o .pdf"):
        svc.run("law", {"law": ("ley.docx", b"x")})


def test_a_law_becomes_an_aligned_ontology(law_aligned):
    ws_id, seen = law_aligned
    assert seen[:5] == ["Separar la ley en artículos", "Indexar los artículos de la ley",
                        "Extraer los conceptos de la ley con la IA",
                        "Aceptar los conceptos por regla",
                        "Armar el perfil de la ley sobre OntoPriv-Core"]
    v = svc.view(ws_id)
    assert (v["mode"], v["mode_label"], v["file"]) == ("law", "Una ley",
                                                       "Perfil de Ley de prueba.txt")
    assert v["profile"] == ("De Ley de prueba.txt: la IA extrajo 6 conceptos y la regla aceptó 6 "
                            "(5 clases y 1 propiedad). Con ellos se armó tu ontología: 2 "
                            "reutilizan conceptos de OntoPriv-Core, 1 se enlaza con él y 3 son "
                            "nuevos.")
    assert v["core"] and v["evidence"] is None and v["extraction_note"] is None
    assert v["concepts"] == 5 and "Artículos de la ley" not in v["rows"][0]
    names = svc.zip_names(svc.download(ws_id, "turtle")["data"])
    assert "ontopriv-core.ttl" in names and "ley-de-prueba-profile.ttl" in names


def test_an_ontology_with_a_law_brings_the_articles_to_review(both_aligned):
    ws_id, seen = both_aligned
    assert "Buscar en la ley los artículos de cada concepto" in seen
    v = svc.view(ws_id)
    assert (v["mode"], v["mode_label"], v["file"]) == ("ontology+law", "Mi ontología y una ley",
                                                       "Mi GDPR.ttl")
    assert v["evidence"] == ("Artículos de ley.txt para revisar: 4 de 4 conceptos tienen "
                             "artículos cercanos (12 enlaces a 3 artículos distintos). Están en "
                             "la columna «Artículos de la ley» de la tabla y en alignment.csv; no "
                             "entran a la ontología, porque la búsqueda sola no es lo bastante "
                             "precisa: revísalos.")
    assert v["profile"] is None and not v["core"]
    assert list(v["rows"][0])[-1] == "Artículos de la ley"
    personal = [r for r in v["rows"] if r["Concepto"] == "PersonalData"][0]
    assert personal["Artículos de la ley"].startswith("art. 1 (0,00); ")
    assert v["metrics"] == svc.view(ws_id)["metrics"]
    names = svc.zip_names(svc.download(ws_id, "turtle")["data"])
    assert not any("evidence" in n for n in names) and "alignment.csv" in names


def test_profile_caption_and_extraction_note():
    assert svc.profile_caption(None) is None and svc.extraction_note(None) is None
    profile = {"law": "gdpr-en.pdf", "classes": 74, "properties": 48, "reused": 13, "linked": 1,
               "new": 108, "law_concepts": {"concepts": 125, "accepted": 122, "repeated": 1,
                                            "rejected": 2}}
    assert svc.profile_caption(profile) == (
        "De gdpr-en.pdf: la IA extrajo 125 conceptos y la regla aceptó 122 (74 clases y 48 "
        "propiedades); 1 repetido y 2 rechazados no entraron. Con ellos se armó tu ontología: "
        "13 reutilizan conceptos de OntoPriv-Core, 1 se enlaza con él y 108 son nuevos.")
    only = {**profile, "law_concepts": {**profile["law_concepts"], "repeated": 0}}
    assert "; 2 rechazados no entraron." in svc.profile_caption(only)
    one = {**profile, "classes": 1, "properties": 0, "reused": 1, "linked": 0, "new": 1,
           "law_concepts": {"concepts": 1, "accepted": 1, "repeated": 1, "rejected": 0}}
    assert svc.profile_caption(one) == (
        "De gdpr-en.pdf: la IA extrajo 1 concepto y la regla aceptó 1 (1 clase y 0 propiedades)"
        "; 1 repetido no entró. Con ellos se armó tu ontología: 1 reutiliza conceptos de "
        "OntoPriv-Core, 0 se enlazan con él y 1 es nuevo.")
    assert svc.evidence_caption(None) is None
    single = svc.evidence_caption({"law": "l.txt", "with_evidence": 1, "concepts": 1,
                                   "links": 1, "articles_cited": 1})
    assert single.startswith("Artículos de l.txt para revisar: 1 de 1 concepto tiene artículos "
                             "cercanos (1 enlace a 1 artículo distinto).")
    assert svc.evidence_caption({"law": "l.txt", "with_evidence": 0, "concepts": 4, "links": 0,
                                 "articles_cited": 0}) == (
        "Ningún concepto tiene artículos de l.txt lo bastante cercanos para revisarlos.")
    done = {"extraction": {"complete": True}, "ai": {"available": True}}
    assert svc.extraction_note(done) is None
    left = {"extraction": {"complete": False, "articles_processed": 64, "articles_total": 99,
                           "pending_articles": list(range(35))}, "ai": {"available": True}}
    what = "la IA leyó 64 de 99 artículos de la ley (faltan 35)"
    assert svc.extraction_note(left) == (f"La IA falló en algunos lotes: {what}. Vuelve a subir "
                                         "el archivo para intentarlo de nuevo.")
    off = svc.extraction_note({**left, "ai": {"available": False}})
    assert off.startswith(f"La IA no estuvo disponible: {what}.")
    quota = svc.extraction_note({**left, "ai": {"available": True, "quota_exhausted": True}})
    assert quota.startswith(f"Se agotó la cuota diaria de la IA: {what}.") and "mañana" in quota
    limit = svc.extraction_note({**left, "ai": {"available": True, "call_limit_reached": True}})
    assert limit.startswith(f"Se llegó al límite de llamadas a la IA de esta subida: {what}.")


def test_the_view_warns_when_the_law_was_not_read_whole(law_aligned):
    from src.core.law_concepts import RESULT_FILE
    from src.workspace.store import load_workspace
    ws = load_workspace(law_aligned[0], cfg=svc.workspace_cfg())
    path = ws.work_dir / RESULT_FILE
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["extraction"] = {**saved["extraction"], "complete": False, "articles_processed": 3,
                           "pending_articles": [4]}
    saved["ai"] = {**saved["ai"], "call_limit_reached": True}
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert svc.view(law_aligned[0])["extraction_note"].startswith(
        "Se llegó al límite de llamadas a la IA de esta subida: la IA leyó 3 de 4 artículos")


def test_download_in_each_format_without_recomputing(aligned):
    ws_id, _ = aligned
    for key, label in svc.FORMATS:
        pkg = svc.download(ws_id, key)
        assert pkg["label"] == label and pkg["file_name"] == f"mi-gdpr-alineada-{key}.zip"
        names = svc.zip_names(pkg["data"])
        assert pkg["main_file"] in names and "alignment.csv" in names
        assert (pkg["note"] is not None) == (key == "jsonld")


def test_friendly_errors(cfg, monkeypatch):
    monkeypatch.setattr(svc, "workspace_cfg", lambda: cfg)
    with pytest.raises(WorkspaceError) as exc:
        svc.align("notas.md", b"hola")
    assert svc.error_message(exc.value).startswith("Formato de ontología no soportado")
    assert "base construida" in svc.error_message(FileNotFoundError("x.json"))
    assert svc.error_message(RuntimeError("boom")) == \
        "No se pudo alinear la ontología: RuntimeError: boom"
    with pytest.raises(WorkspaceError, match="No existe"):
        svc.view("no-existe-00000000")


def test_ai_note_only_when_concepts_were_left_without_the_ai():
    assert svc._ai_note({"pending_ai": 0, "ai": {"available": False}}) is None
    off = svc._ai_note({"pending_ai": 3, "ai": {"available": False}, "threshold_without_ai": 0.75})
    assert "no estuvo disponible: a 3 conceptos" in off and "puntaje >= 0.75" in off
    quota = svc._ai_note({"pending_ai": 2, "ai": {"available": True, "quota_exhausted": True}})
    assert "cuota diaria" in quota and "lo ya decidido no se repite" in quota
    limit = svc._ai_note({"pending_ai": 2, "ai": {"available": True, "call_limit_reached": True}})
    assert "límite de llamadas" in limit


def test_without_ai_the_page_says_what_happened(env, targets, monkeypatch):
    offline = {**env.cfg, "auto": {**env.cfg["auto"], "use_llm": False}}
    monkeypatch.setattr(svc, "workspace_cfg", lambda: offline)
    v = svc.view(svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), base=env.base, targets=targets,
                           embed_fn=Embed(), chroma_path=env.chroma))
    assert v["metrics"]["Aceptadas automáticamente"] == 2 and v["metrics"]["Sin propuesta"] == 1
    assert "La IA no estuvo disponible: a 3 conceptos" in v["ai_note"]
    assert {r["Decidido por"] for r in v["rows"] if r["Estado"] == "Aceptada automáticamente"} \
        == {"regla sin IA"}
    assert "Sin decidir" in {r["Estado"] for r in v["rows"]}
    assert v["labels"] == "Etiquetas en español e inglés: 0 conceptos ya las tenían."
    assert v["labels_note"].startswith("La IA no estuvo disponible: 4 conceptos quedaron sin "
                                       "sus etiquetas en español o inglés.")
    assert {r["Etiqueta (es)"] for r in v["rows"]} == {""}


def test_labels_caption_and_note():
    assert svc.labels_caption(None) is None and svc._labels_note(None) is None
    done = {"complete": 3, "translated": 0, "pending": 0, "added": {"es": 0, "en": 0}}
    assert svc.labels_caption(done) == "Etiquetas en español e inglés: 3 conceptos ya las tenían."
    assert svc._labels_note(done) is None
    left = {"pending": 2, "ai": {"available": True}}
    quota = svc._labels_note({**left, "ai": {"available": True, "quota_exhausted": True}})
    assert quota.startswith("Se agotó la cuota diaria de la IA: 2 conceptos quedaron")
    assert "mañana (lo ya traducido no se repite)" in quota
    limit = svc._labels_note({**left, "ai": {"available": True, "call_limit_reached": True}})
    assert limit.startswith("Se llegó al límite de llamadas a la IA de esta subida")
    assert svc._labels_note(left).startswith("La IA no devolvió etiquetas válidas: 2 conceptos")


def test_a_workspace_of_v060_asks_to_upload_again(aligned):
    from src.alignment import user_alignment as ua
    from src.workspace.store import load_workspace
    ws = load_workspace(aligned[0], cfg=svc.workspace_cfg())
    path = ws.output_dir / ua.MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("decider")                                     # what v0.6.0 wrote
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="vuelve a subirlo"):
        svc.view(aligned[0])


def test_page_is_first_in_the_navigation():
    text = (ROOT / "app/app.py").read_text(encoding="utf-8")
    assert '"Tu ontología": [' in text and '"Construcción de la base": [' in text
    assert text.index("views/align_ontology.py") < text.index("views/inputs.py")
    assert 'title="Alinear con el DPV"' in text


# ------------------------------------------------------------------ the page (AppTest)

@pytest.fixture
def page(cfg, monkeypatch):  # noqa: F811
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    app_dir = str(ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    view_svc = importlib.import_module("services.align_ontology")   # what the page imports
    monkeypatch.setattr(view_svc, "workspace_cfg", lambda: cfg)
    return lambda: AppTest.from_file(str(ROOT / "app/views/align_ontology.py"),
                                     default_timeout=30)


def _uploaders(at) -> list[str]:
    return [e.proto.label for e in at.get("file_uploader")]


def _uploader_ids(at) -> list[str]:
    return [e.proto.id for e in at.get("file_uploader")]


def test_page_without_upload_explains_what_to_do(page):
    at = page()
    at.run()
    assert not at.exception
    assert any("Sube un archivo para empezar" in i.value for i in at.info)
    assert not at.download_button
    mode = at.radio(key="align_mode")
    assert mode.options == ["Mi ontología", "Una ley", "Mi ontología y una ley"]
    assert mode.value == "ontology"
    assert _uploaders(at) == ["Tu ontología: RDF/XML (.rdf, .owl), Turtle (.ttl) o JSON-LD "
                              "(.jsonld)"]


def test_each_mode_asks_for_its_files(page):
    at = page()
    at.run()
    at.radio(key="align_mode").set_value("law").run()
    assert not at.exception
    assert _uploaders(at) == ["Tu ley: texto (.txt) o PDF"]
    assert [i.value for i in at.info] == [svc.MODE_WAIT["law"]]
    at.radio(key="align_mode").set_value("ontology+law").run()
    assert _uploaders(at) == ["Tu ontología: RDF/XML (.rdf, .owl), Turtle (.ttl) o JSON-LD "
                              "(.jsonld)", "Tu ley: texto (.txt) o PDF"]
    assert [i.value for i in at.info] == [svc.MODE_WAIT["ontology+law"]]


def test_changing_the_mode_gives_new_uploaders(page):
    """Streamlit keeps the file of an uploader that was hidden for one run; with the same key,
    coming back to a mode would start a long run (and AI calls) with a file the person no
    longer sees."""
    at = page()
    at.run()
    first = _uploader_ids(at)
    at.run()
    assert _uploader_ids(at) == first                        # same mode: same uploader
    at.radio(key="align_mode").set_value("law").run()
    at.radio(key="align_mode").set_value("ontology").run()
    assert len(_uploader_ids(at)) == 1 and _uploader_ids(at) != first
    assert at.session_state["align_round"] == 3


def test_page_shows_the_result_and_the_download(page, aligned):
    at = page()
    at.session_state["align_ws"] = aligned[0]
    at.run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Validadas en la base", "Aceptadas automáticamente",
                                            "Sugeridas", "Sin correspondencia", "Sin propuesta"]
    assert [m.value for m in at.metric] == ["1", "1", "1", "1", "0"]
    assert all(m.help for m in at.metric)                         # every number explained
    assert any("3 correspondencias en la ontología" in c.value for c in at.caption)
    assert any(c.value.startswith("Regla: confianza alta") for c in at.caption)
    assert any(c.value.startswith("Etiquetas en español e inglés: 0 conceptos ya las tenían; "
                                  "la IA completó las de 4") for c in at.caption)
    assert not at.warning
    assert at.radio(key="align_fmt").value == "Turtle"
    assert at.download_button[0].label == "Descargar ontología alineada (Turtle, .zip)"
    assert any("mi-gdpr-dpv-alignment.ttl" in c.value for c in at.caption)
    at.radio(key="align_fmt").set_value("JSON-LD").run()
    assert at.download_button[0].label == "Descargar ontología alineada (JSON-LD, .zip)"
    assert any("Protégé 5.6 no abre JSON-LD" in i.value for i in at.info)
    at.selectbox(key="align_filter").set_value("Validada en la base").run()
    assert len(at.dataframe[0].value) == 2


def test_page_warns_when_the_ai_left_labels_pending(page, env, targets, monkeypatch):
    offline = {**env.cfg, "auto": {**env.cfg["auto"], "use_llm": False}}
    monkeypatch.setattr(svc, "workspace_cfg", lambda: offline)
    ws_id = svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), base=env.base, targets=targets,
                      embed_fn=Embed(), chroma_path=env.chroma)
    at = page()
    at.session_state["align_ws"] = ws_id
    at.run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert len(warnings) == 2 and warnings[0].startswith("La IA no estuvo disponible: a 3")
    assert warnings[1].startswith("La IA no estuvo disponible: 4 conceptos quedaron sin sus "
                                  "etiquetas")


def test_page_shows_what_was_built_from_a_law(page, law_aligned):
    at = page()
    at.session_state["align_ws"] = law_aligned[0]
    at.run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any(c.startswith("Una ley · 5 conceptos") for c in captions)
    assert any(c.startswith("De Ley de prueba.txt: la IA extrajo 6 conceptos") for c in captions)
    assert any("El paquete incluye OntoPriv-Core, que tu ontología importa." in c
               for c in captions)
    assert not at.warning
    from src.core.law_concepts import RESULT_FILE
    from src.workspace.store import load_workspace
    path = load_workspace(law_aligned[0], cfg=svc.workspace_cfg()).work_dir / RESULT_FILE
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["extraction"] = {**saved["extraction"], "complete": False, "articles_processed": 3,
                           "pending_articles": [4]}
    saved["ai"] = {**saved["ai"], "quota_exhausted": True}
    path.write_text(json.dumps(saved), encoding="utf-8")
    at.run()
    assert [w.value for w in at.warning][0].startswith(
        "Se agotó la cuota diaria de la IA: la IA leyó 3 de 4 artículos de la ley")


def test_page_shows_the_articles_to_review(page, both_aligned):
    at = page()
    at.session_state["align_ws"] = both_aligned[0]
    at.run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any(c.startswith("Mi ontología y una ley · 4 conceptos") for c in captions)
    assert any(c.startswith("Artículos de ley.txt para revisar: 4 de 4") for c in captions)
    assert not any("OntoPriv-Core" in c for c in captions)
    assert "Artículos de la ley" in list(at.dataframe[0].value.columns)


def test_home_presents_the_three_modes():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30)    # home: default page
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    for label in svc.MODES.values():
        assert f"**{label}**" in text
    assert "Alinear con el DPV" in text


class _Upload:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


@pytest.fixture
def live(page, env, targets, monkeypatch, tmp_path):  # noqa: F811
    """The page with uploads it can see (AppTest cannot drive a file uploader), the small base,
    a fake embedder and a fake LLM: what happens after the person drops the files."""
    import streamlit
    import src.ai.llm_client as client
    from src.alignment import inherit as inh, propose as prop
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.ai.rag.vector_store._PATH", str(tmp_path / "chroma-page"))
    monkeypatch.setattr(client, "LLMClient", DeciderLLM)
    monkeypatch.setattr(inh, "load_base", lambda **kw: env.base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)
    uploads = {}

    def uploader(label, *args, **kwargs):
        found = uploads.get("law" if label.startswith("Tu ley") else "ontology")
        return _Upload(*found) if found else None

    monkeypatch.setattr(streamlit, "file_uploader", uploader)
    streamlit.cache_resource.clear()
    yield page, uploads
    streamlit.cache_resource.clear()


def test_the_page_starts_when_the_files_of_the_mode_are_there(live):
    page, uploads = live
    at = page()
    at.run()
    at.radio(key="align_mode").set_value("ontology+law").run()
    uploads["ontology"] = ("Mi GDPR.ttl", TTL.encode("utf-8"))
    at.run()
    assert [i.value for i in at.info] == [
        "Falta tu ley: la alineación arranca cuando estén las dos."]
    assert not at.download_button
    uploads["law"] = ("ley.txt", LAW_E2E.encode("utf-8"))
    at.run()
    assert not at.exception and not at.error
    assert any(c.value.startswith("Mi ontología y una ley · 4 conceptos") for c in at.caption)
    assert any(c.value.startswith("Artículos de ley.txt para revisar: 4 de 4") for c in at.caption)
    assert at.download_button[0].label == "Descargar ontología alineada (Turtle, .zip)"
    assert at.session_state["align_ws"].startswith("mi-gdpr-")


def test_the_page_says_why_it_could_not_finish(live):
    page, uploads = live
    at = page()
    at.run()
    at.radio(key="align_mode").set_value("law").run()
    uploads["law"] = ("ley.docx", b"texto")
    at.run()
    assert not at.exception and not at.download_button
    assert [e.value for e in at.error] == ["El texto de ley debe ser .txt o .pdf (llegó '.docx')."]


def test_page_reports_a_broken_workspace(page):
    at = page()
    at.session_state["align_ws"] = "no-existe-00000000"
    at.run()
    assert not at.exception
    assert any("No existe el espacio de trabajo" in e.value for e in at.error)


def test_upload_limit_matches_the_workspace_limit():
    import tomllib
    from src.workspace.store import workspace_settings
    toml = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    assert toml["server"]["maxUploadSize"] == workspace_settings()["max_upload_mb"]