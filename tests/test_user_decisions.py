"""S10-T03: what the person chooses after testing an upload. A fixed rule checks each change,
the valid ones go to an append-only log and the alignment is written again as a new version;
the previous one stays in history/ with its four zips. The uploaded ontology, the rule's
decisions and the base never change. The LLM is a fake (DeciderLLM): no test calls Gemini."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
import zipfile

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDFS, SKOS

from src.alignment import decisions as dl
from src.alignment import inherit as inh
from src.alignment import propose as prop
from src.alignment import user_alignment as ua
from src.alignment import user_decisions as ud
from src.transform.package import README_FILE
from src.validation import ai_auditor as au
from src.validation import upload_validation as uv
from src.validation.shape_inspector import GREEN, RED
from src.workspace.store import load_workspace
from tests.test_alignment_propose import (TTL_PROPERTY, DeciderLLM, Embed, env,  # noqa: F401
                                          targets)
from tests.test_validation_upload import _align, _real_dpv

EX = Namespace("http://example.org/gdpr#")
DPV = Namespace("https://w3id.org/dpv#")


@pytest.fixture
def aligned(env, targets):  # noqa: F811
    """'Mi GDPR.ttl' aligned (version 1): Consent inherits exactMatch Consent and relatedMatch
    ConsentRecord, the rule accepts PersonalData exactMatch PersonalData, DataController is a
    suggestion and Pizza has no correspondence."""
    w, _, _ = _align(_real_dpv(env), targets)
    return w


def _changes(*texts):
    return [ud.parse_change(t) for t in texts]


def _check(w, targets, *texts):  # noqa: F811
    return ud.check(w, _changes(*texts), targets=targets)


def _apply(w, env, targets, *texts):  # noqa: F811
    return ud.apply(w, _changes(*texts), cfg=env.cfg, targets=targets, llm=DeciderLLM(),
                    sleep_fn=lambda s: None)


def _module(w) -> Graph:
    return Graph().parse(next(w.output_dir.glob("*-dpv-alignment.ttl")), format="turtle")


def _axiom(g: Graph, s, p, o) -> dict | None:
    """The annotations of the owl:Axiom of (s, p, o), as {predicate: value}."""
    for node in g.subjects(OWL.annotatedSource, s):
        if (node, OWL.annotatedProperty, p) in g and (node, OWL.annotatedTarget, o) in g:
            return {pred: obj for pred, obj in g.predicate_objects(node)}
    return None


def _rows(w) -> list[dict]:
    with open(w.output_dir / ua.CSV_FILE, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _md5(paths) -> dict:
    return {p.as_posix(): hashlib.md5(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _validation(w) -> dict:
    return json.loads((w.output_dir / uv.RESULT_FILE).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ what the person writes

def test_the_three_ways_of_writing_a_change():
    c = ud.parse_change("EmploymentContext broadMatch ProcessingContext")
    assert (c.concept, c.action, c.relation, c.dpv, c.replaces) == (
        "EmploymentContext", ud.SET, "broadMatch", "ProcessingContext", None)
    c = ud.parse_change("Consent closeMatch PersonalData en vez de ConsentRecord")
    assert (c.dpv, c.replaces) == ("PersonalData", "ConsentRecord")
    assert c.text() == "Consent closeMatch PersonalData en vez de ConsentRecord"
    assert ud.parse_change("Pizza quitar").action == ud.REMOVE
    assert ud.parse_change("Pizza ninguna").dpv is None
    c = ud.parse_change("Consent Quitar ConsentRecord")
    assert (c.action, c.dpv, c.text()) == (ud.REMOVE, "ConsentRecord",
                                           "Consent quitar ConsentRecord")
    for bad in ("", "Pizza", "Pizza broadMatch", "A broadMatch B en lugar de C",
                "A b c d e"):
        with pytest.raises(ValueError, match="No entiendo el cambio"):
            ud.parse_change(bad)


# ------------------------------------------------------------------ the fixed rule

def test_the_fixed_rule_says_why_in_spanish(aligned, targets):  # noqa: F811
    checked = _check(aligned, targets,
                     "Foo exactMatch PersonalData",
                     "Pizza exactMatch NoExiste",
                     "Pizza sameAs PersonalData",
                     "PersonalData exactMatch PersonalData",
                     "Consent broadMatch PersonalData",
                     "Pizza quitar",
                     "PersonalData quitar DataController",
                     "Consent closeMatch PersonalData en vez de DataController",
                     "DataController exactMatch DataController",
                     "DataController exactMatch NoExiste",
                     "DataController closeMatch DataController")
    assert [c.reason for c in checked] == [
        "No encontré el concepto «Foo» en tu ontología.",
        "«NoExiste» no es un término del DPV 2.3.",
        "«sameAs» no es una relación SKOS: usa exactMatch, closeMatch, broadMatch, narrowMatch "
        "o relatedMatch.",
        "«PersonalData» ya es exactMatch de «PersonalData»: no cambia nada.",
        "«Consent» tiene 2 correspondencias (Consent y ConsentRecord): di cuál cambias con «en "
        "vez de».",
        "«Pizza» no tiene correspondencias que quitar.",
        "«PersonalData» no está alineado con «DataController».",
        "«Consent» no está alineado con «DataController».",
        "",
        "«NoExiste» no es un término del DPV 2.3.",
        "«DataController» ya tiene otro cambio en esta lista: uno por concepto."]
    ok = checked[8]
    assert ok.ok and ok.before == [] and ok.relation == "skos:exactMatch"
    assert ok.after == [{"dpv_iri": str(DPV.DataController), "dpv_name": "DataController",
                         "relation": "skos:exactMatch", "decided_by": ud.BY_USER,
                         "version": None}]
    assert not ud.read(aligned) and not (aligned.path / ud.HISTORY_DIR).exists()  # only checks


def test_two_valid_changes_on_two_concepts(aligned, targets):  # noqa: F811
    first, second = _check(aligned, targets, "PersonalData broadMatch PersonalData",
                           "Consent broadMatch PersonalData en vez de ConsentRecord")
    assert first.ok and first.replaces is None              # same term: only the relation
    assert first.before[0]["decided_by"] == prop.BY_RULE and first.before[0]["version"] == 1
    assert second.ok and second.replaces == str(DPV.ConsentRecord)
    assert [m["dpv_name"] for m in second.after] == ["Consent", "PersonalData"]
    assert ud.render_checked([first, second]).splitlines() == [
        "Cambios pedidos (los revisa una regla fija)",
        "  [válido] PersonalData: exactMatch PersonalData (regla, versión 1) -> broadMatch "
        "PersonalData (usuario)",
        "  [válido] Consent: exactMatch Consent (base, versión 1); relatedMatch ConsentRecord "
        "(base, versión 1) -> broadMatch PersonalData (usuario)"]


def test_a_property_only_with_properties_and_an_individual_never(env, targets,  # noqa: F811
                                                                monkeypatch):
    w, _, _ = ua.align_upload("Con propiedad.ttl", TTL_PROPERTY.encode("utf-8"), fmt="turtle",
                              embed_fn=Embed(), llm=DeciderLLM(), base=_real_dpv(env).base,
                              targets=targets, cfg=env.cfg, chroma_path=env.chroma,
                              sleep_fn=lambda s: None)
    saved = inh.load_result(w)                     # what the inheritance says of an individual
    individual = {"key": str(EX.consent001), "name": "consent001", "reason": "kind"}
    monkeypatch.setattr(inh, "load_result", lambda ws: {
        **saved, "not_inherited": [*saved["not_inherited"], individual]})
    checked = _check(w, targets, "hasRecipient closeMatch PersonalData",
                     "Pizza closeMatch hasRecipient", "consent001 exactMatch Consent",
                     "hasRecipient closeMatch hasEntity")
    assert [c.reason for c in checked[:3]] == [
        "«hasRecipient» es una propiedad y «PersonalData» no: una propiedad solo se alinea "
        "con propiedades del DPV.",
        "«Pizza» es una clase y «hasRecipient» es una propiedad del DPV: una clase solo se "
        "alinea con clases del DPV.",
        "«consent001» es un individuo: no se alinea con el DPV."]
    assert checked[3].ok


# ------------------------------------------------------------------ a new version

def test_a_change_makes_version_2_and_nothing_else_changes(aligned, env, targets,  # noqa: F811
                                                            tmp_path):
    w = aligned
    system = [p for p in tmp_path.rglob("*") if w.path.parent not in p.parents
              and env.chroma not in p.parents]
    kept = list(w.input_dir.rglob("*")) + [w.work_dir / n for n in (
        "inherited.json", "alignment-decisions.json", "labels.json", "ai-audit.json")]
    before, decisions = _md5(system + kept), prop.load_result(w)
    assert len(_md5(kept)) == 5                               # input + the four results

    result = _apply(w, env, targets, "PersonalData broadMatch PersonalData")

    assert result["applied"] and result["version"] == 2
    assert _md5(system + kept) == before and prop.load_result(w) == decisions
    g = _module(w)
    assert (EX.PersonalData, SKOS.exactMatch, DPV.PersonalData) not in g
    axiom = _axiom(g, EX.PersonalData, SKOS.broadMatch, DPV.PersonalData)
    by = URIRef("http://example.org/gdpr-dpv-alignment#decidedBy")
    assert axiom[by] == Literal("usuario", lang="es") and axiom[OWL.versionInfo] == Literal("2")
    assert "Antes: exactMatch PersonalData (regla, versión 1)" in str(axiom[RDFS.comment])
    assert _axiom(g, EX.Consent, SKOS.exactMatch, DPV.Consent)[by] == Literal("base", lang="es")
    assert (URIRef("http://example.org/gdpr-dpv-alignment"), OWL.versionInfo,
            Literal("2")) in g

    personal = [r for r in _rows(w) if r["concept_name"] == "PersonalData"]
    assert [(r["status"], r["decided_by"], r["reviewer"], r["relation"], r["version"])
            for r in personal] == [(ua.USER_ALIGNED, ud.BY_USER, "usuario", "skos:broadMatch",
                                    "2")]
    manifest = ua.load_manifest(w)
    assert manifest["alignment_version"] == 2 and manifest["summary"]["user_mappings"] == 1
    assert manifest["user_decisions"] == {"concepts": 1, "log": "work/user-decisions.jsonl"}
    assert [(v["version"], v["folder"]) for v in manifest["versions"]] == [(1, "history/v1")]
    assert load_workspace(w.id, cfg=env.cfg).manifest["alignment_version"] == 2

    [record] = ud.read(w)                                     # the log of the person
    assert (record.decision, record.reviewer, record.relation, record.source_id) == (
        dl.DECISION_APPROVED, "usuario", "skos:broadMatch", w.source_id)
    assert record.snapshot["version"] == 2 and record.snapshot["after"][0]["version"] == 2
    assert record.snapshot["before"][0]["relation"] == "skos:exactMatch"

    old = w.path / "history" / "v1"                           # version 1 can still be downloaded
    assert sorted(p.name for p in old.glob("*.zip")) == [
        f"mi-gdpr-alineada-{k}.zip" for k in ("jsonld", "owl", "rdfxml", "turtle")]
    assert json.loads((old / ua.MANIFEST_FILE).read_text(encoding="utf-8"))[
        "alignment_version"] == 1
    with zipfile.ZipFile(old / "mi-gdpr-alineada-turtle.zip") as z:
        name = next(n for n in z.namelist() if n.endswith("-dpv-alignment.ttl"))
        assert (EX.PersonalData, SKOS.exactMatch, DPV.PersonalData) in Graph().parse(
            data=z.read(name), format="turtle")
    assert [p.name for p in w.output_dir.glob("*.zip")] == ["mi-gdpr-alineada-v2-turtle.zip"]
    assert result["zip"].name == "mi-gdpr-alineada-v2-turtle.zip"
    with zipfile.ZipFile(result["zip"]) as z:
        readme = next(z.read(n).decode("utf-8") for n in z.namelist()
                      if n.endswith(README_FILE))
    assert "· versión 2" in readme
    assert "      1 decidida por el usuario después de probar (versión 2)." in readme

    validation = _validation(w)
    assert validation["inspectors"][uv.SHAPES]["status"] != RED      # "usuario" is allowed
    assert validation["inspectors"][uv.SHAPES]["new"] == 0
    audit = validation["inspectors"][uv.AUDIT]                        # theirs is not reviewed
    assert audit["summary"]["accepted"] == 0 and audit["summary"]["by_user"] == 1
    assert audit["status"] == GREEN
    console = ua.render_console(w, result["manifest"], result["zip"])
    assert "    decididas por el usuario: 1 correspondencia en 1 concepto\n" in console
    assert "    aceptadas automáticamente: 0 (0 sin tipo)\n" in console
    assert ("la regla no aceptó ninguna correspondencia que revisar (1 concepto decidido por "
            "el usuario: no se revisa de nuevo)") in console
    assert "Módulo de alineación: output/mi-gdpr-dpv-alignment.ttl" in console
    assert "versión 2)" in console


def test_more_changes_give_more_versions_and_a_rerun_keeps_them(aligned, env,  # noqa: F811
                                                                 targets):
    w = aligned
    first = _apply(w, env, targets, "PersonalData broadMatch PersonalData",
                   "Consent broadMatch PersonalData en vez de ConsentRecord")
    assert first["version"] == 2
    assert [(d.concept_key, d.decision) for d in ud.read(w)] == [
        (str(EX.PersonalData), dl.DECISION_APPROVED),
        (str(EX.Consent), dl.DECISION_REJECTED), (str(EX.Consent), dl.DECISION_APPROVED)]
    g = _module(w)
    assert (EX.Consent, SKOS.relatedMatch, DPV.ConsentRecord) not in g
    assert (EX.Consent, SKOS.exactMatch, DPV.Consent) in g             # the base's one stays
    assert (EX.Consent, SKOS.broadMatch, DPV.PersonalData) in g
    consent = sorted((r["status"], r["dpv_name"]) for r in _rows(w)
                     if r["concept_name"] == "Consent")
    assert consent == [(ua.INHERITED_ALIGNED, "Consent"), (ua.USER_ALIGNED, "PersonalData")]

    assert _apply(w, env, targets, "Consent quitar Consent")["version"] == 3
    assert (EX.Consent, SKOS.exactMatch, DPV.Consent) not in _module(w)
    last = _apply(w, env, targets, "Consent quitar")
    assert last["version"] == 4 and ud.read(w)[-1].decision == dl.DECISION_NO_MATCH
    g = _module(w)
    assert not set(g.predicates(EX.Consent, None)) & {
        SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch, SKOS.relatedMatch}
    axiom = _axiom(g, EX.PersonalData, SKOS.broadMatch, DPV.PersonalData)
    assert axiom[OWL.versionInfo] == Literal("2")              # it was decided in version 2
    [row] = [r for r in _rows(w) if r["concept_name"] == "Consent"]
    assert (row["status"], row["version"]) == (ua.USER_NO_MATCH, "4")
    assert row["justification"].startswith("Quitada por el usuario el ")
    assert row["justification"].endswith("Antes: broadMatch PersonalData (usuario, versión 2).")
    assert sorted(p.name for p in (w.path / "history").iterdir()) == ["v1", "v2", "v3"]
    assert sorted(p.name for p in (w.path / "history" / "v3").glob("*.zip"))[-1] == (
        "mi-gdpr-alineada-v3-turtle.zip")
    assert [v["version"] for v in ua.load_manifest(w)["versions"]] == [1, 2, 3]
    console = ua.render_console(w, last["manifest"], last["zip"])
    assert "    sin correspondencia: 2\n" in console                   # Pizza and Consent

    llm = DeciderLLM()                                         # a normal run of the workspace
    w, manifest, path = ua.align_workspace(w, fmt="turtle", embed_fn=Embed(), llm=llm,
                                           base=env.base, targets=targets, cfg=env.cfg,
                                           chroma_path=env.chroma, sleep_fn=lambda s: None)
    assert llm.calls == 0 and manifest["alignment_version"] == 4
    assert path.name == "mi-gdpr-alineada-v4-turtle.zip"
    assert (EX.PersonalData, SKOS.broadMatch, DPV.PersonalData) in _module(w)


def test_without_a_valid_change_nothing_is_written(aligned, env, targets):  # noqa: F811
    w = aligned
    before = _md5(w.path.rglob("*"))
    result = _apply(w, env, targets, "Foo exactMatch PersonalData", "Pizza quitar")
    assert not result["applied"] and result["version"] == 1
    assert _md5(w.path.rglob("*")) == before
    assert not ud.log_path(w).exists() and not (w.path / ud.HISTORY_DIR).exists()


def test_before_aligning_there_is_nothing_to_change(env, targets):  # noqa: F811
    from src.workspace.store import open_ontology
    from tests.test_alignment_propose import TTL
    w = open_ontology("sin alinear.ttl", TTL.encode("utf-8"), cfg=env.cfg)
    with pytest.raises(ValueError, match="Primero hay que alinear"):
        _check(w, targets, "Pizza exactMatch PersonalData")


# ------------------------------------------------------------------ the AI reviewer

def test_the_reviewer_does_not_review_again_what_the_person_decided():
    result = {"entries": [{"key": "a", "verdict": au.COHERENT},
                          {"key": "b", "verdict": au.INCOHERENT},
                          {"key": "c", "verdict": None}],
              "summary": {"accepted": 3}}
    assert au.without(result, set()) is result
    s = au.without(result, {"b"})["summary"]
    assert (s["accepted"], s[au.INCOHERENT], s[au.COHERENT], s["pending"], s["by_user"]) == (
        2, 0, 1, 1, 1)
    assert uv.audit_status({"summary": s}) == GREEN
    block = {"status": GREEN, "summary": s, "prompt": {"name": "audit", "version": 1},
             "ai": {}}
    assert uv.inspector_text(uv.AUDIT, block).endswith(
        "por la regla, 1 sin revisar; solo explica, no cambia lo aceptado (1 concepto decidido "
        "por el usuario: no se revisa de nuevo)")
    from app.services.align_ontology import card_text
    assert card_text(uv.AUDIT, block).endswith(" · sin revisar: 1 · del usuario: 1")


# ------------------------------------------------------------------ web

def test_the_page_shows_version_2_and_what_the_person_decided(aligned, env, targets,  # noqa: F811
                                                               monkeypatch):
    from app.services import align_ontology as svc
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    before = svc.view(aligned.id)
    assert svc.USER_METRIC not in before["metrics"]                  # nothing decided yet
    _apply(aligned, env, targets, "PersonalData broadMatch PersonalData")
    v = svc.view(aligned.id)
    assert v["version"] == 2 and v["metrics"] == {
        "Validadas en la base": 1, "Aceptadas automáticamente": 0, svc.USER_METRIC: 1,
        "Sugeridas": 1, "Sin correspondencia": 1, "Sin propuesta": 0}
    assert svc.summary_captions(v)[0] == (
        "Mi ontología · 4 conceptos · 3 correspondencias en la ontología (2 de la base, 0 "
        "aceptadas por la regla y 1 del usuario) · 1 sugeridas fuera de la ontología · "
        "versión 2")
    [row] = [r for r in v["rows"] if r["Concepto"] == "PersonalData"]
    assert (row["Estado"], row["Decidido por"], row["Relación"]) == (
        "Decidida por el usuario", "usuario", "skos:broadMatch")
    assert [d["file_name"] for d in svc.download_options(aligned.id)] == [
        f"mi-gdpr-alineada-v2-{k}.zip" for k in ("rdfxml", "owl", "turtle", "jsonld")]
    [card] = [c for c in v["validation"]["cards"] if c["key"] == uv.AUDIT]
    assert card["text"] == ("Nada que revisar  \nLa regla no aceptó ninguna correspondencia · "
                            "del usuario: 1")


# ------------------------------------------------------------------ console

@pytest.fixture
def console(monkeypatch, env, targets, tmp_path):  # noqa: F811
    """The console on the test base and DPV; the real LLM would fail (no key)."""
    def no_key(*a, **k):
        raise KeyError("LLM_API_KEY")
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.ai.llm_client.LLMClient", no_key)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma-cli"))
    monkeypatch.setattr("src.config.load_config", lambda: env.cfg)
    monkeypatch.setattr(inh, "load_base", lambda **kw: env.base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)


def test_console_shows_the_changes_and_applies_them(aligned, console, capsys):
    w = aligned
    change = ["--espacio", w.id, "--cambio", "PersonalData broadMatch PersonalData",
              "--cambio", "Pizza exactMatch NoExiste"]
    assert ua._run(change) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "Cambios pedidos (los revisa una regla fija)",
        "  [válido] PersonalData: exactMatch PersonalData (regla, versión 1) -> broadMatch "
        "PersonalData (usuario)",
        "  [rechazado] Pizza exactMatch NoExiste: «NoExiste» no es un término del DPV 2.3.",
        "  1 cambio válido de 2. Para aplicarlo y crear la versión 2, agrega --aplicar."]
    assert not ud.log_path(w).exists()                         # only shown

    assert ua._run([*change, "--aplicar", "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Ontología alineada (S7-T08)" in out and "mi-gdpr-alineada-v2-turtle.zip" in out
    assert "    decididas por el usuario: 1 correspondencia en 1 concepto" in out
    assert out.rstrip().endswith("La versión 1 quedó en history/v1/ con sus cuatro descargas.")
    assert "  Versión 2 lista en " in out

    assert ua._run(["--espacio", w.id, "--cambio", "Foo exactMatch PersonalData",
                    "--aplicar", "--sin-ia"]) == 1
    assert capsys.readouterr().out.endswith(
        "  Ningún cambio es válido: la versión 2 sigue igual.\n")


def test_console_errors(aligned, console, capsys, tmp_path):
    w = aligned
    path = tmp_path / "gdpr.ttl"
    path.write_text("@prefix ex: <http://example.org/> .\n", encoding="utf-8")
    cases = [([str(path), "--cambio", "Pizza quitar"], "Error: --cambio va con --espacio."),
             (["--espacio", w.id, "--aplicar"], "Error: --aplicar va con al menos un --cambio."),
             (["--espacio", w.id, "--cambio", "Pizza"], "Error: No entiendo el cambio «Pizza»."),
             (["--espacio", w.id, "--cambio", "Pizza quitar"],
              "Ninguno se puede aplicar.")]
    for argv, expected in cases:
        code = ua._run(argv)
        assert expected in capsys.readouterr().out
        assert code == (0 if expected.startswith("Ninguno") else 1)
