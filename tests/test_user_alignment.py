"""S6-T07 + S7-T06 + S7-T08: the aligned ontology of an upload (ONE module: inherited
correspondences and the ones the rule accepted, each annotated with who decided it and its
version; suggestions never written), the module of the labels es/en the ontology lacked, its
manifest and CSV, and the zip in every format with a catalog that resolves the imports."""
from __future__ import annotations

import csv
import io
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest
from rdflib import Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.alignment import inherit as inh
from src.alignment import propose as prop
from src.alignment import user_alignment as ua
from src.alignment.bundle import PROTEGE_JSONLD_NOTE
from src.alignment.inherit import inherit_for_workspace, load_base
from src.alignment.user_alignment import (ACCEPTED, CSV_FILE, GENERATED_IRI_BASE,
                                          INHERITED_ALIGNED, MANIFEST_FILE, NOT_ALIGNABLE,
                                          RULE_NO_MATCH, SUGGESTED, UNDECIDED, align_upload,
                                          build_user_package, concept_rows, load_manifest,
                                          write_user_alignment)
from src.transform import labels as lab
from src.transform.convert import load_graph
from src.transform.package import CATALOG_FILE, README_FILE
from src.workspace.store import DONE, open_ontology
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_propose import (TTL, DeciderLLM, Embed, env, run,  # noqa: F401
                                          targets)
from tests.test_alignment_write import DPV, ONTO

EX = "http://example.org/gdpr#"
ONTO_IRI = "http://example.org/gdpr"
MODULE = ONTO_IRI + "-dpv-alignment"
LABELS = ONTO_IRI + "-labels"
D = "https://w3id.org/dpv#"


def _ws(env, targets, *, ttl=TTL, llm=None, use_llm=True):
    w = open_ontology("Mi GDPR.ttl", ttl.encode("utf-8"), cfg=env.cfg)
    inherit_for_workspace(w, base=env.base, use_embeddings=False)
    run(w, env, targets, llm or DeciderLLM(), use_llm=use_llm)
    return w


def _labelled(env, targets, *, use_llm=True):
    """A workspace whose labels (S7-T08) are done too, as ``align_upload`` leaves it."""
    w = _ws(env, targets)
    lab.labels_for_workspace(w, llm=DeciderLLM(), use_llm=use_llm, cfg=env.cfg)
    return w


def _csv(w) -> list[dict]:
    with open(w.output_dir / CSV_FILE, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _unzip(package, folder: Path) -> Path:
    zipfile.ZipFile(io.BytesIO(package.data)).extractall(folder)
    return folder


def _axiom(g, subject, target):
    return next(ax for ax in g.subjects(OWL.annotatedSource, URIRef(subject))
                if g.value(ax, OWL.annotatedTarget) == URIRef(target))


# ------------------------------------------------------------------ writing

def test_needs_the_previous_steps(env):
    w = open_ontology("x.ttl", TTL.encode("utf-8"), cfg=env.cfg)
    with pytest.raises(ValueError, match="heredar"):
        write_user_alignment(w)
    inherit_for_workspace(w, base=env.base, use_embeddings=False)
    with pytest.raises(ValueError, match="decidir lo que no se heredó"):
        write_user_alignment(w)
    with pytest.raises(ValueError, match="escribir la ontología alineada"):
        build_user_package(w)


def test_one_module_with_the_base_and_the_rule(env, targets):
    w = _ws(env, targets)
    m = write_user_alignment(w, cfg=env.cfg)
    assert m["ontology_iri"] == ONTO_IRI and not m["ontology_iri_generated"]
    assert m["modules"] == {"alignment": MODULE} and m["alignment_version"] == 1
    g = load_graph(w.path / m["files"]["alignment"])
    by, conf = URIRef(MODULE + "#decidedBy"), URIRef(MODULE + "#confidence")
    assert (URIRef(MODULE), OWL.imports, URIRef(ONTO_IRI)) in g
    assert (URIRef(MODULE), OWL.versionInfo, Literal("1")) in g
    assert (by, RDF.type, OWL.AnnotationProperty) in g
    # inherited from the base: who and where it comes from
    consent = URIRef(EX + "Consent")
    assert (consent, SKOS.exactMatch, URIRef(DPV + "Consent")) in g
    ax = _axiom(g, consent, DPV + "Consent")
    assert g.value(ax, DCTERMS.source) == URIRef(ONTO + "Consent")
    assert g.value(ax, by) == Literal("base", lang="es")
    assert g.value(ax, OWL.versionInfo) == Literal("1")
    assert "Heredada de Consent (mismo nombre)" in str(g.value(ax, RDFS.comment))
    # accepted by the rule: score, LLM justification, who
    personal = URIRef(EX + "PersonalData")
    assert (personal, SKOS.exactMatch, URIRef(D + "PersonalData")) in g
    ax = _axiom(g, personal, D + "PersonalData")
    assert g.value(ax, by) == Literal("regla", lang="es")
    assert g.value(ax, conf).datatype == XSD.decimal and float(g.value(ax, conf)) >= 0.5
    text = str(g.value(ax, RDFS.comment))
    assert text.startswith("Aceptada automáticamente por la regla (confianza alta de la IA")
    assert "Justificación de la IA: Decisión de prueba para PersonalData." in text
    # suggestions and "sin correspondencia" write nothing
    assert not list(g.triples((URIRef(EX + "DataController"), None, None)))
    assert not list(g.triples((URIRef(EX + "Pizza"), None, None)))
    assert not list(w.output_dir.glob("*-dpv-proposals.ttl"))
    s = m["summary"]
    assert (s["validated_mappings"], s["accepted_mappings"], s["mappings"], s["suggested"]) == (
        2, 1, 3, 1)
    step = w.step("write")
    assert step["status"] == DONE and step["accepted_mappings"] == 1 and step["suggested"] == 1


def test_without_ai_the_accepted_ones_use_see_also(env, targets):
    w = _ws(env, targets, use_llm=False)
    m = write_user_alignment(w, cfg=env.cfg)
    g = load_graph(w.path / m["files"]["alignment"])
    personal = URIRef(EX + "PersonalData")
    assert (personal, RDFS.seeAlso, URIRef(D + "PersonalData")) in g
    ax = _axiom(g, personal, D + "PersonalData")
    assert g.value(ax, URIRef(MODULE + "#decidedBy")) == Literal("regla sin IA", lang="es")
    assert str(g.value(ax, RDFS.comment)).startswith(
        "Aceptada automáticamente sin IA: mejor candidato con puntaje")
    s = m["summary"]
    assert s["accepted_untyped"] == s["accepted_without_ai"] == s["accepted_mappings"] == 2
    assert s["by_status"][UNDECIDED] == 1                                 # Pizza


def test_the_proposals_module_of_v060_is_removed(env, targets):
    w = _ws(env, targets)
    w.output_dir.mkdir(parents=True, exist_ok=True)
    old = w.output_dir / f"{ua.stems(w)['proposals']}.ttl"
    old.write_text("# v0.6.0", encoding="utf-8")
    m = write_user_alignment(w, cfg=env.cfg)
    assert not old.exists() and "proposals" not in m["files"]


def test_ontology_without_iri_gets_one(env, targets, tmp_path):
    ttl = TTL.replace("<http://example.org/gdpr> a owl:Ontology .", "")
    w = _ws(env, targets, ttl=ttl)
    m = write_user_alignment(w, cfg=env.cfg)
    assert m["ontology_iri_generated"] and m["ontology_iri"] == GENERATED_IRI_BASE + w.id
    copy = load_graph(w.path / m["files"]["ontology"])
    assert (URIRef(m["ontology_iri"]), RDF.type, OWL.Ontology) in copy
    folder = _unzip(build_user_package(w), tmp_path / "z")
    g = _resolve(folder, m["modules"]["alignment"])
    assert (URIRef(EX + "Pizza"), RDF.type, OWL.Class) in g          # user ontology imported
    assert w.input_path().read_text(encoding="utf-8") == ttl          # original untouched


def test_csv_and_manifest(env, targets):
    w = _ws(env, targets)
    write_user_alignment(w, cfg=env.cfg)
    with open(w.output_dir / CSV_FILE, encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by = {r["concept_name"]: r for r in rows}
    assert by["Consent"]["status"] == INHERITED_ALIGNED and by["Consent"]["decided_by"] == "base"
    assert sum(r["concept_name"] == "Consent" for r in rows) == 2      # one row per mapping
    assert (by["PersonalData"]["status"], by["PersonalData"]["decided_by"],
            by["PersonalData"]["confidence"]) == (ACCEPTED, "rule", "alta")
    dc = by["DataController"]
    assert (dc["status"], dc["dpv_name"], dc["relation"]) == (SUGGESTED, "DataController",
                                                              "skos:relatedMatch")
    assert dc["estado"] == "sugerida (no entra a la ontología)"
    assert by["Pizza"]["status"] == RULE_NO_MATCH
    assert {r["version"] for r in rows} == {"1"}
    saved = load_manifest(w)
    assert saved["workspace"] == w.id and saved["source_id"] == f"upload:{w.id}"
    assert saved["settings"]["rule"]["min_score"] == 0.5
    assert saved["settings"]["prompts"]["decide"]["name"] == "decide"
    assert saved["decider"]["suggested"] == 1 and saved["summary"]["concepts"] == len(by)
    assert saved["base"]["decisions_log_sha256"] is None or len(
        saved["base"]["decisions_log_sha256"]) == 64


def test_the_labels_go_in_their_own_module(env, targets):
    w = _labelled(env, targets)
    m = write_user_alignment(w, cfg=env.cfg)
    assert m["modules"] == {"alignment": MODULE, "labels": LABELS}
    assert m["files"]["labels"] == "output/mi-gdpr-labels.ttl"
    assert m["triples"]["labels"] == 6 + 8                     # 4 classes x (es, en)
    g = load_graph(w.path / m["files"]["labels"])
    personal = URIRef(EX + "PersonalData")
    assert (personal, RDFS.label, Literal("Dato personal", lang="es")) in g
    assert (personal, RDFS.label, Literal("Personal Data", lang="en")) in g
    assert (URIRef(LABELS), OWL.imports, URIRef(ONTO_IRI)) in g
    a = load_graph(w.path / m["files"]["alignment"])
    assert set(a.objects(URIRef(MODULE), OWL.imports)) == {URIRef(ONTO_IRI), URIRef(LABELS)}
    assert not list(a.triples((None, RDFS.label, Literal("Dato personal", lang="es"))))
    labels = m["labels"]
    assert (labels["concepts"], labels["complete"], labels["translated"], labels["pending"]) == (
        4, 0, 4, 0)
    assert labels["added"] == {"es": 4, "en": 4} and labels["glossary"]["entries"] == 2
    assert labels["ai"]["calls"] == 1 and labels["ai"]["available"] is True
    assert m["settings"]["prompts"]["labels"] == lab.PROMPT.manifest()
    assert any("módulo de etiquetas" in n for n in m["notes"])
    by = {r["concept_name"]: r for r in _csv(w)}
    assert (by["PersonalData"]["label_es"], by["PersonalData"]["label_en"],
            by["PersonalData"]["labels_added"]) == ("Dato personal", "Personal Data", "es, en")
    assert by["Consent"]["label_es"] == "Consentimiento"
    assert list(by["Consent"])[:5] == ["concept_iri", "concept_name", "label_es", "label_en",
                                       "labels_added"]


def test_the_package_carries_the_labels(env, targets, tmp_path):
    w = _labelled(env, targets)
    m = write_user_alignment(w, cfg=env.cfg)
    package = build_user_package(w, "turtle")
    assert [f for f in package.included if f.endswith(".ttl")] == [
        "mi-gdpr-dpv-alignment.ttl", "mi-gdpr-labels.ttl", "mi-gdpr.ttl"]
    assert package.main_file == "mi-gdpr-dpv-alignment.ttl"
    folder = _unzip(package, tmp_path / "z")
    g = _resolve(folder, m["modules"]["alignment"])
    expected = (load_graph(w.path / m["files"]["alignment"]) + load_graph(w.input_path())
                + load_graph(w.path / m["files"]["labels"]))
    assert isomorphic(g, expected)
    assert (URIRef(EX + "Pizza"), RDFS.label, Literal("Pizza", lang="es")) in g
    readme = (folder / README_FILE).read_text(encoding="utf-8")
    assert ("  - mi-gdpr-labels.ttl: las etiquetas que le faltaban a tu ontología (4 en "
            "español y 4 en inglés), escritas por la IA. Tus etiquetas no cambian.") in readme
    assert "quedaron sin sus etiquetas" not in readme
    rdf = build_user_package(w, "rdfxml")
    assert "mi-gdpr-labels.rdf" in rdf.included


def test_labels_left_pending_write_no_module(env, targets, tmp_path):
    w = _labelled(env, targets, use_llm=False)
    m = write_user_alignment(w, cfg=env.cfg)
    assert m["modules"] == {"alignment": MODULE} and "labels" not in m["files"]
    assert m["labels"]["pending"] == 4 and m["labels"]["ai"]["available"] is False
    a = load_graph(w.path / m["files"]["alignment"])
    assert set(a.objects(URIRef(MODULE), OWL.imports)) == {URIRef(ONTO_IRI)}
    assert {r["labels_added"] for r in _csv(w)} == {"pendiente"}
    readme = (_unzip(build_user_package(w), tmp_path / "z") / README_FILE).read_text(
        encoding="utf-8")
    assert "mi-gdpr-labels" not in readme and "las etiquetas que le faltaban" not in readme
    assert ("  - 4 conceptos quedaron sin sus etiquetas en español o inglés (la IA no las "
            "completó)") in readme
    assert not any("módulo de etiquetas" in n for n in m["notes"])


def test_without_labels_result_no_module_and_a_stale_one_goes(env, targets):
    w = _labelled(env, targets)
    write_user_alignment(w, cfg=env.cfg)
    module = w.output_dir / "mi-gdpr-labels.ttl"
    assert module.exists()
    (w.work_dir / lab.RESULT_FILE).unlink()
    m = write_user_alignment(w, cfg=env.cfg)
    assert not module.exists() and m["labels"] is None and "labels" not in m["triples"]
    assert "labels" not in m["settings"]["prompts"]
    assert {r["label_es"] for r in _csv(w)} == {""}


def test_concept_rows_cover_every_status():
    inherited = {"inherited": [], "not_inherited": [{"key": "e#i", "name": "ind1",
                                                       "reason": "kind"},
                                                      {"key": "e#p", "name": "hasX",
                                                       "reason": "kind"}]}
    entries = [{"key": f"e#{s}", "name": s, "status": s, "decided_by": "rule",
                "dpv_iri": D + "X" if s in ("accepted", "suggested") else None,
                "dpv_name": "X", "relation": None, "score": 0.9, "version": 1}
               for s in ("accepted", "suggested", "no_match", "no_proposal", "undecided")]
    entries.append({**entries[-1], "key": "e#p", "name": "hasX"})       # a property, decided
    rows = concept_rows(inherited, {"entries": entries, "alignment_version": 1})
    assert [r["status"] for r in rows] == [ACCEPTED, RULE_NO_MATCH, SUGGESTED, "no_proposal",
                                           UNDECIDED, UNDECIDED, NOT_ALIGNABLE]
    assert rows[0]["relation"] == "rdfs:seeAlso" and rows[0]["dpv_iri"] == D + "X"
    assert "dpv_iri" not in rows[1]                                 # nothing for "ninguno"
    assert rows[-1]["estado"] == "no se alinea (individuo)" and rows[-1]["concept_name"] == "ind1"


# ------------------------------------------------------------------ the package

@pytest.mark.parametrize("fmt,ext", [("rdfxml", ".rdf"), ("owl", ".owl"), ("turtle", ".ttl"),
                                     ("jsonld", ".jsonld")])
def test_package_resolves_in_every_format(env, targets, tmp_path, fmt, ext):
    w = _ws(env, targets)
    m = write_user_alignment(w, cfg=env.cfg)
    package = build_user_package(w, fmt)
    stem = "mi-gdpr"
    assert package.main_file == f"{stem}-dpv-alignment{ext}"
    assert [f for f in package.included if f.endswith(ext)] == [f"{stem}-dpv-alignment{ext}",
                                                                f"{stem}{ext}"]
    assert {MANIFEST_FILE, CSV_FILE, CATALOG_FILE, README_FILE} <= set(package.included)
    folder = _unzip(package, tmp_path / fmt)
    resolved = _resolve(folder, m["modules"]["alignment"])
    expected = load_graph(w.path / m["files"]["alignment"]) + load_graph(w.input_path())
    assert isomorphic(resolved, expected)
    assert (URIRef(EX + "PersonalData"), SKOS.exactMatch, URIRef(D + "PersonalData")) in resolved
    readme = (folder / README_FILE).read_text(encoding="utf-8")
    assert (PROTEGE_JSONLD_NOTE in readme) == (fmt == "jsonld")
    assert "3 correspondencias con el DPV" in readme
    assert "1 aceptadas automáticamente por la regla (confianza alta" in readme
    assert "1 sugeridas" in readme and "NO están en la ontología" in readme


def test_align_upload_runs_everything_and_reuses_steps(env, targets):
    embed, llm, seen = Embed(), DeciderLLM(), []
    kw = dict(fmt="rdfxml", embed_fn=embed, llm=llm, base=env.base, targets=targets,
              cfg=env.cfg, chroma_path=env.chroma, sleep_fn=lambda s: None)
    w, m, path = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), progress=seen.append, **kw)
    assert seen == ["Heredar lo validado", "Buscar candidatos del DPV",
                    "Completar las etiquetas en español e inglés",
                    "Escribir la ontología alineada", "Preparar la descarga"]
    assert path.name == "mi-gdpr-alineada-rdfxml.zip" and path.exists()
    assert all(s["status"] == DONE for s in w.progress())
    assert m["modules"]["labels"] == LABELS and m["labels"]["pending"] == 0
    text = ua.render_console(w, m, path)
    assert ("  Etiquetas es/en: 0 conceptos ya las tenían; la IA agregó 4 en español y 4 en "
            "inglés (módulo output/mi-gdpr-labels.ttl, 14 tripletas)\n") in text
    assert "  IA de las etiquetas: 1 llamadas, 0 desde la caché, 0 respuestas inválidas" in text
    assert m["recomputed"] == {"inherit": True, "decider": True, "labels": True,
                               "evidence": False}
    decider_line = [line for line in text.splitlines() if "IA del decisor" in line][0]
    assert decider_line.startswith("  IA del decisor: ") and ua.REUSED not in text
    calls = (embed.calls, llm.calls)
    seen = []
    w2, m2, path2 = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), progress=seen.append,
                                 **{**kw, "fmt": "jsonld"})
    assert w2.path == w.path and (embed.calls, llm.calls) == calls     # nothing recomputed
    assert seen == ["Escribir la ontología alineada", "Preparar la descarga"]
    assert path2.name == "mi-gdpr-alineada-jsonld.zip"
    assert not any(m2["recomputed"].values())
    again = ua.render_console(w2, m2, path2)                           # S7-T13: says so
    assert (f"  IA del decisor: {ua.REUSED} ({decider_line.split(': ', 1)[1]})") in again
    assert (f"  IA de las etiquetas: {ua.REUSED} (1 llamadas, 0 desde la caché, 0 respuestas "
            "inválidas)") in again
    stricter = {**env.cfg, "decider": {**env.cfg["decider"], "rule": {"min_score": 0.99}}}
    _, m3, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), **{**kw, "cfg": stricter})
    assert llm.calls == calls[1]                                       # decided again: cache
    assert m3["settings"]["rule"]["min_score"] == 0.99


def test_labels_left_pending_are_completed_on_the_next_upload(env, targets):
    kw = dict(fmt="turtle", embed_fn=Embed(), base=env.base, targets=targets, cfg=env.cfg,
              chroma_path=env.chroma, sleep_fn=lambda s: None)
    w, m, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), use_llm=False, **kw)
    assert m["labels"]["pending"] == 4 and "labels" not in m["modules"]
    llm, seen = DeciderLLM(), []
    _, m2, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), llm=llm, progress=seen.append,
                            **kw)
    assert "Completar las etiquetas en español e inglés" in seen
    assert m2["labels"]["pending"] == 0 and m2["modules"]["labels"] == LABELS


def test_a_workspace_of_v060_is_decided_again(env, targets):
    kw = dict(fmt="turtle", embed_fn=Embed(), llm=DeciderLLM(), base=env.base, targets=targets,
              cfg=env.cfg, chroma_path=env.chroma, sleep_fn=lambda s: None)
    w, _, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), **kw)
    (w.work_dir / prop.RESULT_FILE).unlink()                           # as in v0.6.0
    (w.work_dir / "proposals.json").write_text("{}", encoding="utf-8")
    _, m, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), **kw)
    assert prop.load_result(w)["method"] == prop.METHOD and m["summary"]["accepted_mappings"] == 1


def test_decider_ai_text():
    assert ua.decider_ai_text({"calls": 3, "cached": 2}) == ("3 llamadas, 2 respuestas desde la "
                                                             "caché")
    assert ua.decider_ai_text({"calls": 3, "quota_exhausted": True, "errors": ["429 quota"]}) == (
        "3 llamadas, 0 respuestas desde la caché (cuota agotada: vuelve a correr mañana); "
        "error: 429 quota")
    assert ua.decider_ai_text({"calls": 1, "call_limit_reached": True}).endswith(
        "(límite de llamadas: vuelve a correr para seguir)")


def test_without_knowing_what_ran_the_console_says_nothing_about_it(env, targets):
    w, m, path = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), embed_fn=Embed(),
                              llm=DeciderLLM(), base=env.base, targets=targets, cfg=env.cfg,
                              chroma_path=env.chroma, sleep_fn=lambda s: None)
    m = ua.write_user_alignment(w, cfg=env.cfg)                       # called on its own
    assert m["recomputed"] is None and ua.REUSED not in ua.render_console(w, m, path)


def test_cli(tmp_path, monkeypatch, capsys, env, targets):
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma-cli"))
    monkeypatch.setattr("src.config.load_config", lambda: env.cfg)
    monkeypatch.setattr(inh, "load_base", lambda **kw: env.base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)
    path = tmp_path / "gdpr.ttl"
    path.write_text(TTL, encoding="utf-8")
    assert ua._run([str(path), "--formato", "turtle", "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Ontología alineada (S7-T08)" in out and "gdpr-alineada-turtle.zip" in out
    assert re.search(r"aceptadas automáticamente: ([1-9]\d*) \(\1 sin tipo\)", out)  # --sin-ia
    assert "versión 1" in out and "IA del decisor: no usada (IA desactivada (--sin-ia))" in out
    assert "Regla: confianza alta de la IA" in out
    assert ("  Etiquetas es/en: 0 conceptos ya las tenían; la IA agregó 0 en español y 0 en "
            "inglés · pendientes: 4\n") in out
    assert "IA de las etiquetas: no usada (IA desactivada (--sin-ia))" in out
    assert ua._run([str(path), "--formato", "owx"]) == 1
    assert ua._run([str(tmp_path / "no.ttl")]) == 1


# ------------------------------------------------------------------ real base (control)

def test_real_sample_end_to_end(tmp_path):
    from src.config import load_config
    sample = Path(__file__).resolve().parents[1] / "data" / "input" / "sample.ttl"
    try:
        real = load_base()
    except FileNotFoundError:
        pytest.skip("Falta data/output/alignment-candidates.json")
    if not sample.exists():
        pytest.skip("Falta data/input/sample.ttl")
    cfg = {**load_config(), "workspaces": {"dir": str(tmp_path / "ws")}}
    w, m, path = align_upload("sample.ttl", sample.read_bytes(), fmt="turtle", base=real,
                              use_llm=False, cfg=cfg, chroma_path=tmp_path / "chroma")
    onto = [c for c in real.concepts if c.origin == "ontology"]
    assert m["summary"]["validated_mappings"] == sum(len(c.mappings) for c in onto)
    assert m["summary"]["accepted_mappings"] == 0 and m["summary"]["suggested"] == 0
    folder = tmp_path / "unzipped"
    zipfile.ZipFile(path).extractall(folder)
    g = _resolve(folder, m["modules"]["alignment"])
    assert len(list(g.triples((None, SKOS.exactMatch, None)))) > 0