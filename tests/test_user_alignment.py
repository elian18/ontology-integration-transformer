"""S6-T07: the aligned ontology of an upload (validated module + automatic proposals apart),
its manifest and CSV, and the zip in every format with a catalog that resolves the imports."""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.alignment import inherit as inh
from src.alignment import propose as prop
from src.alignment import user_alignment as ua
from src.alignment.bundle import PROTEGE_JSONLD_NOTE
from src.alignment.inherit import inherit_for_workspace, load_base
from src.alignment.propose import propose_for_workspace
from src.alignment.user_alignment import (CSV_FILE, GENERATED_IRI_BASE, INHERITED_ALIGNED,
                                          MANIFEST_FILE, NO_PROPOSAL, NOT_ALIGNABLE, PROPOSED,
                                          align_upload, build_user_package, concept_rows,
                                          load_manifest, write_user_alignment)
from src.transform.convert import load_graph
from src.transform.package import CATALOG_FILE, README_FILE
from src.workspace.store import DONE, open_ontology
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_propose import CFG, TTL, Embed, FakeLLM, targets  # noqa: F401
from tests.test_alignment_write import DPV, ONTO, _decide_all

EX = "http://example.org/gdpr#"
ONTO_IRI = "http://example.org/gdpr"
D = "https://w3id.org/dpv#"


@pytest.fixture
def base(tmp_path):
    data, log = _decide_all(tmp_path)
    cand = tmp_path / "base.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    return load_base(cand, log)


def _ws(tmp_path, base, targets, *, ttl=TTL, llm=None, use_llm=True):
    w = open_ontology("Mi GDPR.ttl", ttl.encode("utf-8"),
                      cfg={"workspaces": {"dir": str(tmp_path / "ws")}})
    inherit_for_workspace(w, base=base, use_embeddings=False)
    propose_for_workspace(w, embed_fn=Embed(), llm=llm or FakeLLM(), use_llm=use_llm,
                          targets=targets, cfg=CFG, sleep_fn=lambda s: None)
    return w


def _unzip(package, folder: Path) -> Path:
    zipfile.ZipFile(io.BytesIO(package.data)).extractall(folder)
    return folder


# ------------------------------------------------------------------ writing

def test_needs_the_previous_steps(tmp_path):
    w = open_ontology("x.ttl", TTL.encode("utf-8"),
                      cfg={"workspaces": {"dir": str(tmp_path / "ws")}})
    with pytest.raises(ValueError, match="heredar"):
        write_user_alignment(w)
    with pytest.raises(ValueError, match="escribir la ontología alineada"):
        build_user_package(w)


def test_validated_and_proposals_modules_stay_apart(tmp_path, base, targets):
    w = _ws(tmp_path, base, targets)
    m = write_user_alignment(w)
    assert m["ontology_iri"] == ONTO_IRI and not m["ontology_iri_generated"]
    assert m["modules"] == {"alignment": ONTO_IRI + "-dpv-alignment",
                            "proposals": ONTO_IRI + "-dpv-proposals"}
    validated = load_graph(w.path / m["files"]["alignment"])
    auto = load_graph(w.path / m["files"]["proposals"])
    consent, personal = URIRef(EX + "Consent"), URIRef(EX + "PersonalData")
    # validated: inherited from the base, with where it comes from
    assert (consent, SKOS.exactMatch, URIRef(DPV + "Consent")) in validated
    assert (consent, SKOS.relatedMatch, URIRef(DPV + "ConsentRecord")) in validated
    assert (URIRef(m["modules"]["alignment"]), OWL.imports, URIRef(ONTO_IRI)) in validated
    ax = validated.value(predicate=OWL.annotatedTarget, object=URIRef(DPV + "Consent"))
    assert validated.value(ax, DCTERMS.source) == URIRef(ONTO + "Consent")
    assert "Heredada de Consent (mismo nombre)" in str(validated.value(ax, RDFS.comment))
    assert not list(validated.triples((personal, None, None)))       # never mixed
    # proposals: automatic, annotated, with confidence
    assert (personal, SKOS.closeMatch, URIRef(D + "PersonalData")) in auto
    ax = auto.value(predicate=OWL.annotatedSource, object=personal)
    assert str(auto.value(ax, RDFS.comment)).startswith("Propuesta automática, no validada.")
    conf = auto.value(ax, URIRef(m["modules"]["proposals"] + "#confidence"))
    assert conf.datatype == XSD.decimal and float(conf) >= 0.6
    assert (URIRef(m["modules"]["proposals"]), OWL.imports, URIRef(ONTO_IRI)) in auto
    assert not list(auto.triples((consent, None, None)))
    assert w.step("write")["status"] == DONE and w.step("write")["validated_mappings"] == 2


def test_untyped_proposal_uses_see_also(tmp_path, base, targets):
    w = _ws(tmp_path, base, targets, use_llm=False)
    m = write_user_alignment(w)
    auto = load_graph(w.path / m["files"]["proposals"])
    assert (URIRef(EX + "PersonalData"), RDFS.seeAlso, URIRef(D + "PersonalData")) in auto
    assert m["summary"]["proposed_untyped"] == m["summary"]["proposed_mappings"] >= 1


def test_no_proposals_means_no_proposals_module(tmp_path, base, targets):
    w = _ws(tmp_path, base, targets, llm=FakeLLM(relation="none"))
    m = write_user_alignment(w)
    assert m["modules"]["proposals"] is None and "proposals" not in m["files"]
    assert not list(w.output_dir.glob("*-dpv-proposals.ttl"))
    readme = zipfile.ZipFile(io.BytesIO(build_user_package(w).data)).read(README_FILE)
    assert "No hubo propuestas automáticas" in readme.decode("utf-8")


def test_ontology_without_iri_gets_one(tmp_path, base, targets):
    ttl = TTL.replace("<http://example.org/gdpr> a owl:Ontology .", "")
    w = _ws(tmp_path, base, targets, ttl=ttl)
    m = write_user_alignment(w)
    assert m["ontology_iri_generated"] and m["ontology_iri"] == GENERATED_IRI_BASE + w.id
    copy = load_graph(w.path / m["files"]["ontology"])
    assert (URIRef(m["ontology_iri"]), RDF.type, OWL.Ontology) in copy
    folder = _unzip(build_user_package(w), tmp_path / "z")
    g = _resolve(folder, m["modules"]["alignment"])
    assert (URIRef(EX + "Pizza"), RDF.type, OWL.Class) in g          # user ontology imported
    assert w.input_path().read_text(encoding="utf-8") == ttl          # original untouched


def test_csv_and_manifest(tmp_path, base, targets):
    w = _ws(tmp_path, base, targets)
    m = write_user_alignment(w)
    with open(w.output_dir / CSV_FILE, encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    status = {r["concept_name"]: r["status"] for r in rows}
    assert status["Consent"] == INHERITED_ALIGNED and status["PersonalData"] == PROPOSED
    assert status["Pizza"] == NO_PROPOSAL
    assert sum(r["concept_name"] == "Consent" for r in rows) == 2      # one row per mapping
    assert m["summary"]["concepts"] == len(status)
    saved = load_manifest(w)
    assert saved["workspace"] == w.id and saved["source_id"] == f"upload:{w.id}"
    assert saved["base"]["decisions_log_sha256"] is None or len(
        saved["base"]["decisions_log_sha256"]) == 64


def test_concept_rows_cover_every_status():
    inherited = {"inherited": [], "not_inherited": [{"key": "e#i", "name": "ind1",
                                                       "reason": "kind"}]}
    proposals = {"proposals": [{"key": "e#a", "name": "A", "status": "ai_none", "score": 0.9,
                                "dpv_iri": D + "X"}]}
    rows = concept_rows(inherited, proposals)
    assert [r["status"] for r in rows] == ["ai_rejected", NOT_ALIGNABLE]
    assert rows[1]["estado"] == "no se alinea (individuo)"


# ------------------------------------------------------------------ the package

@pytest.mark.parametrize("fmt,ext", [("rdfxml", ".rdf"), ("owl", ".owl"), ("turtle", ".ttl"),
                                     ("jsonld", ".jsonld")])
def test_package_resolves_in_every_format(tmp_path, base, targets, fmt, ext):
    w = _ws(tmp_path, base, targets)
    m = write_user_alignment(w)
    package = build_user_package(w, fmt)
    stem = "mi-gdpr"
    assert package.main_file == f"{stem}-dpv-alignment{ext}"
    assert [f for f in package.included if f.endswith(ext)] == [
        f"{stem}-dpv-alignment{ext}", f"{stem}-dpv-proposals{ext}", f"{stem}{ext}"]
    assert {MANIFEST_FILE, CSV_FILE, CATALOG_FILE, README_FILE} <= set(package.included)
    folder = _unzip(package, tmp_path / fmt)
    validated = _resolve(folder, m["modules"]["alignment"])
    expected = load_graph(w.path / m["files"]["alignment"]) + load_graph(w.input_path())
    assert isomorphic(validated, expected)
    auto = _resolve(folder, m["modules"]["proposals"])
    assert (URIRef(EX + "PersonalData"), SKOS.closeMatch, URIRef(D + "PersonalData")) in auto
    readme = (folder / README_FILE).read_text(encoding="utf-8")
    assert (PROTEGE_JSONLD_NOTE in readme) == (fmt == "jsonld")
    assert "propuestas AUTOMÁTICAS, NO validadas" in readme


def test_align_upload_runs_everything_and_reuses_steps(tmp_path, base, targets):
    embed, llm, seen = Embed(), FakeLLM(), []
    kw = dict(fmt="rdfxml", embed_fn=embed, llm=llm, base=base, targets=targets,
              cfg={**CFG, "workspaces": {"dir": str(tmp_path / "ws")}})
    w, m, path = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), progress=seen.append, **kw)
    assert seen == ["Heredar lo validado", "Buscar candidatos del DPV",
                    "Escribir la ontología alineada", "Preparar la descarga"]
    assert path.name == "mi-gdpr-alineada-rdfxml.zip" and path.exists()
    assert all(s["status"] == DONE for s in w.progress())
    calls = (embed.calls, llm.calls)
    w2, _, path2 = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), **{**kw, "fmt": "jsonld"})
    assert w2.path == w.path and (embed.calls, llm.calls) == calls     # nothing recomputed
    assert path2.name == "mi-gdpr-alineada-jsonld.zip"


def test_cli(tmp_path, monkeypatch, capsys, base, targets):
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.config.load_config",
                        lambda: {"workspaces": {"dir": str(tmp_path / "ws")}, **CFG})
    monkeypatch.setattr(inh, "load_base", lambda **kw: base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)
    path = tmp_path / "gdpr.ttl"
    path.write_text(TTL, encoding="utf-8")
    assert ua._run([str(path), "--formato", "turtle", "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "Ontología alineada (S6-T07)" in out and "gdpr-alineada-turtle.zip" in out
    assert re.search(r"propuestas automáticas: ([1-9]\d*) \(\1 sin tipo\)", out)  # --sin-ia
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
                              use_llm=False, cfg=cfg)
    onto = [c for c in real.concepts if c.origin == "ontology"]
    assert m["summary"]["validated_mappings"] == sum(len(c.mappings) for c in onto)
    assert m["summary"]["proposed_mappings"] == 0 and m["modules"]["proposals"] is None
    folder = tmp_path / "unzipped"
    zipfile.ZipFile(path).extractall(folder)
    g = _resolve(folder, m["modules"]["alignment"])
    assert len(list(g.triples((None, SKOS.exactMatch, None)))) > 0