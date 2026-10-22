"""S7-T13: end-to-end test of Sprint 7: the modes with a law of the page "Alinear con el DPV",
on real files.

Runs what the page runs (``app.services.align_ontology.run``, S7-T12) on the real laws of
``data/input/laws`` and on GDPRtEXT, with the real validated base, the real OntoPriv-Core and
the real DPV, but a deterministic embedding and a fake AI, so it is fast and needs no network.
The real embeddings and the real AI were used in the runs whose figures are in the CHANGELOG;
this test locks what does not depend on them:
- "Una ley" on the Colombian law (PDF, Spanish): its 30 articles (S7-T07); the rule accepts the
  well formed concepts the AI gives and rejects the rest (S7-T09); the concepts of the core
  reuse its IRI and the profile imports OntoPriv-Core (S7-T10); the alignment is written, every
  concept gets its labels (S7-T08) and the zip resolves the imports with the core inside;
- "Mi ontología y una ley" on GDPRtEXT + GDPR (PDF, English): 99 articles; the alignment is the
  same as uploading GDPRtEXT alone (the law changes no decision, S7-T11), every concept gets the
  articles to review in alignment.csv and the zip carries no module of the law;
- nothing is written outside the workspaces (S6-T04)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from rdflib import URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, SKOS

from app.services import align_ontology as svc
from src.alignment.inherit import load_base
from src.alignment.user_alignment import CSV_FILE, build_user_package, load_manifest
from src.core.emit import CORE_IRI
from src.workspace.store import BASE_DIRS, DONE, load_workspace
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_propose import DeciderLLM

ROOT = Path(__file__).resolve().parents[1]
LAWS = ROOT / "data" / "input" / "laws"
COLOMBIA = LAWS / "ley-1581-2012-colombia.pdf"
GDPR = LAWS / "gdpr-en.pdf"
GDPRTEXT = ROOT / "data" / "input" / "gdprtext-ontology.ttl"
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (COLOMBIA, GDPR, GDPRTEXT)),
    reason="Faltan las leyes de data/input/laws o data/input/gdprtext-ontology.ttl")

# What the fake AI extracts from article 3 (definitions) of the Colombian law: three concepts
# that OntoPriv-Core already has (as in the real run of S7-T10) and two the rule rejects.
DEFINITIONS = [
    {"name": "PersonalData", "label": "Dato personal", "type": "class", "article": 3,
     "definition": "Cualquier información vinculada a una persona natural determinada."},
    {"name": "Database", "label": "Base de Datos", "type": "class", "article": 3,
     "definition": "Conjunto organizado de datos personales que sea objeto de tratamiento."},
    {"name": "DataSubject", "label": "Titular", "type": "class", "article": 3,
     "definition": "Persona natural cuyos datos personales sean objeto de tratamiento."},
    {"name": "x", "label": "Mal nombrado", "type": "class", "article": 3,
     "definition": "Un concepto con un nombre técnico que la regla no acepta."},
    {"name": "GhostConcept", "label": "Concepto fantasma", "type": "class", "article": 99,
     "definition": "Un concepto que cita un artículo que la ley no tiene."},
]


class ExtractingLLM(DeciderLLM):
    """The extraction prompt (S7-T09): DEFINITIONS for article 3 and one concept for every
    other article; the decider and the labels as DeciderLLM."""

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        if "Extrae los conceptos" in prompt:
            self.calls += 1
            out = []
            for n in (int(n) for n in re.findall(r"=== ARTICULO (\d+) ", context)):
                out += DEFINITIONS if n == 3 else [
                    {"name": f"ArticleConcept{n}", "label": f"Concepto del artículo {n}",
                     "type": "class", "article": n,
                     "definition": f"Lo que la ley regula en su artículo {n}."}]
            return "\n".join(json.dumps(c, ensure_ascii=False) for c in out)
        return super().ask(prompt, context, max_tokens, json_mode)


def _random_embed(texts):
    """Each distinct text gets its own random direction (seeded by the text): two different
    texts are almost orthogonal, so no semantic similarity at all, and no two texts collide
    (one direction per hash bucket would make unrelated texts look identical)."""
    out = []
    for t in texts:
        v = np.random.default_rng(zlib.crc32(t.encode("utf-8"))).standard_normal(256)
        out.append(v / np.linalg.norm(v))
    return out


def _snapshot() -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for base in BASE_DIRS for p in (ROOT / base).rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def real_base():
    try:
        return load_base()
    except FileNotFoundError:
        pytest.skip("Falta la base del Sprint 4-5 (data/output/alignment-candidates.json)")


@pytest.fixture
def web(tmp_path, real_base, monkeypatch):
    """The service of the page with the real config pointed at temporary workspaces, no waits
    between the fake AI calls and every article kept as evidence (with the deterministic
    embedding every article is far from every concept)."""
    from src.config import load_config
    real = load_config()
    cfg = {**real, "workspaces": {"dir": str(tmp_path / "ws")},
           "decider": {**real["decider"], "pause_seconds": 0},
           "labels": {**real["labels"], "pause_seconds": 0},
           "extraction": {**real["extraction"], "pause_seconds": 0},
           "law_evidence": {"top_k": 3, "max_distance": None}}
    monkeypatch.setattr(svc, "workspace_cfg", lambda: cfg)

    def run(mode, files, llm=None, seen=None):
        return svc.run(mode, files, progress=(seen if seen is not None else []).append,
                       base=real_base, embed_fn=_random_embed, llm=llm or DeciderLLM(),
                       chroma_path=tmp_path / "chroma", sleep_fn=lambda s: None)
    return run


def _csv(ws_id) -> list[dict]:
    ws = load_workspace(ws_id, cfg=svc.workspace_cfg())
    with open(ws.output_dir / CSV_FILE, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_a_real_law_becomes_an_aligned_ontology_on_ontopriv_core(web, tmp_path):
    before = _snapshot()
    llm, seen = ExtractingLLM(), []
    ws_id = web("law", {"law": (COLOMBIA.name, COLOMBIA.read_bytes())}, llm=llm, seen=seen)
    assert seen == ["Separar la ley en artículos", "Indexar los artículos de la ley",
                    "Extraer los conceptos de la ley con la IA",
                    "Aceptar los conceptos por regla",
                    "Armar el perfil de la ley sobre OntoPriv-Core", "Heredar lo validado",
                    "Buscar candidatos del DPV", "Completar las etiquetas en español e inglés",
                    "Escribir la ontología alineada", "Preparar la descarga"]
    ws = load_workspace(ws_id, cfg=svc.workspace_cfg())
    assert all(s["status"] == DONE for s in ws.progress())
    assert ws.step("segment")["detail"] == "30 artículos (español)"
    m = load_manifest(ws)
    p = m["profile"]
    assert (p["law_concepts"]["concepts"], p["law_concepts"]["accepted"]) == (34, 32)
    assert p["law_concepts"]["reasons"] == {"bad_name": 1, "no_article": 1}   # art. 99: dropped
    assert p["extraction"]["complete"] and p["language"] == "es"
    profile = json.loads((ws.work_dir / "law-profile.json").read_text(encoding="utf-8"))
    assert (p["reused"], p["linked"], p["new"]) == (3, 0, 29)       # the 3 of the core
    reused = {e["name"]: e["core_iri"] for e in profile["entries"] if e["status"] == "reused"}
    assert reused == {"PersonalData": ONTO + "Personal_data", "Database": ONTO + "Data_base",
                      "DataSubject": ONTO + "Data_Owner"}
    assert m["modules"]["core"] == CORE_IRI and m["files"]["core"] == "output/ontopriv-core.ttl"
    assert m["labels"]["pending"] == 0 and m["labels"]["concepts"] == m["summary"]["concepts"]

    v = svc.view(ws_id)
    assert v["mode"] == "law" and v["core"] and v["extraction_note"] is None
    assert v["profile"].startswith("De ley-1581-2012-colombia.pdf: la IA extrajo 34 conceptos "
                                   "y la regla aceptó 32 (32 clases y 0 propiedades)")
    folder = tmp_path / "zip"
    zipfile.ZipFile(io.BytesIO(build_user_package(ws, "turtle").data)).extractall(folder)
    g = _resolve(folder, m["modules"]["alignment"])
    profile_iri = URIRef(m["ontology_iri"])
    assert (profile_iri, OWL.imports, URIRef(CORE_IRI)) in g
    assert (URIRef(CORE_IRI), RDF.type, OWL.Ontology) in g              # the core resolves
    personal = URIRef(ONTO + "Personal_data")
    assert (personal, DCTERMS.source, None) in g                        # its article
    assert any(g.triples((personal, SKOS.exactMatch, None)))            # inherited from the base
    assert _snapshot() == before                                         # base untouched


def test_an_ontology_with_a_real_law_keeps_its_alignment(web):
    before = _snapshot()
    alone = web("ontology", {"ontology": (GDPRTEXT.name, GDPRTEXT.read_bytes())})
    seen = []
    both = web("ontology+law", {"ontology": (GDPRTEXT.name, GDPRTEXT.read_bytes()),
                                "law": (GDPR.name, GDPR.read_bytes())}, seen=seen)
    assert "Buscar en la ley los artículos de cada concepto" in seen and both != alone
    ws = load_workspace(both, cfg=svc.workspace_cfg())
    assert ws.step("segment")["detail"] == "99 artículos (inglés)"
    m_alone = load_manifest(load_workspace(alone, cfg=svc.workspace_cfg()))
    m_both = load_manifest(ws)
    assert m_both["summary"] == m_alone["summary"]                      # the law changes nothing
    assert m_both["triples"] == m_alone["triples"]
    rows_alone, rows_both = _csv(alone), _csv(both)
    without = (lambda rows: [{k: v for k, v in r.items() if k != "law_evidence"}  # noqa: E731
                             for r in rows])
    assert without(rows_both) == without(rows_alone)
    assert all(r["law_evidence"] == "" for r in rows_alone)
    assert all(r["law_evidence"].count("art. ") == 3 for r in rows_both)
    e = m_both["law_evidence"]
    assert (e["concepts"], e["with_evidence"], e["links"]) == (235, 235, 705)
    names = zipfile.ZipFile(io.BytesIO(build_user_package(ws, "turtle").data)).namelist()
    assert not any("evidence" in n for n in names) and CSV_FILE in names
    assert svc.view(both)["evidence"].startswith("Artículos de gdpr-en.pdf para revisar: 235 de "
                                                 "235 conceptos tienen artículos cercanos")
    assert _snapshot() == before                                         # base untouched