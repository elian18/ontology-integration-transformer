"""S6-T09: end-to-end test of Sprint 6 on GDPRtEXT, an ontology from outside the base.

Runs what a user does in the web (``align_upload``, the same call the page makes) on the real
GDPRtEXT file and the real validated base, with a deterministic embedding and a fake AI, so it
is fast and needs no network. Real embeddings and the real AI were used in the web run whose
figures are in the CHANGELOG; this test locks what does not depend on them:
- the file is read and characterized (223 classes, 13 object properties, no OWL Full signal);
- it converts to every format and back without losing a triple (S6-T01/T02);
- 18 concepts inherit the validated decision by the "same name" rule (S6-T05), the same 18
  of the web run (no embeddings needed for that rule);
- the validated module only holds inherited correspondences and the zip resolves the import of
  GDPRtEXT in the four formats (S6-T07);
- the automatic check solves every inherited concept again and leaves the user's alignment as
  it was (S6-T09);
- nothing is written outside the workspace (S6-T04)."""
from __future__ import annotations

import hashlib
import io
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from rdflib import URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import OWL, RDF

from src.alignment.export import CANDIDATES_JSON
from src.alignment.external_check import check_workspace
from src.alignment.inherit import METHOD_SAME_NAME, load_base, load_result
from src.alignment.user_alignment import align_upload, build_user_package
from src.transform.convert import load_graph, parse_bytes, serialize
from src.workspace.store import BASE_DIRS, DONE
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_propose import FakeLLM

ROOT = Path(__file__).resolve().parents[1]
GDPRTEXT = ROOT / "data" / "input" / "gdprtext-ontology.ttl"
GDPR_IRI = "http://purl.org/adaptcentre/openscience/ontologies/GDPRtEXT"

pytestmark = pytest.mark.skipif(not GDPRTEXT.exists(),
                                reason="Falta data/input/gdprtext-ontology.ttl")


def _orthogonal_embed(texts):
    """Each distinct text gets its own direction: no semantic similarity at all."""
    out = []
    for t in texts:
        v = np.zeros(512)
        v[zlib.crc32(t.encode("utf-8")) % 512] = 1.0
        out.append(v)
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


def test_gdprtext_is_read_and_converts_without_loss():
    from src.ingest.ontology_loader import characterization_summary, load_ontology
    c = characterization_summary(load_ontology(GDPRTEXT))
    assert c["structure"]["classes"] == 223 and c["structure"]["object_properties"] == 13
    assert not c["is_owl_full"]
    g = load_graph(GDPRTEXT)
    for key in ("rdfxml", "turtle", "jsonld"):
        assert isomorphic(g, parse_bytes(serialize(g, key), key)), key


def test_end_to_end_on_gdprtext(tmp_path, real_base):
    from src.config import load_config
    before = _snapshot()
    real = load_config()
    cfg = {**real, "workspaces": {"dir": str(tmp_path / "ws")},
           "alignment": {**real["alignment"], "justify_pause_seconds": 0}}   # fake AI: no wait
    seen = []
    ws, manifest, zip_path = align_upload(
        GDPRTEXT.name, GDPRTEXT.read_bytes(), fmt="turtle", base=real_base,
        embed_fn=_orthogonal_embed, llm=FakeLLM(), cfg=cfg, progress=seen.append)
    assert len(seen) == 4 and all(s["status"] == DONE for s in ws.progress())

    inherited = load_result(ws)
    same_name = [i for i in inherited["inherited"] if i["method"] == METHOD_SAME_NAME]
    assert len(same_name) == 18
    names = {i["name"]: i["base_name"] for i in same_name}
    assert names["DataSubject"] == "Data_Owner" and names["Consent"] == "Consent"

    assert manifest["ontology_iri"] == GDPR_IRI and not manifest["ontology_iri_generated"]
    expected_mappings = sum(len(i["mappings"]) for i in inherited["inherited"])
    assert manifest["summary"]["validated_mappings"] == expected_mappings

    for fmt in ("rdfxml", "owl", "turtle", "jsonld"):
        folder = tmp_path / fmt
        zipfile.ZipFile(io.BytesIO(build_user_package(ws, fmt).data)).extractall(folder)
        g = _resolve(folder, manifest["modules"]["alignment"])
        assert (URIRef(GDPR_IRI + "#Consent"), RDF.type, OWL.Class) in g    # GDPRtEXT imported
    assert zip_path.exists()

    work = [ws.work_dir / CANDIDATES_JSON, ws.work_dir / "proposals.json"]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in work]
    check = check_workspace(ws, embed_fn=_orthogonal_embed, use_llm=False, cfg=cfg)
    a = check["summary"]["all"]
    assert a["concepts"] == len(inherited["inherited"])
    assert a["aligned"] + a["no_match"] == a["concepts"] and a["no_match"] >= 1  # Accountability
    assert {r["name"] for r in check["concepts"]} >= set(names)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in work] == hashes
    assert _snapshot() == before                                         # base untouched