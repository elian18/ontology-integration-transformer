"""Tests for the knowledge collections of the alignment decider (S7-T01).

Fast tests: a fake, deterministic embedder (no model), small hand-made DPV terms and base
concepts, and a temporary ChromaDB folder per test. One test builds the REAL collections from
the base files (DPV 2.3 + the Sprint 5 decisions) with the fake embedder, to lock the counts
1116 and 552; it is skipped if the base files are not on this machine.
"""
import hashlib

import numpy as np
import pytest

from src.ai.rag import vector_store
from src.alignment import knowledge as kn
from src.alignment.dpv_targets import DpvTarget, DpvTargets
from src.alignment.inherit import BaseConcept, BaseIndex


class FakeEmbed:
    """Hashed bag of words (32 dims, L2-normalized); counts how many texts it embedded."""

    def __init__(self):
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        out = []
        for t in texts:
            v = np.zeros(32)
            for w in str(t).lower().replace(".", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


@pytest.fixture
def targets():
    t = [DpvTarget("https://w3id.org/dpv#Consent", "Consent", "class", "Consent",
                   "Consent of the data subject for processing", ["LegalBasis"]),
         DpvTarget("https://w3id.org/dpv#DataSubjectRight", "DataSubjectRight", "class",
                   "Data Subject Right", "The rights applicable or provided to a data subject",
                   ["Right"]),
         DpvTarget("https://w3id.org/dpv#hasPurpose", "hasPurpose", "property", "has purpose",
                   "Indicates association with purpose", []),
         DpvTarget("https://w3id.org/dpv#Processing", "Processing", "class", "Processing",
                   None, [])]
    return DpvTargets(dpv_path="fake.ttl", targets=t)


@pytest.fixture
def base():
    concepts = [
        BaseConcept("onto#Consent", "onto#Consent", "ontology", "Consent", "consentimiento",
                    "Manifestación de voluntad del titular", ("class",), "aligned",
                    [{"dpv_iri": "https://w3id.org/dpv#Consent", "dpv_name": "Consent",
                      "relation": "skos:exactMatch"}]),
        BaseConcept("onto#Right_of_access", "onto#Right_of_access", "ontology",
                    "Right_of_access", "derecho de acceso", None, ("class", "property"),
                    "aligned",
                    [{"dpv_iri": "https://w3id.org/dpv#DataSubjectRight",
                      "dpv_name": "DataSubjectRight", "relation": "skos:broadMatch"},
                     {"dpv_iri": "https://w3id.org/dpv#hasPurpose", "dpv_name": "hasPurpose",
                      "relation": "skos:relatedMatch"}], aliases=["AccessRight"]),
        BaseConcept("ai:Sanction", "http://x/profile#Sanction", "ai", "Sanction", "sanción",
                    "Multa por infracción", ("class",), "no_match")]
    return BaseIndex(concepts=concepts, source_id="ontopriv+lopdp")


def _ensure(name, records, embed, fp, path, *, force=False, calls=None):
    def make():
        if calls is not None:
            calls.append(1)
        return records
    return kn.ensure_collection(name, fp, make, embed_fn=embed, model_name="fake-model",
                                path=path, force=force)


# --------------------------------------------------------------------------- records

def test_dpv_records_keep_the_text_the_ranking_embeds(targets):
    records = kn.dpv_records(targets)
    assert [r["id"] for r in records] == [t.iri for t in targets.targets]
    assert records[0]["document"] == targets.targets[0].semantic_text()
    meta = kn.parse_record_metadata(records[0]["metadata"])
    assert meta["kind"] == "class" and meta["parents"] == ["LegalBasis"]
    assert records[3]["metadata"]["definition"] == ""          # no None in ChromaDB metadata


def test_decision_records_carry_the_human_decision(base):
    records = {r["id"]: r for r in kn.decision_records(base)}
    access = kn.parse_record_metadata(records["onto#Right_of_access"]["metadata"])
    assert access["status"] == "aligned" and access["n_mappings"] == 2
    assert access["mappings"][0] == {"dpv_iri": "https://w3id.org/dpv#DataSubjectRight",
                                     "dpv_name": "DataSubjectRight",
                                     "relation": "skos:broadMatch"}
    assert access["is_class"] is True and access["is_property"] is True
    assert access["aliases"] == ["AccessRight"]
    sanction = kn.parse_record_metadata(records["ai:Sanction"]["metadata"])
    assert sanction["status"] == "no_match" and sanction["mappings"] == []
    assert sanction["is_class"] is True and sanction["is_property"] is False
    assert records["ai:Sanction"]["document"] == base.concepts[2].semantic_text()


# --------------------------------------------------------------------------- manifest

def test_fingerprint_depends_on_model_inputs_and_text_version(monkeypatch):
    fp = kn.fingerprint("model-a", "d1", "d2")
    assert fp == kn.fingerprint("model-a", "d1", "d2")
    assert fp != kn.fingerprint("model-b", "d1", "d2")
    assert fp != kn.fingerprint("model-a", "d1", "d3")
    monkeypatch.setattr(kn, "TEXT_VERSION", kn.TEXT_VERSION + 1)
    assert fp != kn.fingerprint("model-a", "d1", "d2")


def test_missing_collection_is_reported_as_missing(tmp_path):
    status = kn.collection_status("dpv", "fp1", path=tmp_path)
    assert status["state"] == kn.STATE_MISSING and status["items"] == 0


# --------------------------------------------------------------------------- building

def test_build_creates_a_cosine_collection_with_its_manifest(targets, tmp_path):
    embed = FakeEmbed()
    report = _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path)
    assert report["built"] is True and report["previous_state"] == kn.STATE_MISSING
    assert report["items"] == 4 and report["state"] == kn.STATE_UP_TO_DATE
    meta = vector_store.metadata("dpv", tmp_path)
    assert meta["hnsw:space"] == "cosine" and meta["fingerprint"] == "fp1"
    assert meta["items"] == 4 and meta["embedding_model"] == "fake-model"
    assert embed.texts == 4


def test_up_to_date_collection_is_not_rebuilt(targets, tmp_path):
    embed, calls = FakeEmbed(), []
    _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path, calls=calls)
    second = _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path, calls=calls)
    assert second["built"] is False and second["state"] == kn.STATE_UP_TO_DATE
    assert len(calls) == 1                       # the inputs were not even read again
    assert embed.texts == 4                      # no new embeddings


def test_changed_inputs_make_it_stale_and_it_is_rebuilt(targets, tmp_path):
    embed = FakeEmbed()
    _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path)
    assert kn.collection_status("dpv", "fp2", path=tmp_path)["state"] == kn.STATE_STALE
    again = _ensure("dpv", kn.dpv_records(targets), embed, "fp2", tmp_path)
    assert again["built"] is True and again["previous_state"] == kn.STATE_STALE
    assert vector_store.metadata("dpv", tmp_path)["fingerprint"] == "fp2"


def test_incomplete_collection_is_detected_and_rebuilt(targets, tmp_path):
    embed = FakeEmbed()
    _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path)
    vector_store.get_collection("dpv", tmp_path).delete(ids=["https://w3id.org/dpv#Consent"])
    status = kn.collection_status("dpv", "fp1", path=tmp_path)
    assert status["state"] == kn.STATE_INCOMPLETE
    assert (status["items"], status["expected_items"]) == (3, 4)
    again = _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path)
    assert again["built"] is True and again["items"] == 4


def test_force_rebuilds_an_up_to_date_collection(targets, tmp_path):
    embed = FakeEmbed()
    _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path)
    forced = _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path, force=True)
    assert forced["built"] is True and embed.texts == 8


def test_duplicate_ids_are_rejected(targets, tmp_path):
    records = kn.dpv_records(targets)
    with pytest.raises(ValueError, match="repetidos"):
        kn.build_collection("dpv", records + records[:1], embed_fn=FakeEmbed(),
                            fingerprint_value="fp", model_name="m", path=tmp_path)


def test_stored_vectors_find_the_term_itself_and_filter_by_kind(targets, tmp_path):
    embed = FakeEmbed()
    _ensure("dpv", kn.dpv_records(targets), embed, "fp1", tmp_path)
    query = embed([targets.targets[1].semantic_text()])
    hit = vector_store.query(query, 2, name="dpv", path=tmp_path)
    assert hit["ids"][0][0] == "https://w3id.org/dpv#DataSubjectRight"
    assert hit["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    props = vector_store.query(query, 4, name="dpv", path=tmp_path, where={"kind": "property"})
    assert props["ids"][0] == ["https://w3id.org/dpv#hasPurpose"]


def test_decisions_can_be_filtered_by_kind_and_status(base, tmp_path):
    _ensure("decisions", kn.decision_records(base), FakeEmbed(), "fp1", tmp_path)
    props = vector_store.get(name="decisions", path=tmp_path, where={"is_property": True})
    assert props["ids"] == ["onto#Right_of_access"]
    none = vector_store.get(name="decisions", path=tmp_path, where={"status": "no_match"})
    assert none["ids"] == ["ai:Sanction"]


# --------------------------------------------------------------------------- console

def test_console_is_in_spanish():
    built = {"name": "dpv", "items": 1116, "built": True, "previous_state": kn.STATE_MISSING,
             "state": kn.STATE_UP_TO_DATE, "embedding_model": "m"}
    kept = {"name": "decisions", "items": 552, "built": False,
            "previous_state": kn.STATE_UP_TO_DATE, "state": kn.STATE_UP_TO_DATE,
            "embedding_model": "m"}
    text = kn.render_console([built, kept])
    assert "1116 términos del DPV" in text and "construida ahora (antes: no existe)" in text
    assert "552 decisiones de la base" in text and "al día, no se tocó" in text
    status = {"name": "dpv", "items": 3, "expected_items": 4, "state": kn.STATE_INCOMPLETE,
              "embedding_model": None}
    assert "incompleta (3 de 4)" in kn.render_console([status])


# --------------------------------------------------------------------------- real base

def _base_files_present() -> bool:
    try:
        return all(p.exists() for files in kn.knowledge_inputs().values() for p in files)
    except Exception:
        return False


@pytest.mark.skipif(not _base_files_present(),
                    reason="faltan vocab/dpv.ttl o data/output/alignment-candidates.json")
def test_real_base_gives_1116_dpv_terms_and_552_decisions(tmp_path):
    embed = FakeEmbed()
    first = kn.build_knowledge(embed_fn=embed, model_name="fake-model", path=tmp_path)
    assert [(r["name"], r["items"], r["built"]) for r in first] == [
        ("dpv", 1116, True), ("decisions", 552, True)]
    kinds = [kn.parse_record_metadata(m)["kind"]
             for m in vector_store.get(name="dpv", path=tmp_path)["metadatas"]]
    assert kinds.count("class") == 972 and kinds.count("property") == 144
    aligned = vector_store.get(name="decisions", path=tmp_path, where={"status": "aligned"})
    assert len(aligned["ids"]) == 173
    second = kn.build_knowledge(embed_fn=embed, model_name="fake-model", path=tmp_path)
    assert [r["built"] for r in second] == [False, False]
    assert embed.texts == 1116 + 552
    status = kn.knowledge_status(model_name="fake-model", path=tmp_path)
    assert [s["state"] for s in status] == [kn.STATE_UP_TO_DATE, kn.STATE_UP_TO_DATE]
    other_model = kn.knowledge_status(model_name="another-model", path=tmp_path)
    assert [s["state"] for s in other_model] == [kn.STATE_STALE, kn.STATE_STALE]


@pytest.mark.skipif(not _base_files_present(),
                    reason="faltan vocab/dpv.ttl o data/output/alignment-candidates.json")
def test_cli_status_only_reports_and_builds_nothing(capsys):
    assert not vector_store.exists(kn.DPV_COLLECTION)          # isolated base database
    assert kn._run(["--estado"]) == 0
    out = capsys.readouterr().out
    assert "Colecciones de conocimiento para el decisor" in out
    assert out.count("no existe") == 2
    assert not vector_store.exists(kn.DPV_COLLECTION)          # --estado built nothing
    assert not vector_store.exists(kn.DECISIONS_COLLECTION)