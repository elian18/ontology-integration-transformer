"""S4-T07: the alignment candidates file (JSON + CSV) joining flows A and B."""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest.dpv_loader import load_dpv
from src.alignment.sources import (load_ai_concepts, AlignmentSources, SourceConcept,
                                   ORIGIN_ONTOLOGY, ORIGIN_AI)
from src.alignment.dpv_targets import build_dpv_targets, TARGET_PROPERTY
from src.alignment.lexical import normalize_for_lexical
from src.alignment.candidates import AlignmentSettings
from src.alignment.duplicates import STATUS_DUPLICATE, STATUS_NEW
from src.alignment.export import (
    build_candidate_table, write_candidate_files, load_candidate_file, render_console,
    justified_rows,
    COLUMNS, CANDIDATES_JSON, CANDIDATES_CSV, REVIEW_PENDING,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = AlignmentSettings(top_k=3, weight_lexical=0.4, weight_semantic=0.6)

_DPV = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:PersonalData a rdfs:Class , skos:Concept ; skos:prefLabel "Personal Data"@en ;
    skos:definition "Data directly or indirectly associated with an individual."@en .
dpv:DataSubject a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject"@en ;
    skos:definition "The individual whose personal data is being processed."@en .
dpv:DataController a rdfs:Class , skos:Concept ; skos:prefLabel "Data Controller"@en ;
    skos:definition "The individual or organisation that decides the purposes of processing."@en .
dpv:DataSubjectRight a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject Right"@en ;
    skos:definition "The rights exercisable by data subjects."@en .
dpv:SensitivePersonalData a rdfs:Class , skos:Concept ;
    skos:prefLabel "Sensitive Personal Data"@en ;
    skos:definition "Personal data considered sensitive."@en ; skos:broader dpv:PersonalData .
dpv:hasDataSubject a rdf:Property , skos:Concept ; skos:prefLabel "has data subject"@en ;
    skos:definition "Indicates the data subject."@en .
dpv:hasRecipient a rdf:Property , skos:Concept ; skos:prefLabel "has recipient"@en ;
    skos:definition "Indicates the recipient of data."@en .
dpv:hasEntity a rdf:Property , skos:Concept ; skos:prefLabel "has entity"@en ;
    skos:definition "Indicates an entity."@en .
"""


class CountingEmbed:
    """Deterministic bag-of-words embedding that counts how many times it is called."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            v = np.zeros(128)
            for tok in normalize_for_lexical(text).split():
                v[zlib.crc32(tok.encode()) % 128] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


def _onto(name, kinds=("class",), definition=None, family="Rights"):
    return SourceConcept(key=f"http://ex.org/lopdp#{name}", origin=ORIGIN_ONTOLOGY, name=name,
                         kinds=kinds, definition=definition, family=family, iri=None)


ONTO = [
    _onto("Data_subject", definition="natural person whose personal data is processed"),
    _onto("Sensitive_data", family="Terminology"),
    _onto("hasRecipient", kinds=("property",), family="(sin dominio)"),
]
AI = [
    {"name": "DataSubject", "label": "titular", "type": "class",
     "definition": "natural person whose personal data is processed", "articles": [4, 17]},
    {"name": "DigitalEducationRight", "label": "Derecho a la educación digital", "type": "class",
     "definition": "acceso al conocimiento sobre el uso adecuado de las tecnologías",
     "articles": [23]},
    {"name": "hasDataController", "label": "tiene responsable", "type": "property",
     "articles": [4]},
]


def _inputs(tmp_path):
    f = tmp_path / "mini-dpv.ttl"
    f.write_text(_DPV, encoding="utf-8")
    targets = build_dpv_targets(load_dpv(f))
    sources = AlignmentSources("memoria", "propuestas.json",
                               concepts=ONTO + load_ai_concepts({"concepts": AI}),
                               proposals_found=True)
    return sources, targets


def _table(tmp_path, embed=None):
    sources, targets = _inputs(tmp_path)
    return build_candidate_table(sources, targets, embed_fn=embed or CountingEmbed(),
                                 settings=SETTINGS, threshold=0.75, name_match=0.95)


# --- table ----------------------------------------------------------------------------------

def test_one_row_per_concept_and_candidate_without_repeats(tmp_path):
    table = _table(tmp_path)
    assert len(table.rows) == 6 * 3                       # 6 concepts x top_k 3
    pairs = [(r["concept_key"], r["dpv_iri"]) for r in table.rows]
    assert len(pairs) == len(set(pairs))
    assert all(set(r) == set(COLUMNS) for r in table.rows)
    ranks = {}
    for r in table.rows:
        ranks.setdefault(r["concept_key"], []).append(r["rank"])
    assert all(v == [1, 2, 3] for v in ranks.values())


def test_everything_pending_and_ai_columns_empty(tmp_path):
    table = _table(tmp_path)
    assert {r["review_status"] for r in table.rows} == {REVIEW_PENDING}
    assert all(r["proposed_relation"] is None and r["justification"] is None
               and r["evidence_article"] is None for r in table.rows)


def test_duplicate_mark_and_what_goes_to_the_ai(tmp_path):
    table = _table(tmp_path)
    first = {r["concept_name"]: r for r in table.rows if r["rank"] == 1}
    ds = first["DataSubject"]
    assert ds["origin"] == ORIGIN_AI and ds["duplicate_status"] == STATUS_DUPLICATE
    assert ds["duplicate_of_name"] == "Data_subject" and ds["duplicate_score"] >= 0.75
    assert ds["needs_justification"] is False                  # inherits OntoPriv's alignment
    new = first["DigitalEducationRight"]
    assert new["duplicate_status"] == STATUS_NEW and new["duplicate_of"] is None
    assert new["needs_justification"] is True
    onto = first["Data_subject"]
    assert onto["duplicate_status"] is None and onto["needs_justification"] is True
    c = table.counts()
    assert (c["ontology"], c["ai"], c["ai_possible_duplicates"], c["ai_new"]) == (3, 3, 1, 2)
    assert c["concepts_to_justify"] == 5 and c["rows_to_justify"] == 15


def test_rows_are_self_contained_and_kinds_respected(tmp_path):
    table = _table(tmp_path)
    ds = next(r for r in table.rows if r["concept_name"] == "DataSubject" and r["rank"] == 1)
    assert ds["dpv_name"] == "DataSubject"
    assert ds["dpv_definition"].startswith("The individual")
    assert ds["concept_articles"] == [4, 17] and ds["concept_label"] == "titular"
    spd = next(r for r in table.rows if r["concept_name"] == "Sensitive_data" and r["rank"] == 1)
    assert spd["dpv_name"] == "SensitivePersonalData"
    assert spd["dpv_parents"] == ["PersonalData"]                 # parents travel with the row
    props = [r for r in table.rows if r["concept_name"] in ("hasRecipient", "hasDataController")]
    assert props and all(r["dpv_kind"] == TARGET_PROPERTY for r in props)


def test_embeddings_are_computed_once(tmp_path):
    embed = CountingEmbed()
    _table(tmp_path, embed=embed)
    assert embed.calls == 1


def test_metadata_records_settings(tmp_path):
    table = _table(tmp_path)
    m = table.metadata
    assert m["settings"] == {"top_k": 3, "weight_lexical": 0.4, "weight_semantic": 0.6,
                             "duplicate_threshold": 0.75, "duplicate_name_match": 0.95}
    assert m["proposals_path"] == "propuestas.json" and m["generated_at"]


def test_without_ai_proposals(tmp_path):
    _, targets = _inputs(tmp_path)
    sources = AlignmentSources("memoria", None, concepts=list(ONTO))
    table = build_candidate_table(sources, targets, embed_fn=CountingEmbed(), settings=SETTINGS,
                                  threshold=0.75, name_match=0.95)
    assert table.counts()["ai"] == 0 and len(table.rows) == 9


# --- files ----------------------------------------------------------------------------------

def test_write_json_and_csv(tmp_path):
    table = _table(tmp_path)
    json_path, csv_path = write_candidate_files(table, tmp_path / "out")
    assert json_path.name == CANDIDATES_JSON and csv_path.name == CANDIDATES_CSV

    data = load_candidate_file(json_path)
    assert data["columns"] == COLUMNS
    assert len(data["rows"]) == len(table.rows)
    assert data["counts"] == table.counts()

    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert list(frame.columns) == COLUMNS
    assert len(frame) == len(table.rows)
    assert set(frame["review_status"]) == {REVIEW_PENDING}
    assert not frame.duplicated(subset=["concept_key", "dpv_iri"]).any()
    row = frame[(frame["concept_name"] == "DigitalEducationRight") & (frame["rank"] == 1)].iloc[0]
    assert row["concept_label"] == "Derecho a la educación digital"     # accents survive
    assert str(row["concept_articles"]) == "23"                         # pandas reads it as 23
    ds = frame[(frame["concept_name"] == "DataSubject") & (frame["rank"] == 1)].iloc[0]
    assert ds["concept_articles"] == "4|17"


def test_integer_columns_are_written_without_decimals(tmp_path):
    table = _table(tmp_path)
    table.rows[0]["evidence_article"] = 29                 # others stay empty (None)
    _, csv_path = write_candidate_files(table, tmp_path / "out")
    import csv
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["evidence_article"] == "29"               # not "29.0"
    assert rows[1]["evidence_article"] == ""                 # empty stays empty
    assert {r["rank"] for r in rows} == {"1", "2", "3"}      # not "1.0"


def test_justified_rows_protects_ai_work(tmp_path):
    table = _table(tmp_path)
    json_path, _ = write_candidate_files(table, tmp_path / "out")
    assert justified_rows(json_path) == 0
    table.rows[0]["proposed_relation"] = "skos:exactMatch"
    write_candidate_files(table, tmp_path / "out")
    assert justified_rows(json_path) == 1
    assert justified_rows(tmp_path / "no-existe.json") == 0


def test_console_summary(tmp_path):
    text = render_console(_table(tmp_path), "a.json", "a.csv")
    assert "Conceptos: 6 (OntoPriv 3, IA 3: 1 posibles duplicados, 2 nuevos)" in text
    assert "Filas: 18" in text and "a.csv" in text


@pytest.mark.slow
def test_real_project_files(tmp_path):
    """Whole run on the real OntoPriv + AI proposals + DPV (loads the embedding model)."""
    onto = ROOT / "data" / "input" / "ontopriv.rdf"
    dpv = ROOT / "vocab" / "dpv.ttl"
    if not onto.exists() or not dpv.exists():
        pytest.skip("Faltan data/input/ontopriv.rdf o vocab/dpv.ttl")
    from src.ingest.ontology_loader import load_ontology
    from src.alignment.sources import build_alignment_sources
    sources = build_alignment_sources(load_ontology(str(onto)))
    targets = build_dpv_targets(load_dpv(dpv))
    table = build_candidate_table(sources, targets)
    json_path, csv_path = write_candidate_files(table, tmp_path)
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert len(frame) == len(table.rows) == len(sources.concepts) * table.metadata["settings"]["top_k"]
    assert not frame.duplicated(subset=["concept_key", "dpv_iri"]).any()
    assert set(frame["review_status"]) == {REVIEW_PENDING}