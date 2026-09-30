"""S4-T11: the whole Sprint 4 alignment pipeline end to end, with fakes (no model, no LLM).

OntoPriv-like ontology + AI proposals + DPV -> sources -> DPV targets -> candidates (lexical +
embeddings, duplicates) -> JSON/CSV -> AI type + justification -> measurement -> web service.
"""
import csv
import json
import re
import zlib
from types import SimpleNamespace

import numpy as np
import rdflib

from src.ingest.dpv_loader import load_dpv
from src.alignment.sources import build_alignment_sources, ORIGIN_AI
from src.alignment.dpv_targets import build_dpv_targets
from src.alignment.lexical import normalize_for_lexical
from src.alignment.candidates import AlignmentSettings
from src.alignment.export import (build_candidate_table, write_candidate_files,
                                  load_candidate_file, REVIEW_PENDING, COLUMNS)
from src.alignment.justify import justify_candidates, JustifySettings
from src.alignment.evaluate import (make_reference_sample, write_reference_sample,
                                    read_reference_sample, evaluate)
from app.services import alignment as alignment_service

ONTOLOGY = """
@prefix : <http://example.org/lopdp#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:Law a owl:Class .
:Rights a owl:Class ; rdfs:subClassOf :Law .
:Data_subject a owl:Class ; rdfs:subClassOf :Law ;
    rdfs:comment "natural person whose personal data is processed" .
:Data_Controller a owl:Class ; rdfs:subClassOf :Law .
:Right_to_portability a owl:Class ; rdfs:subClassOf :Rights .
:hasRecipient a owl:ObjectProperty .
:banking001 a owl:NamedIndividual , :Law .
"""

DPV = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:DataSubject a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject"@en ;
    skos:definition "The individual whose personal data is being processed."@en .
dpv:DataController a rdfs:Class , skos:Concept ; skos:prefLabel "Data Controller"@en ;
    skos:definition "Decides the purposes of processing."@en .
dpv:DataSubjectRight a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject Right"@en ;
    skos:definition "The rights exercisable by data subjects."@en .
dpv:DataErasurePolicy a rdfs:Class , skos:Concept ; skos:prefLabel "Data Erasure Policy"@en ;
    skos:definition "Policy regarding erasure of data."@en .
dpv:hasRecipient a rdf:Property , skos:Concept ; skos:prefLabel "has recipient"@en ;
    skos:definition "Indicates the recipient of data."@en .
dpv:hasRight a rdf:Property , skos:Concept ; skos:prefLabel "has right"@en ;
    skos:definition "Indicates a right."@en .
"""

PROPOSALS = {"concepts": [
    {"name": "DataSubject", "label": "titular", "type": "class",
     "definition": "natural person whose personal data is processed", "articles": [4]},
    # declared as property by the extraction, named like a class (the S4-T09 finding):
    {"name": "RightToErasure", "label": "Derecho de eliminacion", "type": "property",
     "definition": "right to erasure of personal data", "articles": [15]},
    {"name": "hasPurpose", "label": "tiene finalidad", "type": "property", "articles": [10]},
]}

ARTICLES = {n: {"number": n, "title": f"Articulo {n}", "text": f"Texto del articulo {n}."}
            for n in (4, 10, 15)}


def fake_embed(texts):
    out = []
    for text in texts:
        v = np.zeros(128)
        for tok in normalize_for_lexical(text).split():
            v[zlib.crc32(tok.encode()) % 128] += 1.0
        n = np.linalg.norm(v)
        out.append((v / n if n else v).tolist())
    return out


class FakeLLM:
    """exactMatch when the names match after normalization, relatedMatch otherwise."""

    def __init__(self):
        self.calls = 0

    def ask(self, prompt, context="", max_tokens=1024):
        self.calls += 1
        lines = []
        for block in context.split("=== ")[1:]:
            cid = block.split(" ===", 1)[0]
            concept = re.search(r"Concepto: (\S+)", block).group(1)
            for dpv in re.findall(r"^- (\S+) \|", block, flags=re.M):
                same = normalize_for_lexical(concept) == normalize_for_lexical(dpv)
                lines.append(json.dumps({"concept": cid, "dpv": dpv,
                                         "relation": "exactMatch" if same else "relatedMatch",
                                         "justification": "Prueba."}))
        return "\n".join(lines)


def test_sprint4_pipeline_end_to_end(tmp_path):
    # 1. sources (flow A + flow B) and DPV targets
    graph = rdflib.Graph().parse(data=ONTOLOGY, format="turtle")
    report = SimpleNamespace(graph=graph, path="mini.ttl", source_format="turtle",
                             n_triples=len(graph))
    sources = build_alignment_sources(report, proposals=PROPOSALS)
    counts = sources.counts()
    assert counts["ontology"] == 6 and counts["ai"] == 3
    assert counts["ai_kind_corrected"] == 1 and counts["individuals_skipped"] == 1
    dpv_file = tmp_path / "dpv.ttl"
    dpv_file.write_text(DPV, encoding="utf-8")
    targets = build_dpv_targets(load_dpv(dpv_file))
    assert targets.counts()["properties"] == 2

    # 2. candidates file (ranking + duplicates), everything pending
    table = build_candidate_table(sources, targets, embed_fn=fake_embed,
                                  settings=AlignmentSettings(top_k=2), threshold=0.75,
                                  name_match=0.95)
    out = tmp_path / "output"
    json_path, csv_path = write_candidate_files(table, out)
    data = load_candidate_file(json_path)
    assert {r["review_status"] for r in data["rows"]} == {REVIEW_PENDING}
    first = {r["concept_name"]: r for r in data["rows"] if r["rank"] == 1}
    assert first["Data_Controller"]["dpv_name"] == "DataController"
    assert first["DataSubject"]["duplicate_status"] == "possible_duplicate"   # = Data_subject
    assert first["RightToErasure"]["dpv_kind"] == "class"       # kind corrected by the name
    assert first["hasPurpose"]["dpv_kind"] == "property"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        assert list(csv.DictReader(f).fieldnames) == COLUMNS

    # 3. the AI proposes the type (fake LLM); duplicates are not sent
    llm = FakeLLM()
    report_j = justify_candidates(data, llm=llm, article_lookup=ARTICLES.get,
                                  retrieve_fn=lambda q: [ARTICLES[4]],
                                  settings=JustifySettings(batch_size=10, pause_seconds=0),
                                  out_dir=out, sleep_fn=lambda s: None)
    assert report_j.remaining_concepts == 0 and llm.calls == 1
    saved = load_candidate_file(json_path)
    typed = [r for r in saved["rows"] if r["proposed_relation"]]
    assert typed and all(r["needs_justification"] for r in typed)
    assert saved["metadata"]["justification"]["prompt_version"] == 2

    # 4. measurement against a (here automatic) reference sample
    sample = make_reference_sample(saved, size=4, ai_share=0.5, seed=1)
    for rec in sample:
        rec["gold_dpv_name"] = rec["cand1_dpv"]
        rec["gold_relation"] = "exactMatch"
    ref_path = write_reference_sample(sample, tmp_path / "reference" / "sample.csv")
    metrics = evaluate(saved, read_reference_sample(ref_path))
    assert metrics.labelled == 4 and metrics.hit_at_1 == 4

    # 5. the web service reads what the pipeline wrote
    view = alignment_service.candidates(path=json_path)
    assert len(view["rows"]) == len(saved["rows"])
    ai_new = alignment_service.filter_rows(view["rows"], origin=ORIGIN_AI, duplicate="new")
    assert {r["concept_name"] for r in ai_new} == {"RightToErasure", "hasPurpose"}