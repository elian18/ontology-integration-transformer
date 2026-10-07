"""Write the aligned graph from the human decisions (S5-T06).

Input: the candidates file (Sprint 4) and the decisions log (S5-T01..T05, T08). Output, in the
outputs folder:
- ``ontopriv-dpv-alignment.rdf``: an OWL module that ``owl:imports`` the profile (which imports
  the core) and does NOT import the DPV (agreed in Sprint 5: core and profile stay untouched and
  the DPV is referenced by IRI only). It holds:
    1. the AI concepts a person approved as NEW (S5-T03), declared with the type the person
       chose, their Spanish label and definition and the law article they come from
       (``dct:source``); they are not placed under any OntoPriv class (agreed limitation);
    2. the law article of every AI concept confirmed as DUPLICATE, attached to the OntoPriv
       entity the person named (``same_as``), so that entity gains the law as evidence;
    3. one SKOS mapping triple per approved correspondence (S5-T04/T08):
       ``<concept> skos:exactMatch|closeMatch|broadMatch|narrowMatch|relatedMatch <dpv term>``.
       "No match" decisions write nothing. No OWL axiom points to the DPV (Sprint 4 decision).
- ``alignment-manifest.json``: IRIs, counts, who decided (person alone vs assistant proposal
  confirmed by a person), agreement with the AI type, the models behind the candidates and a
  SHA-256 of the decisions log, so the graph can be traced back to the exact decisions.
- ``alignment-approved.csv``: one row per approved correspondence, readable in Excel.

The graph is written only when every concept is decided (nothing "por validar"), so the module
never mixes decided and undecided content. The profile IRI, its name and the law name are
parameters: another law (Sprints 6-7) gets its own module with the same code.

Known consequence for the validation sprints: in SKOS the mapping properties have domain and
range ``skos:Concept``, so an RDFS/OWL-RL reasoner will also type the mapped entities as
``skos:Concept`` (punning, acceptable because OntoPriv is already OWL Full).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.core.emit import PROFILE_IRI
from src.alignment.dpv_targets import DPV_NAMESPACE
from src.alignment.sources import ORIGIN_AI
from src.alignment.decisions import MAPPING_RELATIONS
from src.alignment.ai_concepts import (
    ai_concepts, concept_decisions, approved_new_concepts, confirmed_duplicates, concept_progress,
)
from src.alignment.mapping_review import (
    mapping_concepts, mapping_decisions, mapping_progress, approved_mappings, ai_agreement,
)
from src.alignment.assistant_review import review_provenance, PROPOSED_BY

ALIGNMENT_FILE = "ontopriv-dpv-alignment.rdf"
ALIGNMENT_MANIFEST = "alignment-manifest.json"
ALIGNMENT_CSV = "alignment-approved.csv"
DEFAULT_PROFILE_NAME = "Ecuador (LOPDP)"
DEFAULT_LAW_NAME = "LOPDP"

DECIDED_BY_PERSON = "person"                      # the reviewer decided alone (card, S5-T05)
DECIDED_BY_ASSISTANT = "assistant_confirmed"      # assistant proposal confirmed by a person (S5-T08)

_ENTITY_TYPES = {"class": OWL.Class, "object_property": OWL.ObjectProperty,
                 "datatype_property": OWL.DatatypeProperty}
_SKOS = {"skos:exactMatch": SKOS.exactMatch, "skos:closeMatch": SKOS.closeMatch,
         "skos:broadMatch": SKOS.broadMatch, "skos:narrowMatch": SKOS.narrowMatch,
         "skos:relatedMatch": SKOS.relatedMatch}
_PREFIXES = {
    "dpv": DPV_NAMESPACE,
    "lopdp": "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#",
    "olopdp": "http://www.semanticweb.org/franc/ontologies/2023/9/OntologiaLOPDP#",
}

CSV_COLUMNS = [
    "concept_key", "subject_iri", "origin", "concept_name", "concept_family", "relation",
    "dpv_iri", "dpv_name", "ai_relation", "same_type_as_ai", "found_by", "decided_by",
    "reviewer", "decided_at", "evidence_article", "concept_articles", "note",
]


def alignment_iri(profile_iri: str) -> str:
    """IRI of the alignment module of a profile: ``<profile>-dpv-alignment``."""
    return profile_iri.rstrip("#/") + "-dpv-alignment"


def article_source(law_name: str, article: int) -> str:
    return f"{law_name}, art. {article}"


@dataclass
class AlignmentContent:
    """Everything the module, the manifest and the CSV are built from."""
    mappings: list[dict]
    new_concepts: list[dict]
    duplicates: list[dict]
    progress: dict
    concept_progress: dict
    provenance: dict
    agreement: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class AlignmentResult:
    rdf_path: str
    manifest_path: str
    csv_path: str
    alignment_iri: str
    profile_iri: str
    manifest: dict


# ------------------------------------------------------------------ gather
def collect(data: dict, records: list, source_id: str,
            profile_iri: str = PROFILE_IRI) -> AlignmentContent:
    """Read the decisions; refuses (in Spanish) while anything is still 'por validar'."""
    ai = ai_concepts(data, profile_iri)
    ai_state = concept_decisions(records, source_id)
    c_progress = concept_progress(ai, ai_state)
    if c_progress.get("pending"):
        raise ValueError(f"Faltan {c_progress['pending']} conceptos de la IA por decidir "
                         f"(pagina 'Aprobar conceptos IA'); el grafo se escribe cuando no "
                         f"queda nada por validar.")
    concepts = mapping_concepts(data, records, source_id, profile_iri)
    current = mapping_decisions(records, source_id)
    progress = mapping_progress(concepts, current)
    if progress["pending"]:
        raise ValueError(f"Faltan {progress['pending']} de {progress['total']} conceptos por "
                         f"validar en 'Alineacion DPV'; el grafo se escribe cuando no queda "
                         f"nada por validar.")
    family = {c.key: c.family for c in concepts}
    articles = {c.key: c.articles for c in concepts}
    mappings = []
    for m in approved_mappings(concepts, current):
        d = current[m["concept_key"]][m["dpv_iri"]]
        by = (DECIDED_BY_ASSISTANT if d.snapshot.get("proposed_by") == PROPOSED_BY
              else DECIDED_BY_PERSON)
        mappings.append({**m, "concept_family": family.get(m["concept_key"]),
                         "concept_articles": articles.get(m["concept_key"]) or [],
                         "decided_by": by,
                         "same_type_as_ai": (m["ai_relation"] == m["relation"]
                                             if m["ai_relation"] in MAPPING_RELATIONS else None)})
    return AlignmentContent(
        mappings=mappings,
        new_concepts=approved_new_concepts(ai, ai_state),
        duplicates=confirmed_duplicates(ai, ai_state),
        progress=progress, concept_progress=c_progress,
        provenance=review_provenance(concepts, current),
        agreement=ai_agreement(concepts, current),
        metadata=data.get("metadata") or {},
    )


# ------------------------------------------------------------------ the graph
def build_graph(content: AlignmentContent, profile_iri: str = PROFILE_IRI,
                profile_name: str = DEFAULT_PROFILE_NAME, law_name: str = DEFAULT_LAW_NAME,
                created: str | None = None) -> Graph:
    """The alignment module as an rdflib graph (see the module docstring)."""
    g = Graph()
    for prefix, ns in _PREFIXES.items():
        g.bind(prefix, Namespace(ns))
    g.bind("profile", Namespace(profile_iri.rstrip("#/") + "#"))
    g.bind("skos", SKOS)
    g.bind("dct", DCTERMS)

    onto = URIRef(alignment_iri(profile_iri))
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, OWL.imports, URIRef(profile_iri)))
    g.add((onto, RDFS.label, Literal(f"Alineación con el DPV del perfil {profile_name}",
                                     lang="es")))
    g.add((onto, RDFS.comment, Literal(
        "Correspondencias SKOS aprobadas por revisión humana entre OntoPriv (núcleo + perfil) "
        "y el Data Privacy Vocabulary (DPV 2.3), más los conceptos nuevos extraídos de la ley "
        "y aprobados por una persona. El DPV se referencia por IRI; no se importa.", lang="es")))
    g.add((onto, DCTERMS.references, URIRef(DPV_NAMESPACE.rstrip("#"))))
    if created:
        g.add((onto, DCTERMS.created, Literal(created)))

    for c in content.new_concepts:
        s = URIRef(c["iri"])
        g.add((s, RDF.type, _ENTITY_TYPES.get(c["entity_kind"], OWL.Class)))
        if c.get("label"):
            g.add((s, RDFS.label, Literal(c["label"], lang="es")))
        if c.get("definition"):
            g.add((s, SKOS.definition, Literal(c["definition"], lang="es")))
        for art in c.get("articles") or []:
            g.add((s, DCTERMS.source, Literal(article_source(law_name, art), lang="es")))

    for d in content.duplicates:
        if not d.get("duplicate_of"):
            continue
        for art in d.get("articles") or []:
            g.add((URIRef(d["duplicate_of"]), DCTERMS.source,
                   Literal(article_source(law_name, art), lang="es")))

    for m in content.mappings:
        g.add((URIRef(m["subject_iri"]), _SKOS[m["relation"]], URIRef(m["dpv_iri"])))
    return g


# ------------------------------------------------------------------ files
def _relative(path) -> str | None:
    """Path relative to the project root when it is inside it (portable between machines)."""
    if not path:
        return None
    from src.config import ROOT
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _sha256(path) -> str | None:
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_manifest(content: AlignmentContent, graph: Graph, *, profile_iri: str,
                   profile_name: str, law_name: str, source_id: str, log_path=None,
                   created: str | None = None) -> dict:
    by = Counter(m["decided_by"] for m in content.mappings)
    origin = Counter(m["origin"] for m in content.mappings)
    annotated = {d["duplicate_of"] for d in content.duplicates if d.get("duplicate_of")}
    meta = content.metadata
    return {
        "alignment_iri": alignment_iri(profile_iri),
        "alignment_file": ALIGNMENT_FILE,
        "imports": profile_iri,
        "imports_dpv": False,
        "dpv_namespace": DPV_NAMESPACE,
        "profile_name": profile_name,
        "law_name": law_name,
        "source_id": source_id,
        "created": created,
        "triples": len(graph),
        "concepts": {
            "reviewed": content.progress["reviewed"],
            "total": content.progress["total"],
            "aligned": content.progress["aligned"],
            "no_match": content.progress["no_match"],
            "pending": content.progress["pending"],
        },
        "mappings": {
            "total": len(content.mappings),
            "by_relation": content.progress["relations"],
            "from_ontopriv": origin.get("ontology", 0),
            "from_ai_concepts": origin.get(ORIGIN_AI, 0),
            "found_by_search": sum(m["found_by"] == "search" for m in content.mappings),
            "decided_by_person": by.get(DECIDED_BY_PERSON, 0),
            "assistant_confirmed": by.get(DECIDED_BY_ASSISTANT, 0),
        },
        "ai_concepts": {
            "proposed": content.concept_progress.get("total"),
            "new_declared": len(content.new_concepts),
            "new_by_kind": dict(Counter(c["entity_kind"] for c in content.new_concepts)),
            "duplicates": len(content.duplicates),
            "ontopriv_entities_with_law_articles": len(annotated),
            "rejected": content.concept_progress.get("rejected"),
        },
        "review_provenance": {k: v for k, v in content.provenance.items()
                              if k != "open_proposals"},
        "agreement_with_ai_type": content.agreement,
        "candidates": {
            "embedding_model": meta.get("embedding_model"),
            "llm_model": (meta.get("justification") or {}).get("llm_model"),
            "prompt_version": (meta.get("justification") or {}).get("prompt_version"),
            "settings": meta.get("settings"),
        },
        "decisions_log": {"path": _relative(log_path),
                          "sha256": _sha256(log_path)},
        "notes": [
            "Las correspondencias 'sin correspondencia' no generan tripletas.",
            "Los conceptos nuevos de la IA no se ubican en la jerarquia de OntoPriv.",
            "Con razonamiento RDFS/OWL-RL las propiedades de mapeo SKOS tipan sujeto y objeto "
            "como skos:Concept.",
        ],
    }


def _csv_value(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if value is None:
        return ""
    return value


def write_csv(mappings: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:   # BOM: Excel shows accents
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for m in mappings:
            w.writerow({c: _csv_value(m.get(c)) for c in CSV_COLUMNS})


def write_alignment(content: AlignmentContent, out_dir, *, profile_iri: str = PROFILE_IRI,
                    profile_name: str = DEFAULT_PROFILE_NAME, law_name: str = DEFAULT_LAW_NAME,
                    source_id: str = "", log_path=None,
                    created: str | None = None) -> AlignmentResult:
    """Write the RDF/XML module, the manifest and the CSV; the RDF is re-read as a check."""
    created = created or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph = build_graph(content, profile_iri, profile_name, law_name, created)
    rdf_path = out / ALIGNMENT_FILE
    graph.serialize(destination=str(rdf_path), format="xml")
    check = Graph().parse(str(rdf_path), format="xml")
    written = sum(1 for p in _SKOS.values() for _ in check.triples((None, p, None)))
    if written != len(content.mappings):
        raise ValueError(f"El archivo escrito tiene {written} correspondencias y se esperaban "
                         f"{len(content.mappings)}.")
    manifest = build_manifest(content, graph, profile_iri=profile_iri, profile_name=profile_name,
                              law_name=law_name, source_id=source_id, log_path=log_path,
                              created=created)
    manifest_path = out / ALIGNMENT_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    csv_path = out / ALIGNMENT_CSV
    write_csv(content.mappings, csv_path)
    return AlignmentResult(rdf_path=str(rdf_path), manifest_path=str(manifest_path),
                           csv_path=str(csv_path), alignment_iri=alignment_iri(profile_iri),
                           profile_iri=profile_iri, manifest=manifest)


def render_console(result: AlignmentResult) -> str:
    m = result.manifest
    c, mp, ai, prov = m["concepts"], m["mappings"], m["ai_concepts"], m["review_provenance"]
    agr = m["agreement_with_ai_type"]
    lines = [
        "Grafo alineado con el DPV (RDF/XML):",
        f"  Modulo:  {result.rdf_path}  ({m['triples']} tripletas)",
        f"  IRI:     {result.alignment_iri}",
        f"  Importa: {result.profile_iri} (el perfil, que importa el nucleo); el DPV no se importa",
        f"  Conceptos revisados: {c['reviewed']} de {c['total']} "
        f"({c['aligned']} alineados, {c['no_match']} sin correspondencia)",
        f"  Correspondencias SKOS: {mp['total']} "
        f"({mp['from_ontopriv']} de OntoPriv, {mp['from_ai_concepts']} de conceptos nuevos)",
    ]
    for rel, n in mp["by_relation"].items():
        lines.append(f"    - {rel}: {n}")
    lines += [
        f"  Conceptos nuevos de la IA declarados: {ai['new_declared']}",
        f"  Entidades de OntoPriv con articulos de la ley (duplicados): "
        f"{ai['ontopriv_entities_with_law_articles']}",
        f"  Quien decidio: {prov['reviewed_by_person']} conceptos la persona sola, "
        f"{prov['confirmed_from_assistant']} confirmados desde propuestas del asistente "
        f"({prov['changed_by_person']} con cambios)",
    ]
    if agr.get("type_agreement") is not None:
        lines.append(f"  Tipo SKOS igual al de la IA: {agr['same_type_as_ai']} de "
                     f"{agr['approved_with_ai_type']} ({agr['type_agreement'] * 100:.1f} %)")
    lines += [f"  Manifiesto: {result.manifest_path}", f"  CSV:        {result.csv_path}"]
    return "\n".join(lines)


def _run() -> None:
    from src.config import load_config, ROOT
    from src.alignment.export import load_candidate_file, CANDIDATES_JSON
    from src.alignment.decisions import review_settings, decisions_path, read_log

    cfg = load_config() or {}
    out_dir = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    cand = out_dir / CANDIDATES_JSON
    if not cand.exists():
        print(f"No hay archivo de candidatos ({cand}). Corre antes: py -m src.alignment.export")
        return
    source_id = review_settings(cfg)["source_id"]
    log = decisions_path(cfg)
    try:
        content = collect(load_candidate_file(cand), read_log(log), source_id)
        result = write_alignment(content, out_dir, source_id=source_id, log_path=log)
    except ValueError as exc:
        print(f"No se escribio el grafo: {exc}")
        return
    print(render_console(result))


if __name__ == "__main__":
    _run()