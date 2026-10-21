"""The "solo ley" mode: the concepts of an uploaded law become a profile that imports
OntoPriv-Core, and that profile is aligned with the DPV and packaged (S7-T10, decision D6).

The Sprint 3 split built OntoPriv-Core (jurisdiction-neutral) and one profile per law; the
LOPDP gave the Ecuador profile. A person who uploads another law gets its own profile, built
from the concepts the AI extracted and the rule accepted (S7-T09), without reviewing anything.
For each accepted concept, decision D6 asks whether it is already in OntoPriv-Core. The test is
the inheritance of S6-T05 against the validated base (same rules, thresholds and kinds):

- REUSED (same IRI or same name, label or alias as an entity of OntoPriv-Core): the profile
  does not declare a new entity; it reuses the IRI of the core and adds to it the label and
  the definition of the law (in its language) and its articles (``dct:source``). The core
  file itself is never changed: these triples live in the profile.
- LINKED ("muy parecido": combined score >= 0.75 or name >= 0.95): a new entity of the
  profile, ``skos:exactMatch`` the core entity (D6 b; a SKOS mapping, not an OWL axiom).
- NEW: anything else, including concepts that only match the Ecuador profile (a sibling
  profile, not imported): a new entity of the profile.

The profile (``<id>-profile``, IRI ``http://www.semanticweb.org/profiles/<workspace id>``) is
saved as the ontology of the workspace, marked as generated from the law, so the rest is the
flow of "Alinear mi ontología" (S7-T06, S7-T08): inherit what the base decided (a reused entity
inherits it by its IRI), the decider and its rule for the rest, labels es/en, the alignment
module and the zip, which also carries OntoPriv-Core so that the import resolves in Protégé.

The decider does not get the article of the law as evidence: measured in S7-T05 with the
chosen rule, it lowered the precision on the base from 75,0 % to 64,1 %.

CLI: ``py -m src.core.law_profile ARCHIVO [--formato turtle] [--sin-ia]`` runs the whole mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.core.emit import CORE_FILE, CORE_IRI

METHOD_VERSION = 1
PROFILE_BASE = "http://www.semanticweb.org/profiles/"
RESULT_FILE = "law-profile.json"
REUSED, LINKED, NEW = "reused", "linked", "new"
STATUS_LABELS = {REUSED: "reutiliza el IRI de OntoPriv-Core",
                 LINKED: "nuevo, enlazado a OntoPriv-Core (muy parecido)",
                 NEW: "nuevo en el perfil"}
_ENTITY_TYPES = {"class": OWL.Class, "property": OWL.ObjectProperty}
_ENTITY_KINDS = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property)


# --------------------------------------------------------------------------- the core

def core_path(cfg: dict | None = None) -> Path:
    """OntoPriv-Core as Sprint 3 wrote it (``<outputs>/ontopriv-core.rdf``)."""
    from src.alignment.inherit import base_paths
    return base_paths(cfg)["candidates"].parent / CORE_FILE


def load_core(cfg: dict | None = None) -> Graph:
    path = core_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"No está el núcleo OntoPriv-Core de la base: {path}")
    return Graph().parse(path)


def core_entities(core: Graph) -> set[str]:
    return {str(s) for t in _ENTITY_KINDS for s in core.subjects(RDF.type, t)
            if isinstance(s, URIRef)}


# --------------------------------------------------------------------------- D6

def law_concepts_as_sources(entries: list[dict]) -> list:
    """The accepted concepts as SourceConcepts (what the inheritance of S6-T05 compares)."""
    from src.alignment.sources import ORIGIN_AI, SourceConcept
    return [SourceConcept(key=f"law:{e['name']}", origin=ORIGIN_AI, name=e["name"],
                          kinds=(e["kind"],), label=e["label"] or None,
                          definition=e["definition"] or None, articles=list(e["articles"]))
            for e in entries if e["status"] == "accepted"]


def match_core(entries: list[dict], base, core_iris: set[str], *, embed_fn=None,
               cfg: dict | None = None) -> list[dict]:
    """D6 for every accepted concept: reused, linked or new (see the module docstring)."""
    from src.alignment.inherit import METHOD_SIMILAR, inherit
    sources = law_concepts_as_sources(entries)
    report = inherit(sources, base, embed_fn=embed_fn, cfg=cfg)
    found = {i.key: i for i in report.inherited}
    out = []
    for s in sources:
        i = found.get(s.key)
        in_core = i is not None and i.base_iri in core_iris      # never an Ecuador concept
        status = (NEW if not in_core else LINKED if i.method == METHOD_SIMILAR else REUSED)
        out.append({"name": s.name, "kind": s.kinds[0], "label": s.label,
                    "definition": s.definition, "articles": s.articles, "status": status,
                    "core_iri": i.base_iri if in_core else None,
                    "base_match": ({"iri": i.base_iri, "name": i.base_name, "method": i.method,
                                    "score": i.score, "origin": i.base_origin}
                                   if i is not None else None)})
    return out


# --------------------------------------------------------------------------- the graph

def profile_iri(ws) -> str:
    return PROFILE_BASE + ws.id


def article_source(law: str, number: int) -> str:
    return f"{law}, art. {number}"


def profile_graph(entries: list[dict], core: Graph, *, iri: str, law: str, language: str,
                  created: str) -> Graph:
    """The profile: header, the new and linked entities, the law's triples on reused ones."""
    ns = Namespace(iri + "#")
    g = Graph()
    g.bind("profile", ns)
    g.bind("skos", SKOS)
    g.bind("dct", DCTERMS)
    g.bind("core", Namespace(_namespace(core)))
    node = URIRef(iri)
    g.add((node, RDF.type, OWL.Ontology))
    g.add((node, OWL.imports, URIRef(CORE_IRI)))
    g.add((node, RDFS.label, Literal(f"Perfil de {law}", lang="es")))
    g.add((node, RDFS.comment, Literal(
        f"Perfil jurisdiccional construido con los conceptos que la IA extrajo de {law} y que "
        "la regla aceptó (S7-T09). Importa OntoPriv-Core: los conceptos que ya estaban en el "
        "núcleo reutilizan su IRI y los muy parecidos se enlazan con skos:exactMatch (S7-T10).",
        lang="es")))
    g.add((node, DCTERMS.created, Literal(created, datatype=XSD.dateTime)))
    g.add((node, DCTERMS.source, Literal(law)))
    g.add((node, OWL.versionInfo, Literal(str(METHOD_VERSION))))
    for e in entries:
        subject = URIRef(e["core_iri"]) if e["status"] == REUSED else ns[e["name"]]
        e["iri"] = str(subject)
        if e["status"] == REUSED:
            for t in core.objects(subject, RDF.type):          # the core's own declaration
                g.add((subject, RDF.type, t))
        else:
            g.add((subject, RDF.type, _ENTITY_TYPES[e["kind"]]))
        if e["status"] == LINKED:
            g.add((subject, SKOS.exactMatch, URIRef(e["core_iri"])))
        if e["label"]:
            g.add((subject, RDFS.label, Literal(e["label"], lang=language)))
        if e["definition"]:
            g.add((subject, SKOS.definition, Literal(e["definition"], lang=language)))
        for number in e["articles"]:
            g.add((subject, DCTERMS.source, Literal(article_source(law, number), lang="es")))
    return g


def _namespace(core: Graph) -> str:
    """The namespace of the OntoPriv entities (for a readable prefix)."""
    first = min(core_entities(core), default=CORE_IRI + "#")
    return first.rsplit("#", 1)[0] + "#"


def summarize(entries: list[dict]) -> dict:
    return {"concepts": len(entries),
            "reused": sum(e["status"] == REUSED for e in entries),
            "linked": sum(e["status"] == LINKED for e in entries),
            "new": sum(e["status"] == NEW for e in entries),
            "core_entities": len({e["core_iri"] for e in entries if e["status"] == REUSED}),
            "classes": sum(e["kind"] == "class" for e in entries),
            "properties": sum(e["kind"] == "property" for e in entries)}


# --------------------------------------------------------------------------- the workspace

STALE_AFTER_PROFILE = ("inherited.json", "alignment-decisions.json", "labels.json")


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def profile_for_workspace(ws, *, base=None, embed_fn=None, cfg: dict | None = None,
                          core: Graph | None = None) -> dict:
    """Build the profile from ``work/law-concepts.json``, save it as the ontology of the
    workspace and mark the step "profile". A profile with other content makes the results of
    the alignment steps stale (their caches stay)."""
    from src.alignment.inherit import load_base
    from src.core.law_concepts import load_result as law_concepts
    from src.ingest.law_upload import current_articles
    from src.workspace.store import DONE, ERROR, KIND_LAW, KIND_ONTOLOGY, RUNNING, WorkspaceError
    concepts = law_concepts(ws)
    if concepts is None:
        raise ValueError("Primero hay que extraer y aceptar los conceptos de la ley (S7-T09).")
    if not concepts["summary"]["accepted"]:
        why = ("la IA no estuvo disponible para extraerlos" if not concepts["ai"].get("available")
               else "la regla no aceptó ninguno")
        message = (f"No hay conceptos aceptados de la ley ({why}); no se puede armar el "
                   "perfil.")
        ws.mark_step("profile", ERROR, message)
        raise WorkspaceError(message)
    ws.mark_step("profile", RUNNING)
    try:
        base = base or load_base(cfg=cfg)
        core = core if core is not None else load_core(cfg)
        entries = match_core(concepts["entries"], base, core_entities(core), embed_fn=embed_fn,
                             cfg=cfg)
        law_file = ws.manifest["inputs"][KIND_LAW]["original_name"]
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        language = current_articles(ws)["language"]
        graph = profile_graph(entries, core, iri=profile_iri(ws), law=Path(law_file).stem,
                              language=language, created=created)
        data = graph.serialize(format="turtle").encode("utf-8")
        stored = f"{Path(ws.manifest['inputs'][KIND_LAW]['stored_as']).stem}-profile.ttl"
        old = ws.manifest["inputs"].get(KIND_ONTOLOGY) or {}
        sha = hashlib.sha256(_without_date(data)).hexdigest()
        if old.get("content_sha256") != sha or not (ws.input_dir / stored).exists():
            (ws.input_dir / stored).write_bytes(data)
            for name in STALE_AFTER_PROFILE:
                (ws.work_dir / name).unlink(missing_ok=True)
        ws.manifest["inputs"][KIND_ONTOLOGY] = {
            "original_name": f"Perfil de {law_file}", "stored_as": stored, "format": "Turtle",
            "generated_from": KIND_LAW, "content_sha256": sha}
        result = {"method_version": METHOD_VERSION, "profile_iri": profile_iri(ws),
                  "concepts_sha256": concepts_digest(concepts),
                  "core_iri": CORE_IRI, "law": law_file, "language": language,
                  "file": f"input/{stored}", "triples": len(graph),
                  "summary": summarize(entries), "entries": entries}
        (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    except Exception as exc:
        ws.mark_step("profile", ERROR, str(exc)[:300])
        raise
    s = result["summary"]
    ws.mark_step("profile", DONE, f"{s['reused']} reutilizan OntoPriv-Core, {s['linked']} "
                                  f"enlazados y {s['new']} nuevos", **{k: s[k] for k in
                                                                       ("reused", "linked",
                                                                        "new")})
    return result


def _without_date(data: bytes) -> bytes:
    """The Turtle without its creation date: what decides whether the profile changed."""
    return b"\n".join(line for line in data.splitlines() if b"dct:created" not in line)


def concepts_digest(concepts: dict) -> str:
    """SHA-256 of the accepted concepts (name, kind, label, definition, articles)."""
    accepted = [[e["name"], e["kind"], e["label"], e["definition"], e["articles"]]
                for e in concepts["entries"] if e["status"] == "accepted"]
    return hashlib.sha256(json.dumps(accepted, ensure_ascii=False).encode("utf-8")).hexdigest()


def is_current(result: dict | None, concepts: dict | None) -> bool:
    """The profile is current when this method built it from the same accepted concepts."""
    if not result or not concepts or result.get("method_version") != METHOD_VERSION:
        return False
    return result["concepts_sha256"] == concepts_digest(concepts)


# --------------------------------------------------------------------------- whole mode

def align_law(name: str, data: bytes, *, fmt: str = "turtle", use_llm: bool | None = None,
              embed_fn=None, llm=None, cfg: dict | None = None, base=None, targets=None,
              progress=None, chroma_path=None, sleep_fn=None, core: Graph | None = None):
    """Upload a law -> segment and index -> extract and accept -> profile -> the alignment
    flow of S7-T06/T08 -> zip. Returns the workspace, the manifest and the zip path."""
    from src.alignment.user_alignment import align_workspace
    from src.core.law_concepts import load_result as law_concepts, prepare_law_concepts
    from src.ingest.law_upload import prepare_law
    from src.workspace.store import STEP_LABELS, open_law
    ws = open_law(name, data, cfg=cfg)
    say = progress or (lambda label: None)
    prepare_law(ws, embed_fn=embed_fn, progress=say)
    prepare_law_concepts(ws, llm=llm, use_llm=use_llm, cfg=cfg, sleep_fn=sleep_fn, progress=say)
    if not is_current(load_result(ws), law_concepts(ws)):
        say(STEP_LABELS["profile"])
        profile_for_workspace(ws, base=base, embed_fn=embed_fn, cfg=cfg, core=core)
    return align_workspace(ws, fmt=fmt, use_llm=use_llm, embed_fn=embed_fn, llm=llm, cfg=cfg,
                           base=base, targets=targets, progress=progress,
                           chroma_path=chroma_path, sleep_fn=sleep_fn, core=core)


# --------------------------------------------------------------------------- console / CLI

def render_console(result: dict, examples: int = 6) -> str:
    s = result["summary"]
    lines = ["Perfil de la ley (S7-T10)",
             f"  {result['law']}: {s['concepts']} conceptos aceptados ({s['classes']} clases, "
             f"{s['properties']} propiedades)",
             f"    reutilizan el IRI de OntoPriv-Core: {s['reused']} (en {s['core_entities']} "
             f"entidades del núcleo)",
             f"    nuevos enlazados al núcleo (muy parecidos, skos:exactMatch): {s['linked']}",
             f"    nuevos en el perfil: {s['new']}",
             f"  Perfil: {result['file']} ({result['triples']} tripletas), importa "
             f"{result['core_iri']}"]
    for status in (REUSED, LINKED):
        shown = [e for e in result["entries"] if e["status"] == status][:examples]
        if shown:
            lines.append(f"  {STATUS_LABELS[status].capitalize()}:")
        for e in shown:
            m = e["base_match"]
            lines.append(f"    {e['name']} \"{e['label']}\" -> {m['name']} "
                         f"({m['method'].replace('_', ' ')}, {m['score']:.2f})")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.user_alignment import render_console as alignment_console
    from src.transform.convert import ConversionError, output_format
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(prog="py -m src.core.law_profile",
                                     description="Modo solo ley: perfil desde una ley subida, "
                                                 "alineado con el DPV y empaquetado (S7-T10).")
    parser.add_argument("file", help="ley en .pdf o .txt")
    parser.add_argument("--formato", default="turtle", help="rdfxml, owl, turtle o jsonld")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        output_format(args.formato)
        ws, manifest, zip_path = align_law(path.name, path.read_bytes(), fmt=args.formato,
                                           use_llm=False if args.sin_ia else None)
    except (WorkspaceError, ConversionError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(load_result(ws)))
    print(alignment_console(ws, manifest, zip_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())