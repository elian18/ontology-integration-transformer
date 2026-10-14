"""Write the aligned ontology of an upload and package it in the chosen format (S6-T07).

Closes the back-end of the "solo ontología" mode: upload -> inherit (S6-T05) -> propose
(S6-T06) -> write + package (this module). Everything is written inside the workspace
(``data/workspaces/<id>/output``); nothing of the base is touched.

What is written (same design as Sprint 5: the user's ontology is never edited, the alignment
lives in separate modules that import it):

1. ``<name>-dpv-alignment``: the VALIDATED module. One SKOS triple per correspondence inherited
   from the base (a person decided it in Sprint 5). Each triple carries an ``owl:Axiom``
   annotation with where it comes from: ``dct:source`` = the base concept and an
   ``rdfs:comment`` such as "Heredada de Data_Owner (mismo nombre); decisión validada por
   Elian Gavilanes el 2026-10-07". "Sin correspondencia" writes nothing (as in Sprint 5).
2. ``<name>-dpv-proposals``: the AUTOMATIC proposals, apart, never mixed with the validated
   module. Typed by the AI -> its SKOS property; untyped -> ``rdfs:seeAlso`` (claims no SKOS
   relation). Each one is annotated "Propuesta automática, no validada", with its confidence
   score (an annotation property declared in the module itself) and the AI's justification.
   Not written when there are no proposals.
3. ``alignment-manifest.json`` (counts, settings, models, base used) and ``alignment.csv``
   (one row per concept of the user's ontology with what happened to it).

Both modules ``owl:imports`` the user's ontology by its IRI and reference the DPV by IRI only.
If the uploaded ontology declares no ``owl:Ontology`` IRI, one is generated
(``http://www.semanticweb.org/uploads/<workspace id>``) and added to a copy of it in
``output/``, because an import needs an IRI; the manifest says so.

The package (``build_user_package``) holds the three modules converted to the chosen format,
the catalog for Protégé, the manifest, the CSV and a LEEME.txt; the JSON-LD one carries the
Protégé note measured in S6-T03.

CLI: ``py -m src.alignment.user_alignment ARCHIVO [--formato turtle] [--sin-ia]`` runs the
whole back-end for one file and writes the zip into the workspace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.alignment.bundle import PROTEGE_JSONLD_NOTE
from src.alignment.inherit import METHOD_LABELS, load_result as inherited_result
from src.alignment.propose import (STATUS_AI_NONE, STATUS_BELOW, STATUS_PROPOSED,
                                   load_result as proposals_result)
from src.transform.convert import OutputFormat, load_graph, output_format, serialize
from src.transform.package import CATALOG_FILE, Module, Package, package_modules

DPV = Namespace("https://w3id.org/dpv#")
GENERATED_IRI_BASE = "http://www.semanticweb.org/uploads/"
MANIFEST_FILE = "alignment-manifest.json"
CSV_FILE = "alignment.csv"
MASTER_FORMAT = "turtle"            # modules kept in output/ as Turtle (readable); zips convert

_SKOS = {"skos:exactMatch": SKOS.exactMatch, "skos:closeMatch": SKOS.closeMatch,
         "skos:broadMatch": SKOS.broadMatch, "skos:narrowMatch": SKOS.narrowMatch,
         "skos:relatedMatch": SKOS.relatedMatch}

# Status of every concept of the user's ontology (CSV, manifest, web).
INHERITED_ALIGNED = "inherited_aligned"
INHERITED_NO_MATCH = "inherited_no_match"
PROPOSED = "proposed"
NO_PROPOSAL = "no_proposal"
AI_REJECTED = "ai_rejected"
NOT_ALIGNABLE = "not_alignable"
STATUS_LABELS = {INHERITED_ALIGNED: "validada (heredada de la base)",
                 INHERITED_NO_MATCH: "sin correspondencia (heredada de la base)",
                 PROPOSED: "propuesta automática, no validada",
                 NO_PROPOSAL: "sin propuesta (bajo el umbral)",
                 AI_REJECTED: "sin propuesta (la IA la descartó)",
                 NOT_ALIGNABLE: "no se alinea (individuo)"}
CSV_COLUMNS = ["concept_iri", "concept_name", "status", "estado", "validated", "relation",
               "dpv_iri", "dpv_name", "score", "method", "base_concept", "decided_by",
               "justification"]


# --------------------------------------------------------------------------- names and IRIs

def stems(ws) -> dict[str, str]:
    """File stems inside the package: the upload's name plus the two module suffixes."""
    base = Path(ws.manifest["inputs"]["ontology"]["stored_as"]).stem
    return {"ontology": base, "alignment": f"{base}-dpv-alignment",
            "proposals": f"{base}-dpv-proposals"}


def ontology_iri(graph: Graph) -> str | None:
    iris = sorted(str(s) for s in graph.subjects(RDF.type, OWL.Ontology)
                  if isinstance(s, URIRef))
    return iris[0] if iris else None


def module_iri(onto_iri: str, suffix: str) -> str:
    return onto_iri.rstrip("#/") + suffix


def _bind(g: Graph, source: Graph) -> None:
    for prefix, ns in source.namespaces():
        if prefix and prefix not in ("brick", "csvw", "doap", "geo", "odrl", "org", "prof",
                                     "qb", "schema", "sh", "sosa", "ssn", "time", "vann",
                                     "void", "wgs", "dcam", "dcat", "dcmitype"):
            g.bind(prefix, ns, override=False)
    for prefix, ns in (("dpv", DPV), ("skos", SKOS), ("dct", DCTERMS), ("owl", OWL)):
        g.bind(prefix, ns, override=True)


def _header(g: Graph, iri: str, imports: str, label: str, comment: str, created: str) -> None:
    node = URIRef(iri)
    g.add((node, RDF.type, OWL.Ontology))
    g.add((node, OWL.imports, URIRef(imports)))
    g.add((node, RDFS.label, Literal(label, lang="es")))
    g.add((node, RDFS.comment, Literal(comment, lang="es")))
    g.add((node, DCTERMS.created, Literal(created, datatype=XSD.dateTime)))


def _annotate(g: Graph, s, p, o, comment: str, extra: list[tuple] = ()) -> None:
    ax = BNode()
    g.add((ax, RDF.type, OWL.Axiom))
    g.add((ax, OWL.annotatedSource, s))
    g.add((ax, OWL.annotatedProperty, p))
    g.add((ax, OWL.annotatedTarget, o))
    g.add((ax, RDFS.comment, Literal(comment, lang="es")))
    for pred, obj in extra:
        g.add((ax, pred, obj))


# --------------------------------------------------------------------------- the graphs

def validated_graph(inherited: dict, *, iri: str, onto_iri: str, name: str, created: str,
                    source: Graph | None = None) -> Graph:
    g = Graph()
    if source is not None:
        _bind(g, source)
    _header(g, iri, onto_iri, f"Alineación validada de {name} con el DPV",
            "Correspondencias heredadas de la base del sistema (OntoPriv + LOPDP), decididas "
            "por una persona en el Sprint 5. El DPV se referencia por IRI, no se importa.",
            created)
    for item in inherited.get("inherited", []):
        subject = URIRef(item["key"])
        for m in item.get("mappings", []):
            prop = _SKOS[m["relation"]]
            target = URIRef(m["dpv_iri"])
            g.add((subject, prop, target))
            who = m.get("reviewer") or "una persona"
            when = (m.get("decided_at") or "")[:10]
            _annotate(g, subject, prop, target,
                      f"Heredada de {item['base_name']} ({METHOD_LABELS[item['method']]}); "
                      f"decisión validada por {who}" + (f" el {when}" if when else "") + ".",
                      [(DCTERMS.source, URIRef(item["base_iri"]))])
    return g


def proposals_graph(proposals: dict, *, iri: str, onto_iri: str, name: str, created: str,
                    source: Graph | None = None) -> Graph | None:
    chosen = [p for p in proposals.get("proposals", []) if p["status"] == STATUS_PROPOSED]
    if not chosen:
        return None
    g = Graph()
    if source is not None:
        _bind(g, source)
    _header(g, iri, onto_iri, f"Propuestas automáticas para {name} (no validadas)",
            "Correspondencias propuestas por el sistema sin revisión humana: el mejor "
            "candidato del DPV con puntaje sobre el umbral calibrado y, si la IA estuvo "
            "disponible, su tipo SKOS. Las que no tienen tipo usan rdfs:seeAlso.", created)
    conf = URIRef(iri.rstrip("#/") + "#confidence")
    g.add((conf, RDF.type, OWL.AnnotationProperty))
    g.add((conf, RDFS.label, Literal("confianza de la propuesta automática", lang="es")))
    g.add((conf, RDFS.comment, Literal("Puntaje de 0 a 1 del candidato (0.4 léxico + 0.6 "
                                       "embeddings) usado para proponerlo.", lang="es")))
    for p in chosen:
        subject, target = URIRef(p["key"]), URIRef(p["dpv_iri"])
        prop = _SKOS.get(p.get("relation") or "", RDFS.seeAlso)
        g.add((subject, prop, target))
        text = "Propuesta automática, no validada."
        if p.get("justification"):
            text += f" Justificación de la IA: {p['justification']}"
        _annotate(g, subject, prop, target, text,
                  [(conf, Literal(f"{p['score']:.4f}", datatype=XSD.decimal))])
    return g


# --------------------------------------------------------------------------- rows

def concept_rows(inherited: dict, proposals: dict | None) -> list[dict]:
    """One row per concept of the user's ontology (aligned ones: one row per mapping)."""
    rows = []
    for item in inherited.get("inherited", []):
        common = {"concept_iri": item["key"], "concept_name": item["name"],
                  "method": METHOD_LABELS[item["method"]], "base_concept": item["base_iri"],
                  "validated": True}
        if item["status"] == "aligned" and item.get("mappings"):
            for m in item["mappings"]:
                rows.append({**common, "status": INHERITED_ALIGNED, "relation": m["relation"],
                             "dpv_iri": m["dpv_iri"], "dpv_name": m.get("dpv_name"),
                             "decided_by": m.get("reviewer")})
        else:
            rows.append({**common, "status": INHERITED_NO_MATCH})
    by_status = {STATUS_PROPOSED: PROPOSED, STATUS_BELOW: NO_PROPOSAL,
                 STATUS_AI_NONE: AI_REJECTED}
    for p in (proposals or {}).get("proposals", []):
        status = by_status[p["status"]]
        row = {"concept_iri": p["key"], "concept_name": p["name"], "status": status,
               "validated": False, "score": p["score"], "justification": p.get("justification")}
        if status == PROPOSED:
            row.update(relation=p.get("relation") or "rdfs:seeAlso", dpv_iri=p["dpv_iri"],
                       dpv_name=p.get("dpv_name"))
        rows.append(row)
    for n in inherited.get("not_inherited", []):
        if n.get("reason") == "kind":
            rows.append({"concept_iri": n["key"], "concept_name": n["name"],
                         "status": NOT_ALIGNABLE, "validated": False})
    for r in rows:
        r["estado"] = STATUS_LABELS[r["status"]]
    return sorted(rows, key=lambda r: (list(STATUS_LABELS).index(r["status"]),
                                       (r.get("concept_name") or "").lower()))


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:   # BOM: Excel accents
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})


def summarize(rows: list[dict]) -> dict:
    concepts: dict[str, str] = {}
    for r in rows:
        concepts.setdefault(r["concept_iri"], r["status"])
    counts = {s: sum(1 for v in concepts.values() if v == s) for s in STATUS_LABELS}
    return {"concepts": len(concepts), "by_status": counts,
            "validated_mappings": sum(1 for r in rows if r["status"] == INHERITED_ALIGNED),
            "proposed_mappings": sum(1 for r in rows if r["status"] == PROPOSED),
            "proposed_untyped": sum(1 for r in rows if r["status"] == PROPOSED
                                    and r.get("relation") == "rdfs:seeAlso")}


# --------------------------------------------------------------------------- writing

def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_user_alignment(ws, cfg: dict | None = None) -> dict:
    """Write both modules (Turtle), the manifest and the CSV into ``output/``; mark the step."""
    from src.alignment.inherit import base_paths
    from src.workspace.store import DONE, ERROR, RUNNING
    inherited = inherited_result(ws)
    if inherited is None:
        raise ValueError("Primero hay que heredar lo validado (paso S6-T05).")
    proposals = proposals_result(ws)
    if proposals is None:
        raise ValueError("Primero hay que calcular las propuestas automáticas (paso S6-T06).")
    ws.mark_step("write", RUNNING)
    try:
        out = ws.output_dir
        out.mkdir(parents=True, exist_ok=True)
        names = stems(ws)
        source_path = ws.input_path()
        user_graph = load_graph(source_path)
        onto_iri = ontology_iri(user_graph)
        generated = onto_iri is None
        ontology_file = source_path
        if generated:                                  # an import needs an IRI
            onto_iri = GENERATED_IRI_BASE + ws.id
            user_graph.add((URIRef(onto_iri), RDF.type, OWL.Ontology))
            ontology_file = out / f"{names['ontology']}.ttl"
            ontology_file.write_bytes(serialize(user_graph, MASTER_FORMAT))
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        display = ws.manifest["inputs"]["ontology"]["original_name"]
        iris = {"alignment": module_iri(onto_iri, "-dpv-alignment"),
                "proposals": module_iri(onto_iri, "-dpv-proposals")}
        files = {"ontology": ontology_file.relative_to(ws.path).as_posix()}

        validated = validated_graph(inherited, iri=iris["alignment"], onto_iri=onto_iri,
                                    name=display, created=created, source=user_graph)
        (out / f"{names['alignment']}.ttl").write_bytes(serialize(validated, MASTER_FORMAT))
        files["alignment"] = f"output/{names['alignment']}.ttl"
        auto = proposals_graph(proposals, iri=iris["proposals"], onto_iri=onto_iri,
                               name=display, created=created, source=user_graph)
        stale = out / f"{names['proposals']}.ttl"
        if auto is not None:
            stale.write_bytes(serialize(auto, MASTER_FORMAT))
            files["proposals"] = f"output/{names['proposals']}.ttl"
        elif stale.exists():
            stale.unlink()                             # an earlier run had proposals

        rows = concept_rows(inherited, proposals)
        write_csv(rows, out / CSV_FILE)
        paths = base_paths(cfg)
        manifest = {
            "workspace": ws.id, "source_id": ws.source_id, "created": created,
            "uploaded_file": display, "ontology_iri": onto_iri,
            "ontology_iri_generated": generated,
            "modules": {"alignment": iris["alignment"],
                        "proposals": iris["proposals"] if auto is not None else None},
            "files": files,
            "triples": {"alignment": len(validated), "proposals": len(auto) if auto else 0},
            "summary": summarize(rows),
            "inheritance": inherited.get("summary"),
            "proposals": proposals.get("summary"),
            "settings": {"inheritance": inherited.get("settings"),
                         "embedding_model": os.environ.get("EMBEDDING_MODEL"),
                         "llm_model": os.environ.get("LLM_MODEL")},
            "base": {"source_id": (inherited.get("settings") or {}).get("base_source_id"),
                     "decisions_log_sha256": _sha256(paths["log"])},
            "notes": [
                "El módulo de alineación solo contiene correspondencias heredadas de la base, "
                "decididas por una persona; la equivalencia con el concepto de la base es "
                "automática y queda anotada (regla y concepto de origen).",
                "Las propuestas automáticas van en un módulo aparte y no están validadas.",
                "'Sin correspondencia' no genera tripletas.",
            ],
        }
        (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    except Exception as exc:
        ws.mark_step("write", ERROR, str(exc)[:300])
        raise
    s = manifest["summary"]
    ws.mark_step("write", DONE, f"{s['validated_mappings']} validadas y "
                                f"{s['proposed_mappings']} propuestas",
                 validated_mappings=s["validated_mappings"],
                 proposed_mappings=s["proposed_mappings"])
    return manifest


def load_manifest(ws) -> dict | None:
    path = ws.output_dir / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- package

def package_name(ws, fmt: str | OutputFormat) -> str:
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    return f"{stems(ws)['ontology']}-alineada-{spec.key}.zip"


def _readme_for(manifest: dict):
    def readme(package: Package) -> str:
        s = manifest["summary"]
        main = package.main_file or ""
        jsonld = package.format.key == "jsonld"
        names = {f.rsplit(".", 1)[0]: f for f in package.included}
        text = [
            f"Ontología alineada con el DPV: {manifest['uploaded_file']}",
            f"Formato: {package.format.label}",
            "",
            (f"Archivo principal: {main}." if jsonld
             else f"Abre {main} en Protégé desde esta carpeta (File > Open)."),
        ]
        if jsonld:
            text += ["Nota: " + PROTEGE_JSONLD_NOTE]
        text += [
            f"El archivo {CATALOG_FILE} resuelve las importaciones (IRI -> archivo).",
            "",
            "Qué contiene:",
            f"  - {main}: {s['validated_mappings']} correspondencias VALIDADAS, heredadas de la "
            "base del sistema (decididas por una persona). Importa tu ontología.",
        ]
        prop = next((f for stem, f in names.items() if stem.endswith("-dpv-proposals")), None)
        if prop:
            text.append(f"  - {prop}: {s['proposed_mappings']} propuestas AUTOMÁTICAS, NO "
                        "validadas (módulo aparte; ábrelo solo si quieres verlas).")
        else:
            text.append("  - No hubo propuestas automáticas sobre el umbral.")
        text += [
            "  - Tu ontología, sin cambios"
            + (" (se le agregó un IRI de ontología porque no tenía)."
               if manifest.get("ontology_iri_generated") else "."),
            f"  - {MANIFEST_FILE}: cifras, ajustes y origen de cada decisión.",
            f"  - {CSV_FILE}: una fila por concepto con lo que pasó (ábrelo en Excel).",
            "",
            "El DPV no se importa: sus términos se referencian por IRI (https://w3id.org/dpv#).",
        ]
        return "\n".join(text) + "\n"
    return readme


def build_user_package(ws, fmt: str | OutputFormat = "turtle") -> Package:
    """The zip in ``fmt``: validated module (opened first), proposals, the user's ontology."""
    manifest = load_manifest(ws)
    if manifest is None:
        raise ValueError("Primero hay que escribir la ontología alineada (paso S6-T07).")
    names = stems(ws)
    onto_iri = manifest["ontology_iri"]
    modules = [Module(manifest["modules"]["alignment"], ws.path / manifest["files"]["alignment"],
                      names["alignment"])]
    if manifest["modules"].get("proposals"):
        modules.append(Module(manifest["modules"]["proposals"],
                              ws.path / manifest["files"]["proposals"], names["proposals"]))
    modules.append(Module(onto_iri, ws.path / manifest["files"]["ontology"], names["ontology"]))
    extras = {MANIFEST_FILE: ws.output_dir / MANIFEST_FILE, CSV_FILE: ws.output_dir / CSV_FILE}
    return package_modules(modules, fmt, extras=extras, readme=_readme_for(manifest))


def write_package(ws, fmt: str | OutputFormat = "turtle") -> Path:
    from src.workspace.store import DONE
    package = build_user_package(ws, fmt)
    path = ws.output_dir / package_name(ws, package.format)
    path.write_bytes(package.data)
    ws.mark_step("package", DONE, f"{path.name} ({package.format.label})",
                 file=path.name, format=package.format.key)
    return path


# --------------------------------------------------------------------------- whole back-end

def align_upload(name: str, data: bytes, *, fmt: str = "turtle", use_llm: bool | None = None,
                 embed_fn=None, llm=None, cfg: dict | None = None, base=None, targets=None,
                 progress=None):
    """Upload -> inherit -> propose -> write -> package. Steps already done are reused.

    ``progress(step_label)`` is called before each step (for the web). Returns the workspace,
    the manifest and the zip path."""
    from src.alignment.inherit import inherit_for_workspace
    from src.alignment.propose import propose_for_workspace
    from src.workspace.store import STEP_LABELS, open_ontology
    ws = open_ontology(name, data, cfg=cfg)
    say = progress or (lambda label: None)
    if inherited_result(ws) is None:
        say(STEP_LABELS["inherit"])
        inherit_for_workspace(ws, base=base, embed_fn=embed_fn, cfg=cfg)
    if proposals_result(ws) is None:
        say(STEP_LABELS["candidates"])
        propose_for_workspace(ws, embed_fn=embed_fn, llm=llm, use_llm=use_llm,
                              targets=targets, cfg=cfg)
    say(STEP_LABELS["write"])
    manifest = write_user_alignment(ws, cfg=cfg)
    say(STEP_LABELS["package"])
    path = write_package(ws, fmt)
    return ws, manifest, path


# --------------------------------------------------------------------------- console

def render_console(ws, manifest: dict, zip_path: Path | None = None) -> str:
    s = manifest["summary"]
    c = s["by_status"]
    lines = ["Ontología alineada (S6-T07)",
             f"  {manifest['uploaded_file']}: {s['concepts']} conceptos",
             f"    validadas (heredadas): {c[INHERITED_ALIGNED]} conceptos, "
             f"{s['validated_mappings']} correspondencias",
             f"    sin correspondencia (heredado): {c[INHERITED_NO_MATCH]}",
             f"    propuestas automáticas: {c[PROPOSED]} ({s['proposed_untyped']} sin tipo)",
             f"    sin propuesta: {c[NO_PROPOSAL] + c[AI_REJECTED]}"
             + (f" · individuos (no se alinean): {c[NOT_ALIGNABLE]}" if c[NOT_ALIGNABLE] else ""),
             f"  Módulo validado: {manifest['files']['alignment']} "
             f"({manifest['triples']['alignment']} tripletas)"]
    if manifest["files"].get("proposals"):
        lines.append(f"  Módulo de propuestas: {manifest['files']['proposals']} "
                     f"({manifest['triples']['proposals']} tripletas)")
    if manifest.get("ontology_iri_generated"):
        lines.append(f"  La ontología no tenía IRI: se usó {manifest['ontology_iri']}")
    if zip_path is not None:
        lines.append(f"  Paquete: {zip_path}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.transform.convert import ConversionError
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.user_alignment",
        description="Alinea una ontología subida con el DPV y la empaqueta (S6-T07).")
    parser.add_argument("file", help="ontología .rdf, .owl, .ttl o .jsonld")
    parser.add_argument("--formato", default="turtle", help="rdfxml, owl, turtle o jsonld")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        output_format(args.formato)
        ws, manifest, zip_path = align_upload(path.name, path.read_bytes(), fmt=args.formato,
                                              use_llm=False if args.sin_ia else None)
    except (WorkspaceError, ConversionError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(ws, manifest, zip_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())