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

from src.alignment import propose as dec
from src.alignment.bundle import PROTEGE_JSONLD_NOTE
from src.alignment.inherit import METHOD_LABELS, load_result as inherited_result
from src.alignment import law_definitions as ld
from src.alignment import law_evidence as ev
from src.alignment.propose import load_result as decisions_result
from src.core.emit import CORE_FILE, CORE_IRI
from src.transform import labels as lab
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

# Status of every concept of the user's ontology (CSV, manifest, web), in display order.
INHERITED_ALIGNED = "inherited_aligned"
ACCEPTED = "accepted"
INHERITED_NO_MATCH = "inherited_no_match"
RULE_NO_MATCH = "rule_no_match"
SUGGESTED = "suggested"
NO_PROPOSAL = "no_proposal"
UNDECIDED = "undecided"
NOT_ALIGNABLE = "not_alignable"
STATUS_LABELS = {INHERITED_ALIGNED: "validada (heredada de la base)",
                 ACCEPTED: "aceptada automáticamente",
                 INHERITED_NO_MATCH: "sin correspondencia (heredada de la base)",
                 RULE_NO_MATCH: "sin correspondencia (decidido por la regla)",
                 SUGGESTED: "sugerida (no entra a la ontología)",
                 NO_PROPOSAL: "sin propuesta",
                 UNDECIDED: "sin decidir (la IA no estuvo disponible)",
                 NOT_ALIGNABLE: "no se alinea (individuo)"}
WRITTEN = (INHERITED_ALIGNED, ACCEPTED)          # the statuses that put triples in the module
_FROM_DECIDER = {dec.ACCEPTED: ACCEPTED, dec.SUGGESTED: SUGGESTED, dec.NO_MATCH: RULE_NO_MATCH,
                 dec.NO_PROPOSAL: NO_PROPOSAL, dec.UNDECIDED: UNDECIDED}
CSV_COLUMNS = ["concept_iri", "concept_name", "label_es", "label_en", "labels_added",
               "law_evidence", "law_definition", "law_definition_source",
               "status", "estado", "decided_by", "validated",
               "relation", "dpv_iri", "dpv_name", "score", "confidence", "method",
               "base_concept", "reviewer", "justification", "version"]


# --------------------------------------------------------------------------- names and IRIs

def stems(ws) -> dict[str, str]:
    """File stems inside the package: the upload's name plus the module suffix (``proposals``:
    the module of v0.6.0, only to remove it from an old workspace)."""
    base = Path(ws.manifest["inputs"]["ontology"]["stored_as"]).stem
    return {"ontology": base, "alignment": f"{base}-dpv-alignment",
            "labels": f"{base}{lab.MODULE_SUFFIX}",
            "definitions": f"{base}{ld.MODULE_SUFFIX}", "proposals": f"{base}-dpv-proposals",
            "core": Path(CORE_FILE).stem}


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


def _header(g: Graph, iri: str, imports: str, label: str, comment: str, created: str,
            version: int) -> None:
    node = URIRef(iri)
    g.add((node, RDF.type, OWL.Ontology))
    g.add((node, OWL.imports, URIRef(imports)))
    g.add((node, RDFS.label, Literal(label, lang="es")))
    g.add((node, RDFS.comment, Literal(comment, lang="es")))
    g.add((node, DCTERMS.created, Literal(created, datatype=XSD.dateTime)))
    g.add((node, OWL.versionInfo, Literal(str(version))))


def _annotation_property(g: Graph, iri: URIRef, label: str, comment: str) -> URIRef:
    g.add((iri, RDF.type, OWL.AnnotationProperty))
    g.add((iri, RDFS.label, Literal(label, lang="es")))
    g.add((iri, RDFS.comment, Literal(comment, lang="es")))
    return iri


def _score(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


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

def alignment_graph(inherited: dict, decisions: dict | None, *, iri: str, onto_iri: str,
                    name: str, created: str, source: Graph | None = None,
                    labels_iri: str | None = None, definitions_iri: str | None = None) -> Graph:
    """The module: inherited correspondences + the ones the rule accepted (never suggestions);
    it imports the user's ontology and, when there are, the labels and definitions modules."""
    decisions = decisions or {}
    version = decisions.get("alignment_version", dec.ALIGNMENT_VERSION)
    g = Graph()
    if source is not None:
        _bind(g, source)
    _header(g, iri, onto_iri, f"Alineación de {name} con el DPV",
            "Correspondencias heredadas de la base del sistema (OntoPriv + LOPDP, decididas "
            "por una persona en el Sprint 5) y aceptadas automáticamente por la regla del "
            "decisor (S7-T05). Las sugeridas no se escriben. El DPV se referencia por IRI, no "
            "se importa.", created, version)
    for extra in (labels_iri, definitions_iri):
        if extra:
            g.add((URIRef(iri), OWL.imports, URIRef(extra)))
    ns = iri.rstrip("#/")
    g.bind("alignment", Namespace(ns + "#"), override=True)
    by = _annotation_property(g, URIRef(ns + "#decidedBy"), "decidido por",
                              "Quién decidió la correspondencia: base (una persona, Sprint 5), "
                              "regla (el decisor con IA y la regla calibrada) o regla sin IA.")
    conf = _annotation_property(g, URIRef(ns + "#confidence"), "puntaje del candidato",
                                "Puntaje de 0 a 1 del término del DPV (0.4 léxico + 0.6 "
                                "embeddings) cuando lo aceptó la regla.")
    stamp = Literal(str(version))
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
                      [(DCTERMS.source, URIRef(item["base_iri"])),
                       (by, Literal("base", lang="es")), (OWL.versionInfo, stamp)])
    rule = (decisions.get("summary") or {}).get("rule", "")
    threshold = decisions.get("threshold_without_ai")
    for e in decisions.get("entries", []):
        if e["status"] != dec.ACCEPTED:
            continue
        subject, target = URIRef(e["key"]), URIRef(e["dpv_iri"])
        prop = _SKOS.get(e.get("relation") or "", RDFS.seeAlso)
        g.add((subject, prop, target))
        if e["decided_by"] == dec.BY_RULE_WITHOUT_AI:
            text = ("Aceptada automáticamente sin IA: mejor candidato con puntaje "
                    f"{_score(e['score'])} (mínimo {_score(threshold or 0)}); sin tipo SKOS.")
            who = "regla sin IA"
        else:
            text = f"Aceptada automáticamente por la regla ({rule})."
            if e.get("justification"):
                text += f" Justificación de la IA: {e['justification']}"
            who = "regla"
        extra = [(by, Literal(who, lang="es")), (OWL.versionInfo, Literal(str(e["version"])))]
        if e.get("score") is not None:
            extra.append((conf, Literal(f"{e['score']:.4f}", datatype=XSD.decimal)))
        _annotate(g, subject, prop, target, text, extra)
    return g


# --------------------------------------------------------------------------- rows

def concept_rows(inherited: dict, decisions: dict | None, labels: dict | None = None,
                 evidence: dict | None = None, definitions: dict | None = None) -> list[dict]:
    """One row per concept of the user's ontology (aligned ones: one row per mapping), with its
    labels in Spanish and English, the articles of the law and the definition it gave."""
    version = (decisions or {}).get("alignment_version", dec.ALIGNMENT_VERSION)
    rows = []
    for item in inherited.get("inherited", []):
        common = {"concept_iri": item["key"], "concept_name": item["name"],
                  "method": METHOD_LABELS[item["method"]], "base_concept": item["base_iri"],
                  "decided_by": "base", "validated": True, "version": version}
        if item["status"] == "aligned" and item.get("mappings"):
            for m in item["mappings"]:
                rows.append({**common, "status": INHERITED_ALIGNED, "relation": m["relation"],
                             "dpv_iri": m["dpv_iri"], "dpv_name": m.get("dpv_name"),
                             "reviewer": m.get("reviewer")})
        else:
            rows.append({**common, "status": INHERITED_NO_MATCH})
    decided = {e["key"] for e in (decisions or {}).get("entries", [])}
    for e in (decisions or {}).get("entries", []):
        status = _FROM_DECIDER[e["status"]]
        row = {"concept_iri": e["key"], "concept_name": e["name"], "status": status,
               "decided_by": e["decided_by"], "validated": False, "score": e.get("score"),
               "confidence": e.get("confidence"), "justification": e.get("justification"),
               "version": e.get("version", version)}
        if status in (ACCEPTED, SUGGESTED):
            row.update(relation=e.get("relation") or "rdfs:seeAlso", dpv_iri=e["dpv_iri"],
                       dpv_name=e.get("dpv_name"))
        rows.append(row)
    for n in inherited.get("not_inherited", []):
        if n.get("reason") == "kind" and n["key"] not in decided:
            rows.append({"concept_iri": n["key"], "concept_name": n["name"],
                         "status": NOT_ALIGNABLE, "validated": False})
    named, articles = lab.by_key(labels), ev.by_key(evidence)
    defined, law = ld.by_key(definitions), ((definitions or {}).get("law") or {}).get("file")
    for r in rows:
        r["estado"] = STATUS_LABELS[r["status"]]
        if r["concept_iri"] in articles:
            r["law_evidence"] = ev.evidence_text(articles[r["concept_iri"]])
        given = defined.get(r["concept_iri"])
        if given is not None:
            r.update(law_definition=given["definition"],
                     law_definition_source=ld.source_text(given, law))
        entry = named.get(r["concept_iri"])
        if entry is not None:
            final = lab.final_labels(entry)
            r.update(label_es=final["es"], label_en=final["en"],
                     labels_added=lab.added_text(entry))
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
    accepted = [r for r in rows if r["status"] == ACCEPTED]
    return {"concepts": len(concepts), "by_status": counts,
            "validated_mappings": sum(1 for r in rows if r["status"] == INHERITED_ALIGNED),
            "accepted_mappings": len(accepted),
            "accepted_without_ai": sum(r["decided_by"] == dec.BY_RULE_WITHOUT_AI
                                       for r in accepted),
            "accepted_untyped": sum(r.get("relation") == "rdfs:seeAlso" for r in accepted),
            "suggested": sum(1 for r in rows if r["status"] == SUGGESTED),
            "mappings": sum(1 for r in rows if r["status"] in WRITTEN)}


# --------------------------------------------------------------------------- writing

def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_user_alignment(ws, cfg: dict | None = None, core: Graph | None = None,
                         recomputed: dict | None = None) -> dict:
    """Write the modules (Turtle), the manifest and the CSV into ``output/``; mark the step.

    The labels of S7-T08 are optional here (``align_upload`` always computes them first): with
    no result, no labels module is written and the CSV has no labels. ``core``: OntoPriv-Core
    already loaded (else it is read from the base when the ontology imports it).
    ``recomputed``: which steps this run did again (S7-T13: the console says when the AI was
    not called and the figures are those of an earlier run); None when it is not known."""
    from src.alignment.inherit import base_paths
    from src.workspace.store import DONE, ERROR, RUNNING
    from src.alignment.decider import PROMPT as DECIDE_PROMPT
    inherited = inherited_result(ws)
    if inherited is None:
        raise ValueError("Primero hay que heredar lo validado (paso S6-T05).")
    decisions = decisions_result(ws)
    if decisions is None:
        raise ValueError("Primero hay que decidir lo que no se heredó (paso S7-T06).")
    labels = lab.load_result(ws)
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
        iri = module_iri(onto_iri, "-dpv-alignment")
        files = {"ontology": ontology_file.relative_to(ws.path).as_posix()}
        modules, triples = {"alignment": iri}, {}

        labels_iri = module_iri(onto_iri, lab.MODULE_SUFFIX)
        label_graph = (lab.labels_graph(labels, iri=labels_iri, onto_iri=onto_iri, name=display,
                                        created=created) if labels else None)
        label_file = out / f"{names['labels']}.ttl"
        if label_graph is not None:
            label_file.write_bytes(serialize(label_graph, MASTER_FORMAT))
            files["labels"] = f"output/{names['labels']}.ttl"
            modules["labels"] = labels_iri
            triples["labels"] = len(label_graph)
        elif label_file.exists():
            label_file.unlink()                        # nothing added any more

        definitions = ld.load_result(ws)
        definitions_iri = module_iri(onto_iri, ld.MODULE_SUFFIX)
        definition_graph = (ld.definitions_graph(definitions, iri=definitions_iri,
                                                 onto_iri=onto_iri, name=display, created=created)
                            if definitions else None)
        definition_file = out / f"{names['definitions']}.ttl"
        if definition_graph is not None:
            definition_file.write_bytes(serialize(definition_graph, MASTER_FORMAT))
            files["definitions"] = f"output/{names['definitions']}.ttl"
            modules["definitions"] = definitions_iri
            triples["definitions"] = len(definition_graph)
        elif definition_file.exists():
            definition_file.unlink()

        graph = alignment_graph(inherited, decisions, iri=iri, onto_iri=onto_iri, name=display,
                                created=created, source=user_graph,
                                labels_iri=modules.get("labels"),
                                definitions_iri=modules.get("definitions"))
        (out / f"{names['alignment']}.ttl").write_bytes(serialize(graph, MASTER_FORMAT))
        files["alignment"] = f"output/{names['alignment']}.ttl"
        triples = {"alignment": len(graph), **triples}
        if (URIRef(onto_iri), OWL.imports, URIRef(CORE_IRI)) in user_graph:
            from src.core.law_profile import load_core
            core = core if core is not None else load_core(cfg)
            (out / f"{names['core']}.ttl").write_bytes(serialize(core, MASTER_FORMAT))
            files["core"] = f"output/{names['core']}.ttl"
            modules["core"], triples["core"] = CORE_IRI, len(core)
        stale = out / f"{names['proposals']}.ttl"
        if stale.exists():
            stale.unlink()                             # module of proposals of v0.6.0

        evidence = ev.load_result(ws)                  # "ontología + ley": only the CSV
        rows = concept_rows(inherited, decisions, labels, evidence, definitions)
        write_csv(rows, out / CSV_FILE)
        paths = base_paths(cfg)
        prompts = {"decide": DECIDE_PROMPT.manifest()}
        if labels:
            prompts["labels"] = labels["prompt"]
        entry = ws.manifest["inputs"]["ontology"]
        profile = None
        if entry.get("generated_from") == "law":
            from src.core.law_concepts import load_result as law_concepts
            from src.core.law_profile import load_result as law_profile
            built, concepts = law_profile(ws) or {}, law_concepts(ws) or {}
            profile = {**(built.get("summary") or {}), "law": built.get("law"),
                       "iri": built.get("profile_iri"), "language": built.get("language"),
                       "law_concepts": concepts.get("summary"),
                       "extraction": concepts.get("extraction")}
            prompts["extract"] = concepts.get("prompt")
        defined = None
        if definitions:
            defined = {**definitions["summary"], "law": definitions["law"]["file"],
                       "language": definitions["law"]["language"],
                       "skipped": definitions["skipped"], "ai": definitions["ai"],
                       "law_concepts": ({k: definitions["law_concepts"][k]
                                         for k in ("summary", "extraction", "ai")}
                                        if definitions["law_concepts"] else None)}
            if not definitions["skipped"]:
                prompts["define"] = definitions["prompt"]
            if definitions["law_concepts"]:
                prompts["extract"] = definitions["law_concepts"]["prompt"]
        manifest = {
            "workspace": ws.id, "source_id": ws.source_id, "created": created,
            "uploaded_file": display, "ontology_iri": onto_iri,
            "ontology_iri_generated": generated,
            "generated_from": entry.get("generated_from"), "profile": profile,
            "alignment_version": decisions.get("alignment_version", dec.ALIGNMENT_VERSION),
            "modules": modules, "files": files, "triples": triples,
            "summary": summarize(rows),
            "inheritance": inherited.get("summary"),
            "decider": decisions.get("summary"),
            "labels": ({**labels["summary"], "glossary": labels["glossary"],
                        "ai": labels["ai"]} if labels else None),
            "law_definitions": defined,
            "law_evidence": ({**evidence["summary"], "law": evidence["law"],
                              "settings": evidence["settings"]} if evidence else None),
            "recomputed": recomputed,
            "settings": {"inheritance": inherited.get("settings"),
                         "rule": decisions.get("rule"),
                         "threshold_without_ai": decisions.get("threshold_without_ai"),
                         "embedding_model": os.environ.get("EMBEDDING_MODEL"),
                         "llm_model": os.environ.get("LLM_MODEL"),
                         "prompts": prompts},
            "base": {"source_id": (inherited.get("settings") or {}).get("base_source_id"),
                     "decisions_log_sha256": _sha256(paths["log"])},
            "notes": [
                "El módulo de alineación contiene las correspondencias heredadas de la base "
                "(decididas por una persona) y las aceptadas automáticamente por la regla; "
                "cada una lleva anotado quién la decidió y su versión.",
                "Las sugeridas (la IA las dio con confianza alta pero la regla no las acepta) "
                "no entran a la ontología: están en alignment.csv para comentarlas.",
                "'Sin correspondencia' no genera tripletas.",
            ] + (["El módulo de etiquetas agrega, con su idioma, las etiquetas en español e "
                  "inglés que le faltaban a la ontología (las escribió la IA); ninguna etiqueta "
                  "original cambia."] if "labels" in modules else [])
            + ([ld.module_note(defined)] if "definitions" in modules else [])
            + ([f"La columna law_evidence de alignment.csv trae los artículos de "
                f"{evidence['law']} más cercanos a cada concepto (búsqueda semántica), para "
                "revisarlos; no entran a la ontología: en GDPRtEXT ninguna regla llegó al 75 % "
                "de precisión (S7-T11)."
                + (" Para los conceptos que tomaron la definición de un concepto de la ley son "
                   "los artículos que citó la IA (citado)."
                   if evidence["summary"].get("cited_by_ai") else "")] if evidence else []),
        }
        (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    except Exception as exc:
        ws.mark_step("write", ERROR, str(exc)[:300])
        raise
    s = manifest["summary"]
    ws.mark_step("write", DONE, f"{s['validated_mappings']} de la base y "
                                f"{s['accepted_mappings']} aceptadas por la regla",
                 validated_mappings=s["validated_mappings"],
                 accepted_mappings=s["accepted_mappings"], suggested=s["suggested"])
    return manifest


def load_manifest(ws) -> dict | None:
    path = ws.output_dir / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- package

def package_name(ws, fmt: str | OutputFormat) -> str:
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    return f"{stems(ws)['ontology']}-alineada-{spec.key}.zip"


def _included(package: Package, path: str) -> str | None:
    """The file of the package that holds the module stored at ``path`` (any format)."""
    return next((f for f in package.included if Path(f).stem == Path(path).stem), None)


def _readme_for(manifest: dict):
    def readme(package: Package) -> str:
        s = manifest["summary"]
        main = package.main_file or ""
        jsonld = package.format.key == "jsonld"
        text = [
            f"Ontología alineada con el DPV: {manifest['uploaded_file']}",
            f"Formato: {package.format.label}",
            "",
            (f"Archivo principal: {main}." if jsonld
             else f"Abre {main} en Protégé desde esta carpeta (File > Open)."),
        ]
        if jsonld:
            text += ["Nota: " + PROTEGE_JSONLD_NOTE]
        rule = ((manifest.get("decider") or {}).get("rule")) or ""
        text += [
            f"El archivo {CATALOG_FILE} resuelve las importaciones (IRI -> archivo).",
            "",
            "Qué contiene:",
            f"  - {main}: {s['mappings']} correspondencias con el DPV. Importa tu ontología.",
            f"      {s['validated_mappings']} heredadas de la base del sistema (decididas por "
            "una persona).",
            f"      {s['accepted_mappings']} aceptadas automáticamente por la regla"
            + (f" ({rule})" if rule else "") + ".",
        ]
        if s.get("accepted_without_ai"):
            text.append(f"      De ellas, {s['accepted_without_ai']} sin IA (sin tipo SKOS: "
                        "rdfs:seeAlso).")
        text.append("      Cada correspondencia lleva anotado quién la decidió y su versión.")
        labels = manifest.get("labels") or {}
        label_file = (_included(package, manifest["files"]["labels"])
                      if manifest["files"].get("labels") else None)
        if label_file:
            added = labels.get("added") or {}
            text.append(f"  - {label_file}: las etiquetas que le faltaban a tu ontología "
                        f"({added.get('es', 0)} en español y {added.get('en', 0)} en inglés), "
                        "escritas por la IA. Tus etiquetas no cambian.")
        if labels.get("pending"):
            text.append(f"  - {labels['pending']} conceptos quedaron sin sus etiquetas en español "
                        "o inglés (la IA no las completó): vuelve a subir el archivo para "
                        "completarlas.")
        defined = manifest.get("law_definitions") or {}
        if manifest["files"].get("definitions"):
            text.append(f"  - {_included(package, manifest['files']['definitions'])}: "
                        f"{ld.added_text(defined, 'tu ontología')}; cada una cita su artículo. "
                        "Tus definiciones no cambian.")
        if defined.get(ld.PENDING):
            text.append(f"  - {defined[ld.PENDING]} conceptos quedaron sin su definición de la "
                        "ley (la IA no la completó): vuelve a subir los archivos para "
                        "completarlas.")
        if s.get("suggested"):
            text.append(f"  - {s['suggested']} sugeridas: la IA las dio con confianza alta "
                        f"pero la regla no las acepta; NO están en la ontología, solo en "
                        f"{CSV_FILE}.")
        if manifest.get("generated_from") == "law":
            p = manifest.get("profile") or {}
            text.append(f"  - {_included(package, manifest['files']['ontology'])}: el perfil "
                        f"de tu ley ({p.get('concepts', 0)} conceptos que la IA extrajo y la "
                        f"regla aceptó: {p.get('reused', 0)} reutilizan OntoPriv-Core, "
                        f"{p.get('linked', 0)} se enlazan con él y {p.get('new', 0)} son "
                        "nuevos). Importa OntoPriv-Core.")
        else:
            text.append("  - Tu ontología, sin cambios"
                        + (" (se le agregó un IRI de ontología porque no tenía)."
                           if manifest.get("ontology_iri_generated") else "."))
        if manifest["files"].get("core"):
            text.append(f"  - {_included(package, manifest['files']['core'])}: el núcleo "
                        "OntoPriv-Core de la base del sistema, sin cambios.")
        text += [
            f"  - {MANIFEST_FILE}: cifras, ajustes y origen de cada decisión.",
            f"  - {CSV_FILE}: una fila por concepto con lo que pasó (ábrelo en Excel).",
        ]
        evidence = manifest.get("law_evidence")
        if evidence:
            text.append(f"    Su columna law_evidence trae, para {evidence['with_evidence']} de "
                        f"tus conceptos, los artículos de {evidence['law']} más cercanos "
                        f"({evidence['links']} en total, con su distancia), encontrados por "
                        "búsqueda semántica: son para que los revises; no entran a la "
                        "ontología.")
            if evidence.get("cited_by_ai"):
                text.append(f"    En {evidence['cited_by_ai']} de ellos el artículo es el que "
                            "citó la IA al sacar su definición de la ley (citado).")
        text += [
            "",
            "El DPV no se importa: sus términos se referencian por IRI (https://w3id.org/dpv#).",
        ]
        return "\n".join(text) + "\n"
    return readme


def build_user_package(ws, fmt: str | OutputFormat = "turtle") -> Package:
    """The zip in ``fmt``: the alignment module (opened first) and the user's ontology."""
    manifest = load_manifest(ws)
    if manifest is None:
        raise ValueError("Primero hay que escribir la ontología alineada (paso S6-T07).")
    names = stems(ws)
    onto_iri = manifest["ontology_iri"]
    modules = [Module(manifest["modules"]["alignment"], ws.path / manifest["files"]["alignment"],
                      names["alignment"])]
    for extra in ("labels", "definitions"):
        if manifest["modules"].get(extra):
            modules.append(Module(manifest["modules"][extra], ws.path / manifest["files"][extra],
                                  names[extra]))
    modules.append(Module(onto_iri, ws.path / manifest["files"]["ontology"], names["ontology"]))
    if manifest["modules"].get("core"):
        modules.append(Module(manifest["modules"]["core"], ws.path / manifest["files"]["core"],
                              names["core"]))
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
                 progress=None, chroma_path=None, sleep_fn=None):
    """Upload -> inherit -> decide -> labels -> write -> package. Steps already done are reused.

    The decision is redone when it is not current (another method, as in v0.6.0, another
    rule, or concepts the AI could not decide while it is available), and so are the labels
    (S7-T08: another prompt or glossary, or concepts left pending while the AI is available);
    the caches of the workspace make that cheap. ``progress(step_label)`` is called before each
    step (for the web). Returns the workspace, the manifest and the zip path."""
    from src.workspace.store import open_ontology
    ws = open_ontology(name, data, cfg=cfg)
    return align_workspace(ws, fmt=fmt, use_llm=use_llm, embed_fn=embed_fn, llm=llm, cfg=cfg,
                           base=base, targets=targets, progress=progress,
                           chroma_path=chroma_path, sleep_fn=sleep_fn)


def align_workspace(ws, *, fmt: str = "turtle", use_llm: bool | None = None, embed_fn=None,
                    llm=None, cfg: dict | None = None, base=None, targets=None, progress=None,
                    chroma_path=None, sleep_fn=None, core: Graph | None = None):
    """The steps of ``align_upload`` on a workspace that already holds its ontology (an
    upload, or the profile of a law in the "solo ley" mode, S7-T10)."""
    from src.alignment.inherit import inherit_for_workspace
    from src.alignment.propose import is_current, propose_for_workspace
    from src.workspace.store import STEP_LABELS
    say = progress or (lambda label: None)
    with_law = ws.manifest.get("mode") == "ontology+law"
    ran = {"inherit": False, "definitions": False, "decider": False, "labels": False,
           "evidence": False}
    if inherited_result(ws) is None:
        say(STEP_LABELS["inherit"])
        inherit_for_workspace(ws, base=base, embed_fn=embed_fn, cfg=cfg)
        ran["inherit"] = True
    if with_law and not ld.is_current(ld.load_result(ws), ws, cfg, use_llm):
        ld.definitions_for_workspace(ws, embed_fn=embed_fn, llm=llm, use_llm=use_llm, cfg=cfg,
                                     sleep_fn=sleep_fn, progress=progress)
        ran["definitions"] = True
    definitions = ld.fingerprint(ld.load_result(ws))
    if not is_current(decisions_result(ws), cfg, use_llm, definitions):
        say(STEP_LABELS["candidates"])
        propose_for_workspace(ws, embed_fn=embed_fn, llm=llm, use_llm=use_llm,
                              targets=targets, cfg=cfg, base=base, chroma_path=chroma_path,
                              sleep_fn=sleep_fn)
        ran["decider"] = True
    if not lab.is_current(lab.load_result(ws), cfg, use_llm):
        say(STEP_LABELS["labels"])
        lab.labels_for_workspace(ws, llm=llm, use_llm=use_llm, cfg=cfg, sleep_fn=sleep_fn)
        ran["labels"] = True
    if with_law and not ev.is_current(ev.load_result(ws), cfg, definitions):
        say(STEP_LABELS["evidence"])
        ev.evidence_for_workspace(ws, embed_fn=embed_fn, cfg=cfg)
        ran["evidence"] = True
    say(STEP_LABELS["write"])
    manifest = write_user_alignment(ws, cfg=cfg, core=core, recomputed=ran)
    say(STEP_LABELS["package"])
    path = write_package(ws, fmt)
    return ws, manifest, path


# --------------------------------------------------------------------------- console

REUSED = "no se volvió a llamar; son las cifras de la corrida anterior"


def decider_ai_text(ai: dict) -> str:
    """The AI of the decider in one line, with why it stopped and its first error."""
    text = f"{ai.get('calls', 0)} llamadas, {ai.get('cached', 0)} respuestas desde la caché"
    if ai.get("quota_exhausted"):
        text += " (cuota agotada: vuelve a correr mañana)"
    elif ai.get("call_limit_reached"):
        text += " (límite de llamadas: vuelve a correr para seguir)"
    if ai.get("errors"):
        text += f"; error: {ai['errors'][0]}"
    return text


def render_console(ws, manifest: dict, zip_path: Path | None = None) -> str:
    again = manifest.get("recomputed") or {}

    def reused(step: str) -> bool:                      # unknown (None): say nothing
        return bool(again) and not again.get(step)

    s = manifest["summary"]
    c = s["by_status"]
    lines = ["Ontología alineada (S7-T08)",
             f"  {manifest['uploaded_file']}: {s['concepts']} conceptos",
             f"    heredadas de la base: {c[INHERITED_ALIGNED]} conceptos, "
             f"{s['validated_mappings']} correspondencias",
             f"    aceptadas automáticamente: {c[ACCEPTED]} ({s['accepted_untyped']} sin tipo)",
             f"    sugeridas (no entran a la ontología): {c[SUGGESTED]}",
             f"    sin correspondencia: {c[INHERITED_NO_MATCH] + c[RULE_NO_MATCH]}",
             f"    sin propuesta: {c[NO_PROPOSAL]}"
             + (f" · sin decidir: {c[UNDECIDED]}" if c[UNDECIDED] else "")
             + (f" · individuos (no se alinean): {c[NOT_ALIGNABLE]}" if c[NOT_ALIGNABLE] else ""),
             f"  Módulo de alineación: {manifest['files']['alignment']} "
             f"({manifest['triples']['alignment']} tripletas, versión "
             f"{manifest['alignment_version']})"]
    decider = manifest.get("decider") or {}
    ai = decider.get("ai") or {}
    if decider.get("rule"):
        lines.append(f"  Regla: {decider['rule']}")
    if ai.get("available"):
        text = decider_ai_text(ai)
        lines.append("  IA del decisor: " + (f"{REUSED} ({text})" if reused("decider") else text))
    elif decider.get("concepts"):
        lines.append(f"  IA del decisor: no usada ({ai.get('error')})")
    labels = manifest.get("labels")
    if labels:
        added = labels["added"]
        lines.append(f"  Etiquetas es/en: {labels['complete']} conceptos ya las tenían; la IA "
                     f"agregó {added['es']} en español y {added['en']} en inglés"
                     + (f" · pendientes: {labels['pending']}" if labels["pending"] else "")
                     + (f" (módulo {manifest['files']['labels']}, "
                        f"{manifest['triples']['labels']} tripletas)"
                        if manifest["files"].get("labels") else ""))
        text = lab.ai_line(labels)
        available = (labels.get("ai") or {}).get("available")
        lines.append("  IA de las etiquetas: "
                     + (f"{REUSED} ({text})" if available and reused("labels") else text))
    defined = manifest.get("law_definitions")
    if defined:
        lines.append(f"  {ld.summary_line(defined)}"
                     + (f" (módulo {manifest['files']['definitions']}, "
                        f"{manifest['triples']['definitions']} tripletas)"
                        if manifest["files"].get("definitions") else "")
                     + (" (de la corrida anterior)" if reused("definitions") else ""))
        if not defined["skipped"]:
            text = ld.ai_line(defined)
            lines.append("  IA de las definiciones: "
                         + (f"{REUSED} ({text})" if reused("definitions") else text))
    evidence = manifest.get("law_evidence")
    if evidence:
        lines.append(f"  Evidencia de {evidence['law']}: {evidence['with_evidence']} de "
                     f"{evidence['concepts']} conceptos con artículos para revisar "
                     f"({evidence['links']} enlaces, {evidence['articles_cited']} artículos "
                     f"distintos) en {CSV_FILE}; no entran a la ontología"
                     + (" (de la corrida anterior)" if reused("evidence") else ""))
    if manifest.get("ontology_iri_generated"):
        lines.append(f"  La ontología no tenía IRI: se usó {manifest['ontology_iri']}")
    if manifest["files"].get("core"):
        lines.append(f"  Núcleo OntoPriv-Core incluido: {manifest['files']['core']} "
                     f"({manifest['triples']['core']} tripletas)")
    if zip_path is not None:
        lines.append(f"  Paquete: {zip_path}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.transform.convert import ConversionError
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.user_alignment",
        description="Alinea una ontología subida con el DPV, completa sus etiquetas es/en y la "
                    "empaqueta (S7-T08).")
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