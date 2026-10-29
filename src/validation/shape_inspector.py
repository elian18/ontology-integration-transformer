from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from src.alignment.sources import is_borrowed
from src.validation import shapes as sh

REPORT_FILE = "shacl-report.json"
LOG_FILE = "shacl-log.txt"
NEW, ONTOLOGY, BASE = "nuevo", "ontología", "base"
RED, AMBER, GREEN = "rojo", "ámbar", "verde"
KIND_BASE, KIND_WORKSPACE = "base", "workspace"
EXAMPLES = 3

MAPPINGS = (SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch,
            SKOS.relatedMatch, RDFS.seeAlso)
PATH = "path"
# The triples behind a finding: (their predicates, whether their object is the finding's value).
# None: every triple of the focus node (an annotation, owl:Axiom).
EVIDENCE = {
    "DpvTermShape": (MAPPINGS, True),
    "MappingConflictShape": (MAPPINGS, True),
    "AlignedConceptShape": (MAPPINGS, False),
    "MappingAnnotationShape": (None, False),
    "LabelShape": ((RDFS.label,), True),
    "DefinitionShape": ((SKOS.definition,), True),
    "LawDefinitionShape": (None, False),
    "LangStringShape": (PATH, True),
    "ImportShape": ((OWL.imports,), True),
}
DECLARED = ((RDF.type,), False)               # completeness: where the concept is declared
CONCEPT_TYPES = (OWL.Class, RDFS.Class, OWL.ObjectProperty, OWL.DatatypeProperty)
COMPLETENESS = {"SpanishLabelShape": "label_es", "EnglishLabelShape": "label_en",
                "DefinitionCompletenessShape": "definition"}


@dataclass
class Part:
    kind: str
    file: str
    origin: str
    graph: Graph
    sha256: str = ""


# --------------------------------------------------------------------------- the parts

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _part(kind: str, path: Path, origin: str, shown: str) -> Part:
    from src.transform.convert import load_graph
    return Part(kind, shown, origin, load_graph(path), _sha256(path))


def workspace_parts(ws) -> list[Part]:
    from src.alignment.user_alignment import load_manifest
    manifest = load_manifest(ws)
    if manifest is None:
        raise ValueError("Primero alinea la ontología: falta output/alignment-manifest.json.")
    own = NEW if ws.manifest["inputs"]["ontology"].get("generated_from") == "law" else ONTOLOGY
    origins = {"ontology": own, "core": BASE}
    return [_part(kind, ws.path / rel, origins.get(kind, NEW), rel)
            for kind, rel in manifest["files"].items()]


def base_parts(cfg: dict | None = None) -> tuple[list[Part], list[Part]]:
    from src.alignment.inherit import base_paths
    from src.alignment.write import ALIGNMENT_FILE
    from src.config import ROOT, load_config
    from src.core.emit import CORE_FILE, PROFILE_FILE
    cfg = cfg if cfg is not None else load_config()
    out = base_paths(cfg)["candidates"].parent
    parts = []
    for kind, name in (("core", CORE_FILE), ("profile", PROFILE_FILE),
                       ("alignment", ALIGNMENT_FILE)):
        path = out / name
        if not path.exists():
            raise FileNotFoundError(f"Falta la base del sistema: {path}")
        parts.append(_part(kind, path, NEW, _shown(path, ROOT)))
    seed = Path(cfg["inputs"]["ontology"])
    seed = seed if seed.is_absolute() else ROOT / seed
    references = [_part("ontopriv", seed, ONTOLOGY, _shown(seed, ROOT))] if seed.exists() else []
    return parts, references


def _shown(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- origins

def _node(text: str):
    return BNode(text[2:]) if text.startswith("_:") else URIRef(text)


def _contains(graph: Graph, triple) -> bool:
    if triple in graph:
        return True
    s, p, o = triple
    return isinstance(s, BNode) and any(isinstance(x, BNode) for x in graph.subjects(p, o))


def evidence(finding: sh.Finding, data: Graph) -> list[tuple]:
    predicates, by_value = EVIDENCE.get(finding.rule, DECLARED)
    focus = _node(finding.focus)
    if predicates == PATH:
        predicates = (URIRef(finding.path),) if finding.path else None
    found = []
    for p in (predicates or [None]):
        for s, pp, o in data.triples((focus, p, None)):
            if not by_value or finding.value is None or sh.node_text(o) == finding.value:
                found.append((s, pp, o))
    return found


def origin(triples: list[tuple], groups: list[Part]) -> str:
    if not triples:
        return NEW
    sets = [{g.origin for g in groups if _contains(g.graph, t)} for t in triples]
    if any(s <= {NEW} for s in sets):                # a triple only the system wrote
        return NEW
    if any(ONTOLOGY in s for s in sets):
        return ONTOLOGY
    return NEW if any(NEW in s for s in sets) else BASE


def _concepts(data: Graph, groups: list[Part]) -> set:
    found = set()
    for t in CONCEPT_TYPES:
        for c in data.subjects(RDF.type, t):
            if isinstance(c, URIRef) and not is_borrowed(str(c)):
                found.add(c)
    return {c for c in found
            if origin(list(data.triples((c, RDF.type, None))), groups) != BASE}


# --------------------------------------------------------------------------- inspecting

def _shapes_info(shapes) -> dict:
    from src.config import ROOT
    return {"file": _shown(sh.SHAPES_FILE, ROOT), "sha256": _sha256(sh.SHAPES_FILE),
            "version": str(shapes.value(URIRef(sh.SHAPES_NS), OWL.versionInfo) or "")}


def status_of(summary: dict) -> str:
    if summary["violations"][NEW]:
        return RED
    if sum(summary["violations"].values()) or sum(summary["warnings"].values()):
        return AMBER
    return GREEN


def inspect(parts: list[Part], references: list[Part] = (), *, dpv: Graph | None = None,
            kind: str = KIND_WORKSPACE, name: str = "") -> dict:
    shapes = sh.load_shapes()
    catalog = sh.rules(shapes)
    data = Graph()
    for part in parts:
        data += part.graph
    triples = len(data)
    if dpv is not None:
        data += dpv
    start = time.perf_counter()
    findings = sh.check(data, shapes)
    seconds = round(time.perf_counter() - start, 2)
    groups = [*parts, *references]
    rows, counts = [], {r.id: {NEW: 0, ONTOLOGY: 0, BASE: 0} for r in catalog}
    for f in findings:
        where = origin(evidence(f, data), groups)
        counts[f.rule][where] += 1
        rows.append({**asdict(f), "origin": where})
    by_severity = {s: {NEW: 0, ONTOLOGY: 0, BASE: 0} for s in (sh.VIOLATION, sh.WARNING)}
    for r in catalog:
        for where, n in counts[r.id].items():
            by_severity[r.severity][where] += n
    concepts = _concepts(data, groups)
    missing = {key: {f.focus for f in findings if f.rule == rule}
               for rule, key in COMPLETENESS.items()}
    summary = {"violations": by_severity[sh.VIOLATION], "warnings": by_severity[sh.WARNING],
               "rules": {r.id: {"name": r.name, "severity": r.severity, **counts[r.id]}
                         for r in catalog}}
    return {
        "kind": kind, "name": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": {"name": "pySHACL", "version": version("pyshacl"), "inference": "none",
                   "advanced": True},
        "shapes": _shapes_info(shapes),
        "parts": [{"kind": p.kind, "file": p.file, "origin": p.origin, "triples": len(p.graph),
                   "sha256": p.sha256} for p in parts],
        "references": [{"kind": p.kind, "file": p.file, "sha256": p.sha256} for p in references],
        "triples": triples, "seconds": seconds, "status": status_of(summary),
        "summary": summary,
        "completeness": {"concepts": len(concepts),
                         **{key: sum(1 for c in concepts if str(c) not in gone)
                            for key, gone in missing.items()}},
        "findings": rows,
    }


def _dpv(cfg: dict | None) -> Graph:
    from src.config import ROOT, load_config
    cfg = cfg if cfg is not None else load_config()
    path = Path(cfg["inputs"]["dpv"])
    return sh.dpv_terms(path if path.is_absolute() else ROOT / path)


def _save(report: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    (folder / LOG_FILE).write_text(render_log(report) + "\n", encoding="utf-8")
    return report


def inspect_workspace(ws, cfg: dict | None = None) -> dict:
    report = inspect(workspace_parts(ws), dpv=_dpv(cfg), kind=KIND_WORKSPACE, name=ws.id)
    return _save(report, ws.work_dir)


def inspect_base(cfg: dict | None = None, out_dir: str | Path | None = None) -> dict:
    from src.alignment.inherit import base_paths
    parts, references = base_parts(cfg)
    report = inspect(parts, references, dpv=_dpv(cfg), kind=KIND_BASE,
                     name="OntoPriv-Core + perfil de Ecuador + alineación con el DPV")
    folder = Path(out_dir) if out_dir is not None else base_paths(cfg)["candidates"].parent
    return _save(report, folder)


def load_result(ws) -> dict | None:
    path = ws.work_dir / REPORT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console / log

def _percent(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}".replace(".", ",") + " %" if total else "—"


def origin_text(where: str, kind: str, n: int = 1) -> str:
    if where == NEW:
        return ("nuevo" if n == 1 else "nuevos") if kind == KIND_WORKSPACE else "del sistema"
    if where == ONTOLOGY:
        return "de tu ontología" if kind == KIND_WORKSPACE else "de OntoPriv"
    return "de la base"


def part_text(part: dict, kind: str) -> str:
    if part["origin"] == ONTOLOGY:
        return "tu ontología"
    if part["origin"] == BASE:
        return "la base del sistema"
    return "escrito por el sistema" if kind == KIND_WORKSPACE else "la base del sistema"


def _counts(row: dict, kind: str) -> str:
    parts = [f"{row[w]} {origin_text(w, kind, row[w])}" for w in (NEW, ONTOLOGY, BASE) if row[w]]
    return ", ".join(parts)


def render_log(report: dict, examples: int | None = None) -> str:
    kind, s = report["kind"], report["summary"]
    title = "de la base del sistema" if kind == KIND_BASE else f"del espacio {report['name']}"
    inherited = s["violations"][ONTOLOGY] + s["violations"][BASE]
    warnings = sum(s["warnings"].values())
    e, shapes = report["engine"], report["shapes"]
    lines = [f"Inspector de forma (SHACL) {title}",
             f"  Fecha: {report['created']}",
             f"  Motor: {e['name']} {e['version']} (sin inferencia, SHACL-AF); reglas "
             f"{shapes['file']} v{shapes['version']} (sha256 {shapes['sha256'][:12]})",
             f"  Archivos ({report['triples']} tripletas):"]
    lines += [f"    - {p['file']} ({part_text(p, kind)}, {p['triples']} tripletas)"
              for p in report["parts"]]
    lines += [f"    - {r['file']} (solo para saber qué es heredado)"
              for r in report["references"]]
    lines.append(f"  Resultado: {report['status']}: "
                 f"{sh.counted(s['violations'][NEW], 'violación nueva', 'violaciones nuevas')}, "
                 f"{sh.counted(inherited, 'heredada', 'heredadas')}, "
                 f"{sh.counted(warnings, 'advertencia', 'advertencias')} "
                 f"({str(report['seconds']).replace('.', ',')} s)")
    c = report["completeness"]
    share = {key: f"{c[key]} ({_percent(c[key], c['concepts'])})"
             for key in ("label_es", "label_en", "definition")}
    lines.append(f"  Completitud de {sh.counted(c['concepts'], 'concepto', 'conceptos')}: "
                 f"etiqueta en español {share['label_es']}, en inglés {share['label_en']}, "
                 f"definición {share['definition']}")
    lines.append("  Reglas:")
    for rule_id, row in s["rules"].items():
        n = row[NEW] + row[ONTOLOGY] + row[BASE]
        mark = "ok" if n == 0 else (RED if row["severity"] == sh.VIOLATION and row[NEW]
                                    else AMBER)
        lines.append(f"    [{mark}] {row['name']}: {n}" + (f" ({_counts(row, kind)})" if n else ""))
        shown = [f for f in report["findings"] if f["rule"] == rule_id]
        for f in shown[:examples] if examples is not None else shown:
            lines.append(f"         {sh.local_name(f['focus'])}: {f['message']} "
                         f"[{origin_text(f['origin'], kind)}]")
        if examples is not None and len(shown) > examples:
            lines.append(f"         … y {len(shown) - examples} más (en {LOG_FILE})")
    return "\n".join(lines)


def _open(target: str, cfg: dict):
    from src.workspace.store import MANIFEST_FILE, load_workspace
    path = Path(target)
    if (path / MANIFEST_FILE).exists():
        return load_workspace(path.name, root=path.parent)
    return load_workspace(target, cfg=cfg)


def _run(argv: list[str] | None = None) -> int:
    from src.config import load_config
    from src.transform.convert import ConversionError
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(prog="py -m src.validation.shape_inspector",
                                     description="Inspector de forma: corre las reglas SHACL "
                                                 "sobre la base o sobre un espacio de trabajo "
                                                 "y guarda el informe y el log.")
    parser.add_argument("workspace", nargs="?", help="id o carpeta del espacio de trabajo")
    parser.add_argument("--base", action="store_true", help="inspeccionar la base del sistema")
    args = parser.parse_args(argv)
    if args.base == bool(args.workspace):
        print("Error: indica un espacio de trabajo o --base.")
        return 1
    cfg = load_config()
    try:
        if args.base:
            report = inspect_base(cfg)
            from src.alignment.inherit import base_paths
            folder = base_paths(cfg)["candidates"].parent
        else:
            ws = _open(args.workspace, cfg)
            report = inspect_workspace(ws, cfg)
            folder = ws.work_dir
    except (FileNotFoundError, ValueError, WorkspaceError, ConversionError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_log(report, examples=EXAMPLES))
    print(f"  Guardado en: {folder / REPORT_FILE} y {folder / LOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())