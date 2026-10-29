from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from rdflib import Graph

from src.validation import shape_inspector as si
from src.validation.shape_inspector import (AMBER, BASE, GREEN, KIND_BASE, KIND_WORKSPACE, NEW,
                                            ONTOLOGY, RED, Part)
from src.validation.shapes import counted, local_name

REPORT_FILE = "reasoner-report.json"
LOG_FILE = "reasoner-log.txt"
JENA, OWLRL = "jena", "owlrl"
MIN_JAVA = 21
CHECKER = Path(__file__).with_name("JenaCheck.java")
FULL, INHERITED = "full", "inherited"
EXAMPLES = 3
REASONERS = {"rdfs": "RDFS", "owl-micro": "OWL Micro", "owl-mini": "OWL Mini", "owl": "OWL"}
KIND_LABELS = {"disjoint": "Un individuo está en dos clases disjuntas",
               "object_value": "Una propiedad de datos tiene como valor un recurso",
               "literal_value": "Una propiedad de objeto tiene como valor un texto",
               "literal_type": "Un valor no es del tipo que pide su rango",
               "other": "Otra contradicción"}
_IRI = re.compile(r"<?(https?://[^\s<>\"',)\[\]]+)>?")
# Jena checks literals against a range rdfs:Literal as if it were a datatype once a file uses it
# as one ("..."^^rdfs:Literal, as GDPRtEXT does); in RDF 1.1 every literal is an rdfs:Literal.
LITERAL_RANGE = "Datatype[http://www.w3.org/2000/01/rdf-schema#Literal]"
_BLANK = re.compile(r"\b[0-9a-f]{32}\b|_:\S+")


@dataclass(frozen=True)
class Settings:
    engine: str = "auto"
    jena_home: str = "tools/apache-jena-6.2.0"
    java: str = "java"
    reasoner: str = "owl-micro"
    timeout_seconds: int = 300


def settings(cfg: dict | None = None) -> Settings:
    from src.config import load_config
    block = (cfg if cfg is not None else load_config()).get("validation", {}) or {}
    s = Settings(**{k: v for k, v in block.items() if k in Settings.__dataclass_fields__})
    if s.engine not in ("auto", JENA, OWLRL) or s.reasoner not in REASONERS:
        raise ValueError("Valores inválidos en el bloque validation de config.yaml "
                         "(engine: auto, jena u owlrl; reasoner: rdfs, owl-micro, owl-mini u owl).")
    return s


# --------------------------------------------------------------------------- the engine

def _home(s: Settings) -> Path:
    from src.config import ROOT
    home = Path(s.jena_home)
    return home if home.is_absolute() else ROOT / home


def java_version(text: str) -> str | None:
    found = re.search(r'version "([0-9][^"]*)"', text)
    return found.group(1) if found else None


def _major(release: str) -> int:
    return int(release.split(".")[0])


def jena_version(home: Path) -> str | None:
    jars = sorted((home / "lib").glob("jena-core-*.jar"))
    return jars[0].stem.removeprefix("jena-core-") if jars else None


def choose_engine(s: Settings, java: str | None = None) -> dict:
    if s.engine == OWLRL:
        return {"name": OWLRL, "reason": "elegido en config.yaml (validation.engine)"}
    home = _home(s)
    jena = jena_version(home)
    if java is None and shutil.which(s.java):
        done = subprocess.run([s.java, "-version"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        java = java_version(done.stderr + done.stdout)
    if java is None:
        reason = f"no se encontró Java ({s.java})"
    elif _major(java) < MIN_JAVA:
        reason = f"Java {java}: Jena 6 necesita Java {MIN_JAVA} o superior"
    elif jena is None:
        reason = f"falta Apache Jena en {s.jena_home} (ver README)"
    else:
        return {"name": JENA, "version": jena, "java": java, "reasoner": s.reasoner,
                "home": home}
    if s.engine == JENA:
        raise RuntimeError(f"No se puede usar Jena: {reason}.")
    return {"name": OWLRL, "reason": reason}


def _jena(engine: dict, sets: dict[str, list[Part]], s: Settings) -> tuple[dict, list[str]]:
    home = engine["home"]
    args = [s.java, f"-Dlog4j2.configurationFile={home / 'log4j2.properties'}",
            "-cp", str(home / "lib" / "*"), str(CHECKER), s.reasoner]
    for name, parts in sets.items():
        args += [name, *[p.path.resolve().as_uri() for p in parts], "--"]
    try:
        done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=s.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Jena tardó más de {s.timeout_seconds} s") from exc
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise RuntimeError("Jena falló: " + (lines[-1] if lines else f"código {done.returncode}"))
    data = json.loads(done.stdout)
    runs = {name: {"triples": r["triples"], "seconds": round(r["millis"] / 1000, 2),
                   "reports": r["reports"]} for name, r in data["sets"].items()}
    notes = sorted({re.sub(r"^\S+\s+WARN\s+riot\s+::\s*", "", line).strip()
                    for line in done.stderr.splitlines() if " WARN " in line and "riot" in line})
    return runs, notes


def _owlrl(sets: dict[str, list[Part]]) -> tuple[dict, list[str]]:
    from owlrl import OWLRL_Semantics
    runs = {}
    for name, parts in sets.items():
        g = Graph()
        for p in parts:
            g += p.graph
        triples = len(g)
        start = time.perf_counter()
        closure = OWLRL_Semantics(g, False, False, rdfs=False)
        closure.closure()
        runs[name] = {"triples": triples, "seconds": round(time.perf_counter() - start, 2),
                      "reports": [{"error": True, "type": "OWL 2 RL", "description": m,
                                   "culprit": ""} for m in closure.error_messages]}
    return runs, []


# --------------------------------------------------------------------------- the findings

def kind_of(report: dict) -> str:
    text = report["description"].lstrip('"')
    if text.startswith(("Individual a member of disjoint classes", "Disjoint classes")):
        return "disjoint"
    if text.startswith("Object value for datatype property") or "non literal value" in text:
        return "object_value"
    if text.startswith("Literal value for object property"):
        return "literal_value"
    if (text.startswith(("Incorrectly typed literal", "Lexical value of the literal"))
            or report["type"] == "dtRange"):
        return "literal_type"
    return "other"


def nodes_of(report: dict) -> list[str]:
    found = []
    for iri in _IRI.findall(report["description"] + " " + (report.get("culprit") or "")):
        if iri not in found:
            found.append(iri)
    return found


def key_of(report: dict) -> tuple:
    return (report["error"], report["type"], _BLANK.sub("_", report["description"]))


def _finding(report: dict, origin: str) -> dict:
    nodes = nodes_of(report)
    return {"error": report["error"], "kind": kind_of(report), "type": report["type"],
            "description": report["description"].strip(), "nodes": nodes, "origin": origin}


def inherited_origin(parts: list[Part]) -> str:
    return ONTOLOGY if any(p.origin == ONTOLOGY for p in parts) else BASE


def status_of(summary: dict) -> str:
    if summary["errors"][NEW]:
        return RED
    if sum(summary["errors"].values()) or sum(summary["warnings"].values()):
        return AMBER
    return GREEN


def inspect(parts: list[Part], references: list[Part] = (), *, cfg: dict | None = None,
            engine: str | None = None, kind: str = KIND_WORKSPACE, name: str = "",
            java: str | None = None) -> dict:
    s = settings(cfg)
    if engine is not None:
        s = Settings(**{**s.__dict__, "engine": engine})
    chosen = choose_engine(s, java)
    inherited = list(references) if kind == KIND_BASE else [p for p in parts if p.origin != NEW]
    sets = {FULL: parts, INHERITED: inherited}
    start = time.perf_counter()
    if chosen["name"] == JENA:
        try:
            runs, notes = _jena(chosen, sets, s)
        except RuntimeError as exc:
            if s.engine == JENA:
                raise
            chosen = {"name": OWLRL, "reason": str(exc)}
            runs, notes = _owlrl(sets)
    else:
        runs, notes = _owlrl(sets)
    seconds = round(time.perf_counter() - start, 2)
    dismissed = {n: [x for x in r["reports"] if LITERAL_RANGE in x["description"]]
                 for n, r in runs.items()}
    for n, r in runs.items():
        r["reports"] = [x for x in r["reports"] if LITERAL_RANGE not in x["description"]]
    known = {key_of(r) for r in runs[INHERITED]["reports"]}
    old = inherited_origin(inherited)
    findings = [_finding(r, old if key_of(r) in known else NEW) for r in runs[FULL]["reports"]]
    summary = {"errors": {NEW: 0, ONTOLOGY: 0, BASE: 0},
               "warnings": {NEW: 0, ONTOLOGY: 0, BASE: 0},
               "kinds": {k: {"label": label, "errors": 0, "warnings": 0, NEW: 0, ONTOLOGY: 0,
                             BASE: 0} for k, label in KIND_LABELS.items()}}
    for f in findings:
        summary["errors" if f["error"] else "warnings"][f["origin"]] += 1
        row = summary["kinds"][f["kind"]]
        row["errors" if f["error"] else "warnings"] += 1
        row[f["origin"]] += 1
    engine_info = {k: v for k, v in chosen.items() if k != "home"}
    if chosen["name"] == OWLRL:
        engine_info["version"] = version("owlrl")
    return {
        "kind": kind, "name": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine_info,
        "parts": [{"kind": p.kind, "file": p.file, "origin": p.origin, "triples": len(p.graph),
                   "sha256": p.sha256} for p in parts],
        "inherited_parts": [{"kind": p.kind, "file": p.file} for p in inherited],
        "runs": {n: {k: v for k, v in r.items() if k != "reports"} | {
            "errors": sum(x["error"] for x in r["reports"]),
            "warnings": sum(not x["error"] for x in r["reports"])} for n, r in runs.items()},
        "seconds": seconds, "status": status_of(summary), "consistent": not any(
            f["error"] for f in findings),
        "summary": summary, "parser_notes": notes, "dismissed": len(dismissed[FULL]),
        "findings": findings,
    }


def _save(report: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    (folder / LOG_FILE).write_text(render_log(report) + "\n", encoding="utf-8")
    return report


def inspect_workspace(ws, cfg: dict | None = None, engine: str | None = None) -> dict:
    report = inspect(si.workspace_parts(ws), cfg=cfg, engine=engine, kind=KIND_WORKSPACE,
                     name=ws.id)
    return _save(report, ws.work_dir)


def inspect_base(cfg: dict | None = None, out_dir: str | Path | None = None,
                 engine: str | None = None) -> dict:
    from src.alignment.inherit import base_paths
    parts, references = si.base_parts(cfg)
    report = inspect(parts, references, cfg=cfg, engine=engine, kind=KIND_BASE,
                     name="OntoPriv-Core + perfil de Ecuador + alineación con el DPV")
    folder = Path(out_dir) if out_dir is not None else base_paths(cfg)["candidates"].parent
    return _save(report, folder)


def load_result(ws) -> dict | None:
    path = ws.work_dir / REPORT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console / log

def engine_text(engine: dict) -> str:
    if engine["name"] == JENA:
        return (f"Apache Jena {engine['version']}, razonador {REASONERS[engine['reasoner']]} "
                f"(Java {engine['java']})")
    return f"owlrl {engine['version']} (OWL 2 RL en Python): {engine['reason']}"


def _seconds(value: float) -> str:
    return str(value).replace(".", ",")


def example(f: dict) -> str:
    names = [local_name(n) for n in f["nodes"]]
    if f["kind"] == "disjoint" and len(names) >= 3:
        first, *classes = names if f["type"] != "OWL 2 RL" else [names[-1], *names[:-1]]
        return f"{first}: {' y '.join(sorted(classes[:2]))}"
    if f["kind"] in ("object_value", "literal_value", "literal_type") and len(names) >= 2:
        return f"{names[0]}: {', '.join(names[1:3])}"
    return f["description"].splitlines()[0][:160]


def render_log(report: dict, examples: int | None = None) -> str:
    kind, s, engine = report["kind"], report["summary"], report["engine"]
    title = "de la base del sistema" if kind == KIND_BASE else f"del espacio {report['name']}"
    inherited = s["errors"][ONTOLOGY] + s["errors"][BASE]
    warnings = sum(s["warnings"].values())
    lines = [f"Inspector de lógica (razonador) {title}",
             f"  Fecha: {report['created']}",
             f"  Motor: {engine_text(engine)}",
             f"  Archivos ({report['runs'][FULL]['triples']} tripletas):"]
    lines += [f"    - {p['file']} ({si.part_text(p, kind)}, {p['triples']} tripletas)"
              for p in report["parts"]]
    base_of = ", ".join(p["file"] for p in report["inherited_parts"]) or "nada"
    lines.append(f"  Para saber qué es heredado razonó también sobre: {base_of}")
    lines.append(f"  Resultado: {report['status']}: "
                 f"{counted(s['errors'][NEW], 'inconsistencia nueva', 'inconsistencias nuevas')}, "
                 f"{counted(inherited, 'heredada', 'heredadas')}, "
                 f"{counted(warnings, 'advertencia', 'advertencias')} "
                 f"({_seconds(report['seconds'])} s)")
    for note in report["parser_notes"]:
        lines.append(f"  Aviso del lector de Jena: {note}")
    if report["dismissed"]:
        lines.append(f"  Descartadas: {report['dismissed']} comparaciones de Jena con el rango "
                     "rdfs:Literal (la ontología lo usa como tipo de un texto; en RDF 1.1 todo "
                     "literal es un rdfs:Literal)")
    lines.append("  Tipos:")
    for key, row in s["kinds"].items():
        n = row["errors"] + row["warnings"]
        if n == 0:
            continue
        mark = RED if row[NEW] and row["errors"] else AMBER
        parts = [f"{row[w]} {si.origin_text(w, kind, row[w])}" for w in (NEW, ONTOLOGY, BASE)
                 if row[w]]
        what = "advertencias" if row["errors"] == 0 else "inconsistencias"
        lines.append(f"    [{mark}] {row['label']} ({what}): {n} ({', '.join(parts)})")
        shown = [f for f in report["findings"] if f["kind"] == key]
        for f in shown[:examples] if examples is not None else shown:
            lines.append(f"         {example(f)} [{si.origin_text(f['origin'], kind)}]")
        if examples is not None and len(shown) > examples:
            lines.append(f"         … y {len(shown) - examples} más (en {LOG_FILE})")
    if not report["findings"]:
        lines.append("    [ok] El razonador no encontró contradicciones.")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.config import load_config
    from src.transform.convert import ConversionError
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(prog="py -m src.validation.logic_inspector",
                                     description="Inspector de lógica: razona sobre la base o "
                                                 "sobre un espacio de trabajo, busca "
                                                 "contradicciones y guarda el informe y el log.")
    parser.add_argument("workspace", nargs="?", help="id o carpeta del espacio de trabajo")
    parser.add_argument("--base", action="store_true", help="inspeccionar la base del sistema")
    parser.add_argument("--motor", choices=(JENA, OWLRL), help="forzar Jena u owlrl")
    args = parser.parse_args(argv)
    if args.base == bool(args.workspace):
        print("Error: indica un espacio de trabajo o --base.")
        return 1
    cfg = load_config()
    try:
        if args.base:
            from src.alignment.inherit import base_paths
            report = inspect_base(cfg, engine=args.motor)
            folder = base_paths(cfg)["candidates"].parent
        else:
            ws = si.open_workspace(args.workspace, cfg)
            report = inspect_workspace(ws, cfg, engine=args.motor)
            folder = ws.work_dir
    except (FileNotFoundError, ValueError, RuntimeError, WorkspaceError, ConversionError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_log(report, examples=EXAMPLES))
    print(f"  Guardado en: {folder / REPORT_FILE} y {folder / LOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())