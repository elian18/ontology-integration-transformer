# Competency questions answered with SPARQL (S9-T04): Apache Jena runs the .rq queries without
# inference (rdflib when there is no Java) and each answer is judged by the rule of the catalog.
# Every upload answers the generic questions of its mode (S9-T05); the AI says why the yellow
# ones are yellow (S9-T06).
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.console import counted
from src.validation import logic_inspector as li
from src.validation import question_catalog as qc
from src.validation import question_explainer as qe
from src.validation.shape_inspector import GREEN, YELLOW

QUERIES_DIR = Path(__file__).with_name("queries")
RUNNER = Path(__file__).with_name("QuestionRunner.java")
REPORT_FILE, LOG_FILE = "questions-report.json", "questions-log.txt"
JENA, RDFLIB = li.JENA, "rdflib"
GRAY, NOT_APPLICABLE = "gris", "no aplica"
BASE_SET, ORIGINAL_SET, WORKSPACE_SET = "base", "original", "workspace"
SHOWN = 3
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def query_path(qid: str) -> Path:
    return QUERIES_DIR / f"{qid}.rq"


def query_text(qid: str) -> str:
    return query_path(qid).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the engines

def choose_engine(cfg: dict | None = None, engine: str | None = None,
                  java: str | None = None) -> tuple[dict, li.Settings]:
    s = li.settings(cfg)
    wanted = {JENA: JENA, RDFLIB: li.OWLRL, None: s.engine}[engine]
    s = li.Settings(**{**s.__dict__, "engine": wanted})
    chosen = li.choose_engine(s, java)
    if chosen["name"] == JENA:
        return chosen, s
    reason = "elegido con --motor" if engine == RDFLIB else chosen["reason"]
    return {"name": RDFLIB, "reason": reason}, s


def _jena(engine: dict, sets: dict[str, list[Path]], ids: list[str], s: li.Settings,
          paths: dict[str, Path] | None = None, query_timeout: float | None = None) -> dict:
    home = engine["home"]
    limit = [f"-Dquestions.timeout.ms={int(query_timeout * 1000)}"] if query_timeout else []
    args = [s.java, f"-Dlog4j2.configurationFile={home / 'log4j2.properties'}", *limit,
            "-cp", str(home / "lib" / "*"), str(RUNNER),
            *[str((paths or {}).get(i) or query_path(i)) for i in ids], "++"]
    for name, files in sets.items():
        args += [name, *[Path(f).resolve().as_uri() for f in files], "--"]
    try:
        done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=s.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Jena tardó más de {s.timeout_seconds} s") from exc
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise RuntimeError("Jena falló: " + (lines[-1] if lines else f"código {done.returncode}"))
    data = json.loads(done.stdout)
    return {name: {"triples": r["triples"], "seconds": round(r["millis"] / 1000, 2),
                   "answers": r["answers"]} for name, r in data["sets"].items()}


def _cell(term) -> dict | None:
    if term is None:
        return None
    if isinstance(term, BNode):
        return {"t": "bnode"}
    if isinstance(term, URIRef):
        return {"t": "uri", "v": str(term)}
    lang = term.language or None
    dt = str(term.datatype) if term.datatype else str(RDF.langString if lang else XSD.string)
    return {"t": "literal", "v": str(term), "dt": dt, "lang": lang}


def _select(g: Graph, text: str, query_timeout: float | None) -> dict:
    result = g.query(text)
    if result.type != "SELECT":
        raise ValueError(f"no es una consulta SELECT ({result.type})")
    start, rows = time.perf_counter(), []
    for row in result:
        if query_timeout and time.perf_counter() - start > query_timeout:
            raise TimeoutError(f"la consulta tardó más de {query_timeout:g} s")
        rows.append([_cell(c) for c in row])
    return {"vars": [str(v) for v in result.vars], "rows": rows}


def _rdflib(sets: dict[str, list[Path]], ids: list[str], paths: dict[str, Path] | None = None,
            query_timeout: float | None = None) -> dict:
    from src.transform.convert import load_graph
    parsed: dict[Path, Graph] = {}
    runs = {}
    for name, files in sets.items():
        start = time.perf_counter()
        g = Graph()
        for f in map(Path, files):
            if f not in parsed:
                parsed[f] = load_graph(f)
            g += parsed[f]
        answers = {}
        for qid in ids:
            try:
                path = (paths or {}).get(qid) or query_path(qid)
                answers[qid] = _select(g, Path(path).read_text(encoding="utf-8"), query_timeout)
            except Exception as exc:  # a bad query or a parser limit: that question goes gray
                answers[qid] = {"error": f"{type(exc).__name__}: {exc}"}
        runs[name] = {"triples": len(g), "seconds": round(time.perf_counter() - start, 2),
                      "answers": answers}
    return runs


def run(sets: dict[str, list[Path]], ids: list[str], *, cfg: dict | None = None,
        engine: str | None = None, java: str | None = None,
        paths: dict[str, Path] | None = None,
        query_timeout: float | None = None) -> tuple[dict, dict]:
    """Every query on every set of files: (engine that ran, answers by set). ``paths``: the file of
    a query that is not in queries/ (a draft of the AI); ``query_timeout``: seconds per query."""
    chosen, s = choose_engine(cfg, engine, java)
    if chosen["name"] == JENA:
        try:
            runs = _jena(chosen, sets, ids, s, paths, query_timeout)
            return {k: v for k, v in chosen.items() if k != "home"}, runs
        except (RuntimeError, ValueError) as exc:          # ValueError: unreadable JSON
            if engine == JENA or s.engine == JENA:
                raise
            chosen = {"name": RDFLIB, "reason": str(exc)}
    chosen["version"] = version("rdflib")
    return chosen, _rdflib(sets, ids, paths, query_timeout)


# --------------------------------------------------------------------------- judging an answer

def _local(iri: str) -> str:
    return re.split(r"[#/]", iri.rstrip("/#"))[-1]


def row_texts(row: list) -> str:
    return " ".join(_local(c["v"]) if c["t"] == "uri" else c["v"]
                    for c in row if c and c["t"] != "bnode")


def row_key(row: list) -> tuple:
    """A row as a comparable value (blank nodes have no stable name)."""
    out = []
    for c in row:
        if c is None or c["t"] == "bnode":
            out.append((c or {}).get("t"))
        elif c["t"] == "uri":
            out.append(("uri", c["v"]))
        else:
            lang = (c.get("lang") or "").lower()
            out.append(("literal", c["v"], str(RDF.langString) if lang else c["dt"], lang))
    return tuple(out)


def rows_of(answer: dict) -> set:
    return {row_key(r) for r in answer.get("rows", [])}


def _column(answer: dict, name: str) -> list:
    if name not in answer["vars"]:
        return []
    i = answer["vars"].index(name)
    return [r[i] for r in answer["rows"]]


def judge(q: qc.Question, answer: dict | None) -> dict:
    out = {"id": q.id, "group": q.group, "question": q.question, "kind": q.kind,
           "criterion": qc.criterion_text(q), "rows": 0, "answer": [], "error": None}
    if answer is None or "error" in answer:
        reason = answer["error"] if answer else "la consulta no corrió"
        return {**out, "color": GRAY, "error": reason}
    rows = answer["rows"]
    texts = [row_texts(r) for r in rows]
    out["rows"], out["answer"] = len(rows), sorted({t for t in texts if t})[:SHOWN]   # any engine
    if q.kind == qc.VALUE:
        found = sorted({float(n.replace(",", ".")) for r in rows for c in r
                        if c and c["t"] == "literal" for n in NUMBER.findall(c["v"])})
        ok = q.expected.value in found
        return {**out, "color": GREEN if ok else YELLOW, "values": [f"{v:g}" for v in found]}
    if q.kind == qc.AT_LEAST:
        concepts = {json.dumps(c, sort_keys=True) for c in _column(answer, "concept") if c}
        return {**out, "color": GREEN if len(concepts) >= q.min else YELLOW,
                "count": len(concepts)}
    sources = [c["v"] for c in _column(answer, "source") if c and c["t"] == "literal"]
    found, missing = [], []
    for e in q.expected.elements:
        if any(qc.names(e, t) for t in texts):
            found.append({"name": e.name, "how": "nombre"})
        elif qc.found(q, e, [], sources):
            found.append({"name": e.name, "how": "artículo"})
        else:
            missing.append(e.name)
    ok = not missing if q.kind == qc.ALL else bool(found)
    return {**out, "color": GREEN if ok else YELLOW, "found": found, "missing": missing}


def not_applicable(q: qc.Question, mode: str) -> dict:
    return {"id": q.id, "group": q.group, "question": q.question, "kind": q.kind,
            "criterion": qc.criterion_text(q), "rows": 0, "answer": [], "error": None,
            "color": GRAY, "applies": False,
            "reason": f"no aplica a «{qc.MODES[mode]}»: este modo no produce lo que la "
                      f"pregunta necesita"}


def summary(results: list[dict]) -> dict:
    applies = [r for r in results if r.get("applies", True)]
    return {**{c: sum(r["color"] == c for r in applies) for c in (GREEN, YELLOW, GRAY)},
            NOT_APPLICABLE: len(results) - len(applies)}


def status_of(results: list[dict]) -> str:
    """Green when every question that applies is answered; a gray one is also to look at."""
    applies = [r for r in results if r.get("applies", True)]
    return GREEN if all(r["color"] == GREEN for r in applies) else YELLOW


# --------------------------------------------------------------------------- the base

def base_files(cfg: dict | None = None) -> dict[str, list[Path]]:
    """The base in each of its four formats, and the original OntoPriv; each with the DPV."""
    from src.alignment.inherit import base_paths
    from src.alignment.write import ALIGNMENT_FILE
    from src.config import ROOT, load_config
    from src.core.emit import CORE_FILE, PROFILE_FILE
    from src.transform.convert import OUTPUT_FORMATS
    cfg = cfg if cfg is not None else load_config()
    out = base_paths(cfg)["candidates"].parent
    dpv = Path(cfg["inputs"]["dpv"])
    dpv = dpv if dpv.is_absolute() else ROOT / dpv
    seed = Path(cfg["inputs"]["ontology"])
    seed = seed if seed.is_absolute() else ROOT / seed
    sets = {}
    for key, spec in OUTPUT_FORMATS.items():
        files = [out / Path(name).with_suffix(spec.extension).name
                 for name in (CORE_FILE, PROFILE_FILE, ALIGNMENT_FILE)]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Falta la base del sistema: {missing[0]} "
                                    f"(py -m src.transform.convert ... --to all)")
        sets[BASE_SET if key == "rdfxml" else f"{BASE_SET}-{key}"] = [*files, dpv]
    if not seed.exists():
        raise FileNotFoundError(f"Falta OntoPriv: {seed}")
    sets[ORIGINAL_SET] = [seed, dpv]
    return sets


def _shown(path: Path) -> str:
    from src.config import ROOT
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def compare(runs: dict, ids: list[str], one: str, other: str) -> list[dict]:
    """The questions whose rows differ between two sets of files."""
    out = []
    for qid in ids:
        a, b = runs[one]["answers"].get(qid, {}), runs[other]["answers"].get(qid, {})
        ra, rb = rows_of(a), rows_of(b)
        if ra != rb or ("error" in a) != ("error" in b):
            out.append({"id": qid, "only_in_" + one: len(ra - rb), "only_in_" + other: len(rb - ra)})
    return out


def inspect_base(cfg: dict | None = None, *, engine: str | None = None,
                 java: str | None = None, out_dir: str | Path | None = None, llm=None,
                 use_llm: bool | None = False, sleep_fn=time.sleep) -> dict:
    from src.alignment.inherit import base_paths
    from src.config import load_config
    cfg = cfg if cfg is not None else load_config()
    catalog = qc.load_catalog(qc.CATALOG_FILE)
    questions = qc.for_mode(catalog, "base")
    ids = [q.id for q in questions]
    sets = base_files(cfg)
    start = time.perf_counter()
    used, runs = run(sets, ids, cfg=cfg, engine=engine, java=java)
    seconds = round(time.perf_counter() - start, 2)
    results = [judge(q, runs[BASE_SET]["answers"].get(q.id)) for q in questions]
    paper = [q.id for q in questions if q.source == "paper"]
    original = {q["id"]: judge(next(x for x in questions if x.id == q["id"]),
                               runs[ORIGINAL_SET]["answers"].get(q["id"]))["color"]
                for q in results if q["id"] in paper}
    formats = {name: compare(runs, ids, BASE_SET, name) for name in sets
               if name.startswith(BASE_SET + "-")}
    folder = Path(out_dir) if out_dir else base_paths(cfg)["candidates"].parent
    folder.mkdir(parents=True, exist_ok=True)
    explained = qe.explain(catalog, results, runs[BASE_SET]["answers"], cfg=cfg, llm=llm,
                           use_llm=use_llm, cache_path=folder / qe.CACHE_FILE, sleep_fn=sleep_fn)
    report = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": used, "seconds": seconds,
        "catalog": {"file": _shown(qc.CATALOG_FILE), "version": catalog.version,
                    "sha256": catalog.sha256},
        "files": [_shown(f) for f in sets[BASE_SET]], "triples": runs[BASE_SET]["triples"],
        "results": results, "summary": summary(results), "explanations": explained,
        "original": {"files": [_shown(f) for f in sets[ORIGINAL_SET]],
                     "differences": compare(runs, paper, BASE_SET, ORIGINAL_SET),
                     "colors": original},
        "formats": {name.removeprefix(BASE_SET + "-"): diff for name, diff in formats.items()},
    }
    (folder / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (folder / LOG_FILE).write_text(render_log(report) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------- an upload

def workspace_files(ws, cfg: dict | None = None) -> list[Path]:
    """What the upload is (its ontology and the modules written for it) and the DPV; not the
    copy of OntoPriv-Core, which is the base and not the user's ontology."""
    from src.alignment.user_alignment import load_manifest
    from src.config import ROOT, load_config
    manifest = load_manifest(ws)
    if manifest is None:
        raise ValueError("Primero alinea la ontología: falta output/alignment-manifest.json.")
    cfg = cfg if cfg is not None else load_config()
    dpv = Path(cfg["inputs"]["dpv"])
    dpv = dpv if dpv.is_absolute() else ROOT / dpv
    return [ws.path / rel for kind, rel in manifest["files"].items() if kind != "core"] + [dpv]


def inspect_workspace(ws, cfg: dict | None = None, engine: str | None = None,
                      java: str | None = None, *, llm=None, use_llm: bool | None = False,
                      sleep_fn=time.sleep) -> dict:
    """The generic questions on an upload: the ones its mode cannot answer do not apply."""
    from src.workspace.store import shared_cache
    catalog = qc.load_catalog(qc.CATALOG_FILE)
    mode = ws.manifest.get("mode") or "ontology"
    generic = [q for q in catalog.questions if q.group == qc.GENERIC]
    applies = [q for q in generic if mode in q.modes]
    files = workspace_files(ws, cfg)
    start = time.perf_counter()
    used, runs = run({WORKSPACE_SET: files}, [q.id for q in applies], cfg=cfg, engine=engine,
                     java=java)
    answers = runs[WORKSPACE_SET]["answers"]
    results = [{**judge(q, answers.get(q.id)), "applies": True} if q in applies
               else not_applicable(q, mode) for q in generic]
    ws.work_dir.mkdir(parents=True, exist_ok=True)
    explained = qe.explain(catalog, results, answers, cfg=cfg, llm=llm, use_llm=use_llm,
                           cache_path=shared_cache(ws, qe.CACHE_FILE), sleep_fn=sleep_fn)
    report = {"workspace": ws.id, "mode": mode,
              "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "engine": used, "seconds": round(time.perf_counter() - start, 2),
              "catalog": {"file": _shown(qc.CATALOG_FILE), "version": catalog.version,
                          "sha256": catalog.sha256},
              "files": [f.relative_to(ws.path).as_posix() if f.is_relative_to(ws.path)
                        else _shown(f) for f in files],
              "triples": runs[WORKSPACE_SET]["triples"], "results": results,
              "summary": summary(results), "status": status_of(results),
              "explanations": explained}
    (ws.work_dir / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    (ws.work_dir / LOG_FILE).write_text(render_workspace(report) + "\n", encoding="utf-8")
    return report


def load_result(ws) -> dict | None:
    path = ws.work_dir / REPORT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console

def engine_text(engine: dict) -> str:
    if engine["name"] == JENA:
        return f"Apache Jena {engine['version']}, SPARQL sin inferencia (Java {engine['java']})"
    return f"rdflib {engine['version']}, SPARQL sin inferencia ({engine['reason']})"


def engine_short(engine: dict) -> str:
    return f"{'Apache Jena' if engine['name'] == JENA else 'rdflib'} {engine['version']}"


def answer_text(r: dict) -> str:
    if not r.get("applies", True):
        return r["reason"][0].upper() + r["reason"][1:]
    if r["color"] == GRAY:
        return f"No corrió: {r['error']}"
    if r["kind"] == qc.VALUE:
        values = ", ".join(r["values"]) or "ninguno"
        return f"Valores que trae: {values} · {'; '.join(r['answer']) or 'sin filas'}"
    if r["kind"] == qc.AT_LEAST:
        examples = "; ".join(r["answer"])
        n = r["count"]
        return counted(n, "concepto", "conceptos") + (f" (p. ej. {examples})" if examples else "")
    total = len(r["found"]) + len(r["missing"])
    names = ", ".join(f["name"] + (" (por artículo)" if f["how"] == "artículo" else "")
                      for f in r["found"]) or "ninguno"
    text = f"Trae {len(r['found'])} de {total}: {names}"
    return text + (f". Falta: {', '.join(r['missing'])}" if r["missing"] else "")


def counts_text(s: dict, total: int | None = None) -> str:
    text = f"{s[GREEN]} en verde, {s[YELLOW]} en amarillo y {s[GRAY]} en gris"
    na = s.get(NOT_APPLICABLE, 0)
    if total is not None:
        text += f" de las {total - na} que aplican" if na else f" de {total}"
    if na:
        text += f"; {counted(na, 'no aplica', 'no aplican')}"
    return text


def _head(title: str, report: dict) -> list[str]:
    return [title,
            f"  Fecha: {report['date']}",
            f"  Motor: {engine_text(report['engine'])}",
            f"  Catálogo: {report['catalog']['file']} v{report['catalog']['version']} "
            f"(sha256 {report['catalog']['sha256'][:12]})",
            f"  Archivos ({report['triples']} tripletas):",
            *(f"    - {f}" for f in report["files"]),
            f"  Resultado: {counts_text(report['summary'], len(report['results']))} "
            f"({report['seconds']} s)",
            f"  IA que explica las amarillas (prompt {prompt_name(report)}): "
            f"{qe.ai_text(report['explanations'])}"]


def prompt_name(report: dict) -> str:
    p = report["explanations"]["prompt"]
    return f"{p['name']}.v{p['version']}"


def _question_lines(rows: list[dict]) -> list[str]:
    lines = []
    for r in rows:
        mark = NOT_APPLICABLE if not r.get("applies", True) else r["color"]
        lines += [f"    [{mark}] {r['id']} {r['question']}", f"         {answer_text(r)}"]
        if r.get("explanation"):
            lines.append(f"         {why_text(r)}")
    return lines


def why_text(r: dict) -> str:
    return f"Por qué{' (IA)' if r['explained_by'] == qe.AI else ''}: {r['explanation']}"


def render_workspace(report: dict) -> str:
    lines = _head(f"Preguntas de competencia del espacio {report['workspace']} "
                  f"({qc.MODES[report['mode']]}): {report['status']}", report)
    return "\n".join(lines + ["", f"  {qc.GROUPS[qc.GENERIC]} ({len(report['results'])})",
                              *_question_lines(report["results"])])


def render_log(report: dict) -> str:
    from src.transform.convert import OUTPUT_FORMATS
    lines = _head("Preguntas de competencia de la base del sistema", report)
    for group, title in qc.GROUPS.items():
        rows = [r for r in report["results"] if r["group"] == group]
        lines += ["", f"  {title} ({len(rows)})", *_question_lines(rows)]
    o = report["original"]
    same = [qid for qid in o["colors"] if qid not in {d["id"] for d in o["differences"]}]
    lines += ["", f"  OntoPriv original ({', '.join(o['files'])}): las {len(o['colors'])} "
                  f"preguntas del artículo"]
    if not o["differences"]:
        lines.append("    dan las mismas filas que la base.")
    else:
        lines.append(f"    dan las mismas filas que la base en {len(same)}; cambian:")
        for d in o["differences"]:
            lines.append(f"    - {d['id']}: {d['only_in_base']} filas solo en la base, "
                         f"{d['only_in_original']} solo en el original "
                         f"(color en el original: {o['colors'][d['id']]})")
    labels = [OUTPUT_FORMATS[k].label for k in ("rdfxml", *report["formats"])]
    different = {d["id"] for diff in report["formats"].values() for d in diff}
    lines += ["", f"  Formatos ({', '.join(labels)}):"]
    if not different:
        lines.append(f"    los cuatro dan las mismas filas en las {len(report['results'])} preguntas.")
    for key, diff in report["formats"].items():
        for d in diff:
            lines.append(f"    - {OUTPUT_FORMATS[key].label}: {d['id']} da otras filas")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    from src.config import load_config
    from src.transform.convert import ConversionError
    parser = argparse.ArgumentParser(
        prog="py -m src.validation.questions",
        description="Responde las preguntas de competencia con SPARQL y guarda el informe y el log.")
    parser.add_argument("workspace", nargs="?", help="id o carpeta del espacio de trabajo")
    parser.add_argument("--base", action="store_true", help="responder sobre la base del sistema")
    parser.add_argument("--motor", choices=(JENA, RDFLIB), help="forzar Jena o rdflib")
    parser.add_argument("--sin-ia", action="store_true",
                        help="no pedirle a la IA que explique las amarillas (frase fija)")
    args = parser.parse_args(argv)
    use_llm = False if args.sin_ia else None
    if args.base == bool(args.workspace):
        print("Error: indica un espacio de trabajo o --base.")
        return 1
    from src.validation.shape_inspector import open_workspace
    from src.workspace.store import WorkspaceError
    cfg = load_config()
    try:
        if args.base:
            report = inspect_base(cfg, engine=args.motor, use_llm=use_llm)
            folder, text = base_paths(cfg)["candidates"].parent, render_log(report)
        else:
            ws = open_workspace(args.workspace, cfg)
            report = inspect_workspace(ws, cfg, engine=args.motor, use_llm=use_llm)
            folder, text = ws.work_dir, render_workspace(report)
    except (FileNotFoundError, ValueError, RuntimeError, ConversionError, qc.CatalogError,
            WorkspaceError) as exc:
        print(f"Error: {exc}")
        return 1
    print(text)
    print(f"  Guardado en: {folder / REPORT_FILE} y {folder / LOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
