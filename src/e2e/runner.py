"""The test runner of Sprint 11 (S11-T09): each case goes through what the web runs (the same
service), and what came out is written down by itself: the alignment, the validation, the calls
to the AI, the time and whether the four downloads say the same."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rdflib import Graph
from rdflib.compare import isomorphic

from app.services import align_ontology as svc
from src.alignment import user_alignment as ua
from src.console import counted
from src.transform.convert import OUTPUT_FORMATS, input_format
from src.workspace.store import CACHE_DIR, load_workspace, workspace_id, workspace_settings

TRANSFORM = "transform"
MODES = (svc.MODE_ONTOLOGY, svc.MODE_LAW, svc.MODE_BOTH, TRANSFORM)
MODE_NAMES = {**svc.MODES, TRANSFORM: svc.PATHS[svc.PATH_TRANSFORM]}
SPACES = "data/workspaces/_e2e"       # its own workspaces: the person's uploads are not touched
RUNS_FILE = "runs.jsonl"
SUMMARY_FILE = "resumen.md"
STATES = {key: svc.METRICS[label] for key, label in (       # the numbers of the web
    ("base", "Validadas en la base"), ("rule", "Aceptadas automáticamente"),
    ("user", svc.USER_METRIC), ("suggested", "Sugeridas"), ("no_match", "Sin correspondencia"),
    ("no_proposal", "Sin propuesta"))}
PARTS = {"shapes": "forma", "logic": "lógica", "questions": "preguntas", "audit": "IA revisora"}
COLUMNS = ("Caso", "Modo", "Resultado", "Conceptos", "Base", "Regla", "Sugeridas", "Sin corr.",
           "Sin prop.", "Validación", "Llamadas a la IA", "Artículos", "Descarga", "Segundos")


@dataclass(frozen=True)
class Case:
    id: str
    mode: str
    files: dict[str, Path] = field(default_factory=dict)       # {kind: path}
    control: bool = False            # nothing should align: what aligns is a false positive
    note: str = ""


def load_cases(path: Path, root: Path) -> list[Case]:
    cases = []
    for raw in yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]:
        if raw["mode"] not in MODES:
            raise ValueError(f"caso {raw['id']}: modo desconocido {raw['mode']}")
        files = {kind: root / raw[kind] for kind in ("ontology", "law") if raw.get(kind)}
        cases.append(Case(raw["id"], raw["mode"], files, bool(raw.get("control")),
                          raw.get("note", "")))
    return cases


def e2e_cfg(root: Path) -> dict:
    """The configuration with the workspaces of the runner in ``root``."""
    from src.config import load_config
    cfg = load_config()
    return {**cfg, "workspaces": {**(cfg.get("workspaces") or {}), "dir": str(root)}}


def share_cache(root: Path) -> int:
    """The answers of the AI the person's uploads already have, so a case does not ask again
    what was asked before; returns how many files it copied."""
    main = Path(workspace_settings()["root"]) / CACHE_DIR
    ours = root / CACHE_DIR
    ours.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(main.glob("*")) if main.is_dir() else []:
        if path.is_file() and not (ours / path.name).exists():
            shutil.copy2(path, ours / path.name)
            copied += 1
    return copied


@contextmanager
def pointing(cfg: dict):
    """The service of the web, with its workspaces where ``cfg`` says."""
    original = svc.workspace_cfg
    svc.workspace_cfg = lambda: cfg
    try:
        yield
    finally:
        svc.workspace_cfg = original


@contextmanager
def counting_ai():
    """Counts the requests to the LLM while the case runs (cached answers are not requests)."""
    from src.ai import llm_client
    cls = llm_client.LLMClient
    original, calls = cls.ask, [0]

    def ask(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)
    cls.ask = ask
    try:
        yield calls
    finally:
        cls.ask = original


def _uploads(case: Case) -> dict[str, tuple[str, bytes]]:
    return {kind: (path.name, path.read_bytes()) for kind, path in case.files.items()}


def _files(case: Case) -> list[dict]:
    return [{"kind": kind, "name": path.name, "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12]}
            for kind, path in case.files.items()]


def same_downloads(blobs: dict[str, bytes], reference: Graph | None = None) -> dict:
    """The triples of each download and whether all say the same (as the reference, if any)."""
    graphs = {k: Graph().parse(data=b, format=OUTPUT_FORMATS[k].rdflib_name)
              for k, b in blobs.items()}
    first = reference if reference is not None else graphs["turtle"]
    return {"triples": {k: len(g) for k, g in graphs.items()},
            "same": all(isomorphic(g, first) for g in graphs.values())}


def _aligned(ws_id: str, cfg) -> dict:
    ws = load_workspace(ws_id, cfg=cfg)
    manifest = ua.load_manifest(ws)
    by = manifest["summary"]["by_status"]
    validation_file = ws.output_dir / "validation.json"
    validation = (json.loads(validation_file.read_text(encoding="utf-8"))
                  if validation_file.exists() else None)
    articles = ws.work_dir / "law-articles.json"
    return {
        "workspace": ws_id, "version": manifest.get("alignment_version"),
        "concepts": manifest["summary"]["concepts"],
        **{key: sum(by.get(code, 0) for code in codes) for key, codes in STATES.items()},
        "validation": validation and {"status": validation["status"], **{
            part: block.get("status") for part, block in validation["inspectors"].items()}},
        "articles": (json.loads(articles.read_text(encoding="utf-8"))["n_articles"]
                     if articles.exists() else None),
        "recomputed": sorted(k for k, ran in (manifest.get("recomputed") or {}).items() if ran),
        "download": same_downloads({k: ua.single_file(ws, k) for k in OUTPUT_FORMATS})}


def _transformed(case: Case) -> dict:
    name, data = next(iter(_uploads(case).values()))
    downloads = svc.transform_downloads(name, data)
    source = Graph().parse(data=data, format=input_format(name))
    return {"download": same_downloads({d["key"]: d["data"]() for d in downloads}, source)}


def run_case(case: Case, *, fresh: bool = False, cfg=None, **run_kwargs) -> dict:
    """Runs ``case`` and returns its record (an error is a result too: it is written down)."""
    cfg = cfg if cfg is not None else svc.workspace_cfg()
    record = {"case": case.id, "mode": case.mode, "control": case.control,
              "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "files": _files(case)}
    if fresh and case.mode != TRANSFORM:
        root = Path(workspace_settings(cfg)["root"])
        shutil.rmtree(root / workspace_id(_uploads(case)), ignore_errors=True)
    start = time.perf_counter()
    with pointing(cfg), counting_ai() as calls:
        try:
            if case.mode == TRANSFORM:
                record.update(_transformed(case))
            else:
                record.update(_aligned(svc.run(case.mode, _uploads(case), **run_kwargs), cfg))
            record["result"] = "ok"
        except Exception as exc:                          # noqa: BLE001 - the case's result
            record.update(result="error", error=svc.error_message(exc))
    record.update(seconds=round(time.perf_counter() - start, 1), ai_calls=calls[0])
    return record


def _cell(record: dict, key: str) -> str:
    value = record.get(key)
    return "—" if value is None else str(value)


def row(r: dict) -> list[str]:
    if r["result"] != "ok":
        return [r["case"], MODE_NAMES[r["mode"]], f"error: {r['error']}", *["—"] * 7,
                str(r["ai_calls"]), "—", "—", str(r["seconds"])]
    v, d = r.get("validation"), r["download"]
    download = ("iguales" if d["same"] else "distintas") + f" ({d['triples']['turtle']} tripletas)"
    validation = "—" if not v else v["status"] + " (" + ", ".join(
        f"{PARTS.get(p, p)}: {s}" for p, s in v.items() if p != "status") + ")"
    return [r["case"], MODE_NAMES[r["mode"]], "ok" + (" (control)" if r["control"] else ""),
            *(_cell(r, k) for k in ("concepts", "base", "rule", "suggested", "no_match",
                                    "no_proposal")),
            validation, str(r["ai_calls"]), _cell(r, "articles"), download, str(r["seconds"])]


def summary(records: list[dict]) -> str:
    """The last run of each case, as a Markdown table."""
    last = {r["case"]: r for r in records}
    lines = ["# Pruebas del Sprint 11", "", "| " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(row(r)) + " |" for r in last.values()]
    return "\n".join(lines) + "\n"


def save(record: dict, folder: Path) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / RUNS_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    records = [json.loads(line) for line in (folder / RUNS_FILE).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    (folder / SUMMARY_FILE).write_text(summary(records), encoding="utf-8")
    return records


def render(r: dict) -> str:
    if r["result"] != "ok":
        return f"  {r['case']}: error en {r['seconds']} s: {r['error']}"
    if r["mode"] == TRANSFORM:
        return (f"  {r['case']}: los cuatro formatos "
                f"{'dicen lo mismo' if r['download']['same'] else 'NO dicen lo mismo'} que el "
                f"archivo ({r['download']['triples']['turtle']} tripletas), {r['seconds']} s")
    text = (f"  {r['case']}: {counted(r['concepts'], 'concepto', 'conceptos')}; base "
            f"{r['base']}, regla {r['rule']}, sugeridas {r['suggested']}; validación "
            f"{(r['validation'] or {}).get('status', '—')}; "
            f"{counted(r['ai_calls'], 'llamada', 'llamadas')} a la IA; {r['seconds']} s")
    if r["control"]:
        text += f"; posibles falsos positivos: {r['base'] + r['rule']}"
    return text


def _run(argv: list[str] | None = None) -> int:
    from src.config import ROOT
    parser = argparse.ArgumentParser(
        prog="py -m src.e2e.runner",
        description="Corre los casos de prueba como la web y anota los resultados en "
                    "data/output/e2e/s11 (runs.jsonl y resumen.md).")
    parser.add_argument("casos", type=Path, help="archivo .yaml con los casos")
    parser.add_argument("--solo", nargs="+", metavar="ID", help="corre solo estos casos")
    parser.add_argument("--limpio", action="store_true",
                        help="borra antes el espacio del caso (la caché de la IA se queda)")
    parser.add_argument("--salida", type=Path, default=ROOT / "data/output/e2e/s11")
    parser.add_argument("--espacios", type=Path, default=ROOT / SPACES,
                        help="carpeta de los espacios de las pruebas")
    args = parser.parse_args(argv)
    cases = load_cases(args.casos, ROOT)
    unknown = set(args.solo or ()) - {c.id for c in cases}
    if unknown:
        print(f"Error: casos que no están en {args.casos}: {', '.join(sorted(unknown))}")
        return 1
    print("Pruebas del Sprint 11 (S11-T09)")
    cfg = e2e_cfg(args.espacios)
    share_cache(args.espacios)
    for case in cases:
        if args.solo and case.id not in args.solo:
            continue
        missing = [str(p) for p in case.files.values() if not p.exists()]
        if missing:
            print(f"  {case.id}: falta {', '.join(missing)}; no se corrió")
            continue
        record = run_case(case, fresh=args.limpio, cfg=cfg)
        save(record, args.salida)
        print(render(record))
    print(f"  Espacios: {args.espacios}\n  Registro: {args.salida / RUNS_FILE}\n"
          f"  Resumen: {args.salida / SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
