"""Automatic check of the "solo ontología" mode on an ontology from outside the base (S6-T09).

The threshold was calibrated on the base (S6-T06) and the inheritance rules were measured there
(S6-T05). This module measures the automatic proposer on an ontology nobody in the project
wrote, GDPRtEXT (ADAPT Centre, CC-BY-4.0; id12 of the scoping review), and nobody labels
anything:

- The reference already exists. The concepts of the upload that inherited a validated decision
  (S6-T05) are concepts whose answer a person gave while building the base (Sprint 5). In
  GDPRtEXT: 22 concepts (18 by the same name, 4 very similar), 21 with a DPV term and 1 without.
- They are solved again AS IF THEY WERE NEW, with the same steps the web applies to the rest
  (S6-T06): ranking, best candidate only, threshold ``auto.threshold``, and the AI types it or
  says "none". Each answer is then compared with the person's.
- It also records where the correct DPV term sits among the first ``DEPTH`` (10) candidates:
  how many answers a step that chooses AMONG candidates could reach. That is the baseline the
  Sprint 7 redesign (RAG + LLM that decides) will be measured against.

Nothing of the user's alignment changes: the check keeps its own candidates and AI types in
``work/external-check/`` (a second run reuses them and makes no LLM call) and writes
``work/external-check.json``; the CLI also saves a copy next to the base outputs
(``data/output/<file>-external-check.json``) for the CHANGELOG and the thesis.

CLI:
    py -m src.alignment.external_check data/input/gdprtext-ontology.ttl [--sin-ia]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.alignment.calibrate import auto_settings
from src.alignment.candidates import alignment_settings
from src.alignment.export import (CANDIDATES_JSON, build_candidate_table, load_candidate_file,
                                  write_candidate_files)
from src.alignment.inherit import METHOD_LABELS
from src.alignment.inherit import load_result as inherited_result
from src.alignment.justify import RELATION_UNTYPED, justify_settings
from src.alignment.propose import (STATUS_AI_NONE, STATUS_PROPOSED, ask_ai, decide,
                                   load_dpv_targets, mark_for_ai)
from src.alignment.sources import build_alignment_sources
from src.ingest.ontology_loader import load_ontology

CHECK_DIR = "external-check"
RESULT_FILE = "external-check.json"
DEPTH = 10
HITS_AT = (1, 3, 10)

HIT = "acierto"
WRONG = "error"
ABSTAIN_OK = "sin_correspondencia_bien"
LOST_THRESHOLD = "perdida_por_umbral"
LOST_AI = "descartada_por_la_ia"
LOST_RANK = "fuera_del_puesto_1"
VERDICT_LABELS = {HIT: "acierto", WRONG: "error",
                  ABSTAIN_OK: "sin correspondencia, bien detectado",
                  LOST_THRESHOLD: "el correcto era el 1.º pero quedó bajo el umbral",
                  LOST_AI: "el correcto era el 1.º pero la IA dijo none",
                  LOST_RANK: "el correcto no estaba en el puesto 1"}
VERDICT_SHORT = {HIT: "acierto", WRONG: "error", ABSTAIN_OK: "sin correspondencia (bien)",
                 LOST_THRESHOLD: "bajo el umbral", LOST_AI: "la IA dijo none",
                 LOST_RANK: "no era el 1.º"}


# --------------------------------------------------------------------------- inputs

def _local(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def known_answers(inherited: dict) -> list[dict]:
    """The inherited concepts with the decision a person took for them (the reference)."""
    out = []
    for item in inherited.get("inherited", []):
        mappings = item.get("mappings", [])
        out.append({"key": item["key"], "name": item["name"], "method": item["method"],
                    "base_name": item.get("base_name"),
                    "gold": {m["dpv_iri"]: m["relation"] for m in mappings},
                    "gold_names": {m["dpv_iri"]: m.get("dpv_name") or _local(m["dpv_iri"])
                                   for m in mappings}})
    return sorted(out, key=lambda k: k["name"].lower())


def known_sources(ws, keys: set[str]):
    """The known concepts as SourceConcepts, exactly as the proposer would read them."""
    sources = build_alignment_sources(load_ontology(ws.input_path()), proposals={})
    return replace(sources, concepts=[c for c in sources.concepts if c.key in keys])


def check_candidates(ws, sources, *, targets=None, embed_fn=None, cfg=None,
                     depth: int = DEPTH) -> dict:
    """Rank ``depth`` DPV candidates per known concept, in the check's own folder.

    Reuses the saved file (and the AI types in it) when it holds the same concepts and depth."""
    folder = ws.work_dir / CHECK_DIR
    path = folder / CANDIDATES_JSON
    wanted = sorted(c.key for c in sources.concepts)
    if path.exists():
        data = load_candidate_file(path)
        meta = data.get("metadata", {})
        if meta.get("check_keys") == wanted and meta.get("check_depth") == depth:
            return data
    targets = targets if targets is not None else load_dpv_targets(cfg)
    settings = replace(alignment_settings(cfg), top_k=depth)
    table = build_candidate_table(sources, targets, embed_fn=embed_fn, settings=settings)
    for row in table.rows:
        row["needs_justification"] = False             # set later, only for rank 1 >= threshold
    table.metadata.update({"check_keys": wanted, "check_depth": depth})
    write_candidate_files(table, folder)
    return load_candidate_file(path)


# --------------------------------------------------------------------------- comparing

def _verdict(known: dict, entry: dict | None, gold_rank: int | None) -> str:
    proposed = entry is not None and entry["status"] == STATUS_PROPOSED
    if not known["gold"]:
        return WRONG if proposed else ABSTAIN_OK
    if proposed:
        return HIT if entry["dpv_iri"] in known["gold"] else WRONG
    if gold_rank == 1:
        return LOST_AI if entry is not None and entry["status"] == STATUS_AI_NONE \
            else LOST_THRESHOLD
    return LOST_RANK


def compare(known: list[dict], data: dict, entries: list[dict]) -> list[dict]:
    """One row per known concept: what the proposer did and how it compares with the person."""
    ranks: dict[str, dict[str, int]] = {}
    for r in data.get("rows", []):
        ranks.setdefault(r["concept_key"], {})[r["dpv_iri"]] = int(r["rank"])
    by_key = {e["key"]: e for e in entries}
    rows = []
    for k in known:
        found = [ranks.get(k["key"], {}).get(iri) for iri in k["gold"]]
        gold_rank = min((r for r in found if r is not None), default=None)
        entry = by_key.get(k["key"])
        verdict = _verdict(k, entry, gold_rank)
        relation = entry.get("relation") if entry else None
        relation_ok = (relation == k["gold"][entry["dpv_iri"]]
                       if verdict == HIT and relation not in (None, RELATION_UNTYPED) else None)
        rows.append({
            "key": k["key"], "name": k["name"], "method": k["method"],
            "base_name": k["base_name"],
            "gold": [{"dpv_name": k["gold_names"][iri], "relation": rel}
                     for iri, rel in k["gold"].items()],
            "gold_rank": gold_rank,
            "top1": entry["dpv_name"] if entry else None,
            "score": entry["score"] if entry else None,
            "status": entry["status"] if entry else None,
            "relation": relation, "relation_ok": relation_ok, "verdict": verdict})
    return rows


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def summarize(rows: list[dict], *, threshold: float, depth: int = DEPTH) -> dict:
    """Precision, coverage, where the correct term was, and why the rest was lost."""
    def block(subset: list[dict]) -> dict:
        aligned = [r for r in subset if r["gold"]]
        verdicts = Counter(r["verdict"] for r in subset)
        proposals = verdicts[HIT] + verdicts[WRONG]
        typed = [r for r in subset if r["relation_ok"] is not None]
        return {"concepts": len(subset), "aligned": len(aligned),
                "no_match": len(subset) - len(aligned),
                "proposals": proposals, "correct": verdicts[HIT],
                "precision": _ratio(verdicts[HIT], proposals),
                "coverage": _ratio(verdicts[HIT], len(aligned)),
                "relation_typed": len(typed),
                "relation_ok": sum(r["relation_ok"] for r in typed),
                "no_match_ok": verdicts[ABSTAIN_OK],
                "verdicts": dict(verdicts),
                "hits": {str(n): sum(1 for r in aligned
                                     if r["gold_rank"] is not None and r["gold_rank"] <= n)
                         for n in HITS_AT if n <= depth},
                "not_in_depth": sum(1 for r in aligned if r["gold_rank"] is None)}

    return {"threshold": threshold, "depth": depth, "all": block(rows),
            "by_method": {m: block([r for r in rows if r["method"] == m])
                          for m in dict.fromkeys(r["method"] for r in rows)}}


# --------------------------------------------------------------------------- workspace

def check_workspace(ws, *, targets=None, embed_fn=None, llm=None, use_llm: bool | None = None,
                    cfg: dict | None = None, sleep_fn=None, depth: int = DEPTH) -> dict:
    """Solve the inherited concepts again as if they were new and compare (S6-T09)."""
    inherited = inherited_result(ws)
    if inherited is None:
        raise ValueError("Primero hay que heredar lo validado (paso S6-T05).")
    known = known_answers(inherited)
    if not known:
        raise ValueError("La ontología no heredó ningún concepto: no hay respuestas conocidas "
                         "con las que comparar.")
    settings = auto_settings(cfg)
    use_llm = settings["use_llm"] if use_llm is None else use_llm
    sources = known_sources(ws, {k["key"] for k in known})
    data = check_candidates(ws, sources, targets=targets, embed_fn=embed_fn, cfg=cfg, depth=depth)

    to_ai = mark_for_ai(data, settings["threshold"])
    ai = {"available": False, "error": "IA desactivada (--sin-ia)"}
    if use_llm and to_ai:
        js = replace(justify_settings(cfg), max_calls=settings["llm_max_calls"])
        ai = ask_ai(SimpleNamespace(work_dir=ws.work_dir / CHECK_DIR), data, llm=llm,
                    settings=js, sleep_fn=sleep_fn)
    elif use_llm:
        ai = {"available": True, "calls": 0, "rows_typed": 0}
    entries = decide(data, settings["threshold"])

    rows = compare(known, data, entries)
    result = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ontology": ws.manifest["inputs"]["ontology"]["original_name"],
        "workspace": ws.id,
        "embedding_model": data.get("metadata", {}).get("embedding_model")
        or os.environ.get("EMBEDDING_MODEL"),
        "llm_model": (data.get("metadata", {}).get("justification") or {}).get("llm_model"),
        "summary": summarize(rows, threshold=settings["threshold"], depth=depth),
        "ai": ai, "concepts": rows}
    (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return result


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def report_path(ws, folder: str | Path | None = None) -> Path:
    """``data/output/<file>-external-check.json`` (next to the base outputs, ignored by git)."""
    if folder is None:
        from src.alignment.inherit import base_paths
        folder = base_paths()["candidates"].parent
    stem = Path(ws.manifest["inputs"]["ontology"]["stored_as"]).stem
    return Path(folder) / f"{stem}-external-check.json"


# --------------------------------------------------------------------------- console

def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:.0f} %"


def _gold_text(row: dict) -> str:
    if not row["gold"]:
        return "(sin correspondencia)"
    return ", ".join(f"{g['dpv_name']} ({g['relation'].replace('skos:', '')})"
                     for g in row["gold"])


def render_console(result: dict) -> str:
    s, a = result["summary"], result["summary"]["all"]
    hits = a["hits"]
    lines = [
        "Comprobación automática con una ontología externa (S6-T09)",
        f"  Ontología: {result['ontology']}",
        f"  Conceptos con respuesta conocida (heredados de la base): {a['concepts']} "
        f"({a['aligned']} con término del DPV, {a['no_match']} sin correspondencia)",
        f"  Se resolvieron como si fueran nuevos, con el diseño actual (puesto 1 + umbral "
        f"{s['threshold']:.2f} + IA):",
        f"    Propuestas: {a['proposals']}; correctas: {a['correct']} "
        f"(precisión {_pct(a['precision'])})",
        f"    Cobertura: {a['correct']} de {a['aligned']} ({_pct(a['coverage'])})",
        (f"    Tipo SKOS igual al de la persona: {a['relation_ok']} de {a['relation_typed']} "
         f"aciertos con tipo de la IA" if a["relation_typed"] else
         "    Tipo SKOS: no se midió (ningún acierto con tipo de la IA)"),
        f"    Sin correspondencia bien detectada: {a['no_match_ok']} de {a['no_match']}",
        f"  Dónde quedó el término correcto ({a['aligned']} conceptos con término):",
        "    " + " · ".join(f"{'puesto 1' if n == '1' else 'top ' + n}: {v}"
                            for n, v in hits.items())
        + f" · fuera del top {s['depth']}: {a['not_in_depth']}",
        "  Por qué no se acertó el resto:"]
    for verdict in (LOST_THRESHOLD, LOST_AI, LOST_RANK, WRONG):
        n = a["verdicts"].get(verdict, 0)
        if n:
            lines.append(f"    {VERDICT_LABELS[verdict]}: {n}")
    for method, b in s["by_method"].items():
        lines.append(f"  Solo '{METHOD_LABELS.get(method, method)}': {b['correct']} correctas de "
                     f"{b['proposals']} propuestas, cobertura {b['correct']} de {b['aligned']}")
    ai = result["ai"]
    if ai.get("available"):
        lines.append(f"  IA: {ai.get('calls', 0)} llamadas en esta corrida"
                     + (" (cuota agotada)" if ai.get("quota_exhausted") else ""))
    else:
        lines.append(f"  IA: no usada ({ai.get('error')})")
    lines.append("  Detalle (persona -> sistema):")
    for r in result["concepts"]:
        rank = f"puesto {r['gold_rank']}" if r["gold_rank"] else \
            ("-" if not r["gold"] else f"fuera del top {s['depth']}")
        top = f"{r['top1']} {r['score']:.2f}" if r["top1"] else "sin candidatos"
        rel = f" {r['relation'].replace('skos:', '')}" if r["relation"] and \
            r["relation"] != RELATION_UNTYPED else ""
        lines.append(f"    {r['name']}: {_gold_text(r)} | 1.º {top}{rel} | {rank} | "
                     f"{VERDICT_SHORT[r['verdict']]}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import inherit_for_workspace
    from src.workspace.store import WorkspaceError, open_ontology
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.external_check",
        description="Comprobación automática con una ontología externa (S6-T09).")
    parser.add_argument("file", help="ontología .rdf, .owl, .ttl o .jsonld")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM (sin tipo SKOS)")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        ws = open_ontology(path.name, path.read_bytes())
        if inherited_result(ws) is None:
            inherit_for_workspace(ws)
        result = check_workspace(ws, use_llm=False if args.sin_ia else None)
    except (WorkspaceError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    out = report_path(ws)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(render_console(result))
    print(f"  Guardado en: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())