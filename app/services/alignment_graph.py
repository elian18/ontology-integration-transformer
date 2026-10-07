"""Bridge for the "Grafo alineado" tab of the "Alineación DPV" view (Sprint 5, S5-T07).

Writes the aligned graph with ``src.alignment.write`` (S5-T06) and offers the files and a zip
ready for Protégé (``src.alignment.bundle``). The paths come from the view
(``services.mapping_review.default_paths()``), so both tabs always read and write the same
files; the outputs go next to the candidates file (data/output).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.alignment.decisions import read_log, review_settings
from src.alignment.export import load_candidate_file
from src.alignment.write import (collect, write_alignment, ALIGNMENT_FILE, ALIGNMENT_MANIFEST,
                                 ALIGNMENT_CSV)
from src.alignment.bundle import build_bundle, BUNDLE_FILE

MIME = {ALIGNMENT_FILE: "application/rdf+xml", ALIGNMENT_MANIFEST: "application/json",
        ALIGNMENT_CSV: "text/csv", BUNDLE_FILE: "application/zip"}


def output_dir(paths: dict) -> Path:
    """The outputs folder: where the candidates file lives (data/output)."""
    return Path(paths["candidates"]).parent


def readiness(state: dict) -> dict:
    """Can the graph be written? (from the review state the page already loaded)."""
    p = state["progress"]
    ready = p["pending"] == 0
    if ready:
        msg = (f"Los {p['total']} conceptos están revisados: {p['aligned']} alineados y "
               f"{p['no_match']} sin correspondencia ({p['mappings']} correspondencias).")
    else:
        msg = (f"Faltan {p['pending']} de {p['total']} conceptos por validar. El grafo se "
               f"escribe cuando no queda nada por validar.")
    return {"ready": ready, "message": msg, "pending": p["pending"], "total": p["total"]}


def write_graph(paths: dict, cfg: dict | None = None) -> tuple[bool, str]:
    """Write the graph, manifest and CSV into the outputs folder."""
    settings = review_settings(cfg)
    try:
        content = collect(load_candidate_file(paths["candidates"]), read_log(paths["log"]),
                          settings["source_id"])
        result = write_alignment(content, output_dir(paths), source_id=settings["source_id"],
                                 log_path=paths["log"])
    except ValueError as exc:
        return False, f"No se escribió el grafo: {exc}"
    m = result.manifest
    return True, (f"Grafo escrito: {m['mappings']['total']} correspondencias SKOS, "
                  f"{m['ai_concepts']['new_declared']} conceptos nuevos y {m['triples']} "
                  f"tripletas.")


def current_graph(paths: dict) -> dict | None:
    """The graph already written (by this page or by ``py -m src.alignment.write``), with the
    bytes of each file and of the Protégé zip; None if it has not been written yet."""
    out = output_dir(paths)
    if not (out / ALIGNMENT_FILE).exists():
        return None
    manifest = {}
    if (out / ALIGNMENT_MANIFEST).exists():
        manifest = json.loads((out / ALIGNMENT_MANIFEST).read_text(encoding="utf-8"))
    files = {name: (out / name).read_bytes()
             for name in (ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV)
             if (out / name).exists()}
    bundle, included, missing = build_bundle(out)
    files[BUNDLE_FILE] = bundle
    return {"manifest": manifest, "files": files, "bundle_includes": included,
            "bundle_missing": missing, "out_dir": out}


def summary_rows(manifest: dict) -> list[dict]:
    """Relation counts for the small table of the tab."""
    labels = {"skos:exactMatch": "equivalente", "skos:closeMatch": "casi equivalente",
              "skos:broadMatch": "el DPV es más general",
              "skos:narrowMatch": "el DPV es más específico", "skos:relatedMatch": "relacionado"}
    by = (manifest.get("mappings") or {}).get("by_relation") or {}
    return [{"Tipo SKOS": rel, "Significado": labels.get(rel, ""), "Correspondencias": n}
            for rel, n in by.items()]


def stale(state: dict, manifest: dict) -> bool:
    """True when the review changed after the graph was written (counts differ)."""
    p = state["progress"]
    m = manifest.get("mappings") or {}
    c = manifest.get("concepts") or {}
    return m.get("total") != p["mappings"] or c.get("no_match") != p["no_match"]