"""The law of an upload: clean it, split it into articles and index it in its workspace (S7-T07).

Modes "solo ley" (S7-T10) and "ley + ontología" (S7-T11) start here. A person uploads a
data-protection law (.pdf or .txt, Spanish or English: the LOPDP, the GDPR, the Colombian law
1581 ...) and, without doing anything else:

1. ``segment``: the text is cleaned (``law_text``: page headers and footers, EUR-Lex markers,
   ligatures) and split into numbered articles (``legal_segmenter``: every heading style, the
   numbers must follow each other). The result, with the language, the skipped numbers and
   what the cleaning removed, goes to ``work/law-articles.json``. A text without numbered
   articles is refused with a message in Spanish.
2. ``index``: one vector per article in the collection "law" of the ChromaDB of the WORKSPACE
   (``<workspace>/chroma``). The base database (``chroma_db``: normativa, dpv, decisions) is
   never written. The collection carries its manifest and is rebuilt only when the law, the
   segmenter or the embedding model change (the fingerprint of S7-T01), so uploading the same
   law again costs nothing.

``retrieve_law_articles`` searches that collection: the extraction (S7-T09) and the evidence
of the "ley + ontología" mode (S7-T11) will use it.

CLI: ``py -m src.ingest.law_upload ARCHIVO [--consulta "texto"]``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest.law_text import clean_law_file
from src.ingest.legal_segmenter import SEGMENTER_VERSION, segment_articles

LAW_COLLECTION = "law"
ARTICLES_FILE = "law-articles.json"
CHROMA_DIR = "chroma"
LANGUAGE_LABELS = {"es": "español", "en": "inglés"}


def chroma_path(ws) -> Path:
    """Folder of the ChromaDB of a workspace (never the base one)."""
    return ws.path / CHROMA_DIR


def _law_entry(ws) -> dict:
    from src.workspace.store import KIND_LAW, WorkspaceError
    entry = ws.manifest.get("inputs", {}).get(KIND_LAW)
    if entry is None:
        raise WorkspaceError("Este espacio de trabajo no tiene una ley subida.")
    return entry


# --------------------------------------------------------------------------- 1. segment

def load_articles(ws) -> dict | None:
    path = ws.work_dir / ARTICLES_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def segment_workspace_law(ws) -> dict:
    """Clean and segment the uploaded law; save ``work/law-articles.json`` and mark the step."""
    from src.workspace.store import DONE, ERROR, KIND_LAW, RUNNING, WorkspaceError
    entry = _law_entry(ws)
    ws.mark_step("segment", RUNNING)
    try:
        text, cleaning = clean_law_file(ws.input_path(KIND_LAW))
        summary = segment_articles(text, source=entry["original_name"])
    except Exception as exc:
        ws.mark_step("segment", ERROR, str(exc)[:300])
        raise
    if not summary["n_articles"]:
        message = (f"No se encontraron artículos numerados en {entry['original_name']}. El "
                   "sistema reconoce encabezados como 'Art. 1.-', 'Artículo 1.', 'Artículo 1°.' "
                   "o 'Article 1' al inicio de una línea.")
        ws.mark_step("segment", ERROR, message)
        raise WorkspaceError(message)
    result = {"segmenter_version": SEGMENTER_VERSION, "file": entry["original_name"],
              "law_sha256": entry["sha256"], "cleaning": cleaning, **summary}
    ws.work_dir.mkdir(parents=True, exist_ok=True)
    (ws.work_dir / ARTICLES_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    ws.mark_step("segment", DONE, f"{result['n_articles']} artículos "
                                  f"({LANGUAGE_LABELS[result['language']]})",
                 articles=result["n_articles"], language=result["language"],
                 missing=len(result["missing"]))
    return result


def current_articles(ws) -> dict:
    """The segmentation of the workspace, redone if missing or made by another segmenter."""
    result = load_articles(ws)
    if result is None or result.get("segmenter_version") != SEGMENTER_VERSION:
        result = segment_workspace_law(ws)
    return result


# --------------------------------------------------------------------------- 2. index

def article_records(result: dict) -> list[dict]:
    """One record per article for the collection "law" (id ``art-<number>``)."""
    return [{"id": f"art-{a['number']}", "document": a["text"],
             "metadata": {"number": a["number"], "title": a["title"],
                          "source": result["file"], "char_start": a["char_start"],
                          "char_end": a["char_end"]}}
            for a in result["articles"]]


def law_fingerprint(result: dict, model_name: str) -> str:
    from src.alignment.knowledge import fingerprint
    return fingerprint(model_name, result["law_sha256"], f"segmenter-{SEGMENTER_VERSION}")


def index_workspace_law(ws, *, embed_fn=None, model_name: str | None = None) -> dict:
    """Index the articles in the workspace's ChromaDB (only if needed); mark the step."""
    from src.alignment.knowledge import embedding_model_name, ensure_collection
    from src.workspace.store import DONE, ERROR, RUNNING
    result = current_articles(ws)
    if embed_fn is None:                       # lazy: the fast tests never load the model
        from src.ai.rag.embeddings import embed as embed_fn
    model_name = model_name or embedding_model_name()
    ws.mark_step("index", RUNNING)
    try:
        report = ensure_collection(LAW_COLLECTION, law_fingerprint(result, model_name),
                                   lambda: article_records(result), embed_fn=embed_fn,
                                   model_name=model_name, path=chroma_path(ws))
    except Exception as exc:
        ws.mark_step("index", ERROR, str(exc)[:300])
        raise
    ws.mark_step("index", DONE, f"{report['items']} artículos indexados",
                 items=report["items"], built=report["built"])
    return report


def prepare_law(ws, *, embed_fn=None, model_name: str | None = None, progress=None) -> dict:
    """Both steps for a workspace with a law: ``{"articles": ..., "index": ...}``."""
    from src.workspace.store import STEP_LABELS
    say = progress or (lambda label: None)
    say(STEP_LABELS["segment"])
    articles = current_articles(ws)
    say(STEP_LABELS["index"])
    index = index_workspace_law(ws, embed_fn=embed_fn, model_name=model_name)
    return {"articles": articles, "index": index}


def retrieve_law_articles(ws, query: str, top_k: int | None = None, *, embed_fn=None) -> list:
    """The articles of the uploaded law closest in meaning to ``query``."""
    from src.ai.rag.retriever import retrieve_articles
    return retrieve_articles(query, top_k, name=LAW_COLLECTION, path=chroma_path(ws),
                             embed_fn=embed_fn)


# --------------------------------------------------------------------------- console / CLI

def render_console(ws, prepared: dict, hits: list | None = None, query: str | None = None) -> str:
    r, index = prepared["articles"], prepared["index"]
    c = r["cleaning"]
    arts = r["articles"]
    missing = (f"faltan {', '.join(map(str, r['missing']))}" if r["missing"]
               else "no falta ninguno")
    lines = ["Ley subida (S7-T07)",
             f"  Archivo: {r['file']} ({c['pages']} páginas) · idioma: "
             f"{LANGUAGE_LABELS[r['language']]}",
             f"  Limpieza: {c['removed_lines']} líneas de cabecera o pie de página, "
             f"{c['markers']} marcas de EUR-Lex, {c['spaced_headings']} títulos con letras "
             f"espaciadas, {c['ligatures']} ligaduras",
             f"  Artículos: {r['n_articles']} (del {arts[0]['number']} al {arts[-1]['number']}; "
             f"{missing}) · encabezados descartados: {r['rejected']}",
             "  Primeros: " + "; ".join(f"{a['number']}. {a['title'] or '(sin título)'}"
                                        for a in arts[:3]),
             f"  Último: {arts[-1]['number']}. {arts[-1]['title'] or '(sin título)'}",
             f"  Índice: colección \"{LAW_COLLECTION}\" en {chroma_path(ws)}, {index['items']} "
             f"artículos ({'construida ahora' if index['built'] else 'ya estaba al día'})",
             f"  Espacio de trabajo: {ws.id}"]
    if hits is not None:
        lines.append(f"  Consulta \"{query}\":")
        lines.extend(f"    Art. {h['number']}. {h['title'] or '(sin título)'} "
                     f"(distancia {h['distance']:.3f})" for h in hits)
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.workspace.store import WorkspaceError, open_law
    parser = argparse.ArgumentParser(
        prog="py -m src.ingest.law_upload",
        description="Separa en artículos una ley subida y la indexa en su espacio de trabajo "
                    "(S7-T07).")
    parser.add_argument("file", help="ley en .pdf o .txt")
    parser.add_argument("--consulta", help="texto para probar la búsqueda en la ley indexada")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        ws = open_law(path.name, path.read_bytes())
        prepared = prepare_law(ws)
        hits = retrieve_law_articles(ws, args.consulta, 3) if args.consulta else None
    except (WorkspaceError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(ws, prepared, hits, args.consulta))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())