"""Knowledge collections the alignment decider retrieves from (S7-T01).

Until Sprint 6 the DPV candidates were computed in memory on every upload (the 1116 DPV terms
were embedded again each time) and the 552 decisions of the base were only used to inherit
when a concept was the same. The decider of Sprint 7 RETRIEVES from two ChromaDB collections
that are built once from the base and then only read:

- ``dpv``: the 1116 alignable DPV terms (972 classes, 144 properties). Document = the same
  text the Sprint 4 ranking embeds (``DpvTarget.semantic_text``: label + trimmed definition),
  so a stored vector is exactly the one the hybrid ranking would compute. Metadata: IRI, name,
  label, kind (class | property), definition and direct DPV parents.
- ``decisions``: the 552 decided base concepts (527 OntoPriv entities + 25 new LOPDP concepts)
  with what a person decided in Sprint 5: the DPV term(s) and SKOS relation, or "sin
  correspondencia". Document = the text the inheritance step compares
  (``BaseConcept.semantic_text``). These are the precedents the decider sees (S7-T04).

Both use cosine distance (1 - cosine similarity). Each collection carries its own manifest in
its ChromaDB metadata: a fingerprint of what it was built from (embedding model, the input
files and the version of the text builders), how many items it must hold and when it was
built. ``collection_status`` compares that manifest with the current inputs, so a stale or
half-built collection is rebuilt and an up-to-date one is left alone (no embeddings computed).

Uploads only READ these collections: they live in the base database (``chroma_db``), never in
a workspace.

CLI: ``py -m src.alignment.knowledge [--estado] [--reconstruir]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.ai.rag import vector_store

DPV_COLLECTION = "dpv"
DECISIONS_COLLECTION = "decisions"
TEXT_VERSION = 1                 # bump when dpv_records / decision_records change their text
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # same default as embeddings.py
BATCH_SIZE = 256

STATE_UP_TO_DATE = "up_to_date"
STATE_STALE = "stale"
STATE_INCOMPLETE = "incomplete"
STATE_MISSING = "missing"
STATE_LABELS = {STATE_UP_TO_DATE: "al día", STATE_STALE: "desactualizada",
                STATE_INCOMPLETE: "incompleta", STATE_MISSING: "no existe"}
ITEM_LABELS = {DPV_COLLECTION: "términos del DPV", DECISIONS_COLLECTION: "decisiones de la base"}


# --------------------------------------------------------------------------- records

def dpv_records(targets) -> list[dict]:
    """One record per DPV term: id = IRI, document = the text the ranking embeds."""
    return [{"id": t.iri, "document": t.semantic_text(),
             "metadata": {"iri": t.iri, "name": t.name, "label": t.label or "",
                          "kind": t.kind, "definition": t.definition or "",
                          "parents": json.dumps(list(t.parents), ensure_ascii=False)}}
            for t in targets.targets]


def decision_records(base) -> list[dict]:
    """One record per decided base concept, with what a person decided in Sprint 5."""
    out = []
    for c in base.concepts:
        mappings = [{"dpv_iri": m["dpv_iri"], "dpv_name": m.get("dpv_name"),
                     "relation": m.get("relation")} for m in c.mappings]
        out.append({"id": c.key, "document": c.semantic_text(),
                    "metadata": {"key": c.key, "iri": c.iri, "origin": c.origin,
                                 "name": c.name, "label": c.label or "",
                                 "definition": c.definition or "",
                                 "kinds": json.dumps(list(c.kinds)),
                                 "is_class": "class" in c.kinds,
                                 "is_property": "property" in c.kinds,
                                 "status": c.status, "n_mappings": len(mappings),
                                 "mappings": json.dumps(mappings, ensure_ascii=False),
                                 "aliases": json.dumps(list(c.aliases), ensure_ascii=False)}})
    return out


def parse_record_metadata(metadata: dict) -> dict:
    """Stored metadata back to Python values (the JSON fields become lists)."""
    out = dict(metadata or {})
    for key in ("parents", "kinds", "mappings", "aliases"):
        if isinstance(out.get(key), str):
            out[key] = json.loads(out[key])
    return out


# --------------------------------------------------------------------------- manifest

def embedding_model_name() -> str:
    import src.config  # noqa: F401  (loads .env)
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def file_digest(path: str | Path) -> str:
    """sha256 of a file (the inputs a collection was built from)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(model_name: str, *digests: str) -> str:
    """What a collection depends on: embedding model + input files + text version."""
    payload = json.dumps({"model": model_name, "inputs": list(digests),
                          "text_version": TEXT_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def collection_status(name: str, expected_fingerprint: str, *,
                      path: str | Path | None = None) -> dict:
    """Is the stored collection built from the current inputs, and complete?"""
    if not vector_store.exists(name, path):
        return {"name": name, "state": STATE_MISSING, "items": 0, "expected_items": None,
                "embedding_model": None, "built_at": None}
    meta = vector_store.metadata(name, path)
    items = vector_store.count(name=name, path=path)
    expected = meta.get("items")
    if meta.get("fingerprint") != expected_fingerprint:
        state = STATE_STALE
    elif expected is None or items != expected:
        state = STATE_INCOMPLETE
    else:
        state = STATE_UP_TO_DATE
    return {"name": name, "state": state, "items": items, "expected_items": expected,
            "embedding_model": meta.get("embedding_model"), "built_at": meta.get("built_at")}


# --------------------------------------------------------------------------- building

def build_collection(name: str, records: list[dict], *, embed_fn, fingerprint_value: str,
                     model_name: str, path: str | Path | None = None,
                     batch_size: int = BATCH_SIZE) -> int:
    """Rebuild a collection from scratch with its manifest; returns how many items it holds."""
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Hay ids repetidos en la colección {name}.")
    manifest = {"fingerprint": fingerprint_value, "items": len(records),
                "embedding_model": model_name, "text_version": TEXT_VERSION,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    vector_store.reset(name=name, path=path, space=vector_store.SPACE_COSINE, metadata=manifest)
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        docs = [r["document"] for r in chunk]
        vectors = [list(map(float, v)) for v in embed_fn(docs)]
        vector_store.upsert(ids=[r["id"] for r in chunk], docs=docs, embeddings=vectors,
                            metadatas=[r["metadata"] for r in chunk], name=name, path=path)
    return vector_store.count(name=name, path=path)


def ensure_collection(name: str, fingerprint_value: str, make_records: Callable[[], list[dict]],
                      *, embed_fn, model_name: str, path: str | Path | None = None,
                      force: bool = False) -> dict:
    """Build the collection only when it is missing, stale or incomplete (or ``force``).

    ``make_records`` is called only when a build is needed, so an up-to-date collection costs
    neither parsing the inputs nor computing embeddings."""
    before = collection_status(name, fingerprint_value, path=path)
    if before["state"] == STATE_UP_TO_DATE and not force:
        return {**before, "built": False, "previous_state": before["state"]}
    items = build_collection(name, make_records(), embed_fn=embed_fn,
                             fingerprint_value=fingerprint_value, model_name=model_name,
                             path=path)
    after = collection_status(name, fingerprint_value, path=path)
    return {**after, "items": items, "built": True, "previous_state": before["state"]}


def knowledge_inputs(cfg: dict | None = None) -> dict[str, list[Path]]:
    """Files each collection is built from (their changes make it stale)."""
    from src.config import ROOT, load_config
    from src.alignment.inherit import base_paths
    cfg = cfg if cfg is not None else load_config()
    dpv = Path((cfg.get("inputs", {}) or {}).get("dpv", "vocab/dpv.ttl"))
    base = base_paths(cfg)
    return {DPV_COLLECTION: [dpv if dpv.is_absolute() else ROOT / dpv],
            DECISIONS_COLLECTION: [base["candidates"], base["log"]]}


def expected_fingerprints(cfg: dict | None = None, model_name: str | None = None) -> dict:
    model_name = model_name or embedding_model_name()
    out = {}
    for name, files in knowledge_inputs(cfg).items():
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError("Falta el archivo de la base: " + ", ".join(missing))
        out[name] = fingerprint(model_name, *(file_digest(f) for f in files))
    return out


def build_knowledge(cfg: dict | None = None, *, embed_fn=None, model_name: str | None = None,
                    force: bool = False, path: str | Path | None = None,
                    targets=None, base=None) -> list[dict]:
    """Make sure "dpv" and "decisions" are up to date (S7-T01); returns one report per collection.

    ``targets`` / ``base`` can be passed already loaded (tests); otherwise they are read from
    the base files only if a build is needed."""
    if embed_fn is None:                       # lazy: the fast tests never load the model
        from src.ai.rag.embeddings import embed as embed_fn
    model_name = model_name or embedding_model_name()
    fps = expected_fingerprints(cfg, model_name)

    def make_dpv():
        from src.alignment.propose import load_dpv_targets
        return dpv_records(targets if targets is not None else load_dpv_targets(cfg))

    def make_decisions():
        from src.alignment.inherit import load_base
        return decision_records(base if base is not None else load_base(cfg=cfg))

    return [ensure_collection(DPV_COLLECTION, fps[DPV_COLLECTION], make_dpv, embed_fn=embed_fn,
                              model_name=model_name, path=path, force=force),
            ensure_collection(DECISIONS_COLLECTION, fps[DECISIONS_COLLECTION], make_decisions,
                              embed_fn=embed_fn, model_name=model_name, path=path, force=force)]


def knowledge_status(cfg: dict | None = None, *, model_name: str | None = None,
                     path: str | Path | None = None) -> list[dict]:
    fps = expected_fingerprints(cfg, model_name)
    return [collection_status(name, fps[name], path=path)
            for name in (DPV_COLLECTION, DECISIONS_COLLECTION)]


# --------------------------------------------------------------------------- console

def render_console(reports: list[dict]) -> str:
    """Spanish summary: one line per collection (after a build or a status check)."""
    lines = ["Colecciones de conocimiento para el decisor (S7-T01)"]
    for r in reports:
        what = ITEM_LABELS.get(r["name"], "documentos")
        if r.get("built") is True:
            how = f"construida ahora (antes: {STATE_LABELS[r['previous_state']]})"
        elif r.get("built") is False:
            how = "al día, no se tocó"
        else:                                     # status check only (--estado)
            how = STATE_LABELS[r["state"]]
            if r["state"] == STATE_INCOMPLETE:
                how += f" ({r['items']} de {r['expected_items']})"
        lines.append(f"  {r['name']:<10} {r['items']:>5} {what:<22} {how}")
    models = {r.get("embedding_model") for r in reports if r.get("embedding_model")}
    if models:
        lines.append(f"  Modelo de embeddings: {', '.join(sorted(models))}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.knowledge",
        description="Construye las colecciones 'dpv' y 'decisions' de ChromaDB (S7-T01).")
    parser.add_argument("--estado", action="store_true",
                        help="solo muestra si están al día, sin construir nada")
    parser.add_argument("--reconstruir", action="store_true",
                        help="las vuelve a construir aunque estén al día")
    args = parser.parse_args(argv)
    try:
        if args.estado:
            print(render_console(knowledge_status()))
            return 0
        print("Revisando las colecciones (si hay que construirlas, se calculan los embeddings)...")
        reports = build_knowledge(force=args.reconstruir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(reports))
    print(f"  Carpeta de ChromaDB: {vector_store.client_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())