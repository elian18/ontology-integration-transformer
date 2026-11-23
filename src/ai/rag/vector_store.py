from __future__ import annotations

import os
import time
from pathlib import Path

import chromadb
from chromadb.errors import InternalError, NotFoundError

from src.config import load_config

_cfg = load_config().get("vector_store", {}) or {}
# CHROMA_PATH (env) wins so the test suite can isolate ChromaDB in a temp dir.
_PATH = os.getenv("CHROMA_PATH") or _cfg.get("path") or "chroma_db"
_COLLECTION = _cfg.get("collection") or "normativa"

SPACE_COSINE = "cosine"          # distance = 1 - cosine similarity (new collections of S7)
_SPACE_KEY = "hnsw:space"

_clients: dict[str, object] = {}   # folder -> chromadb.PersistentClient

READ_RETRIES = 5                   # attempts of a read whose index is not on disk yet
READ_WAIT_SECONDS = 0.5            # wait before the 2nd attempt (doubles before each next one)
_NOT_ON_DISK = "nothing found on disk"
_sleep = time.sleep                # the tests replace it


def not_on_disk(exc: Exception) -> bool:
    """True for the error of a read that arrived before ChromaDB wrote the index to disk."""
    return isinstance(exc, InternalError) and _NOT_ON_DISK in str(exc).lower()


def _read(call):
    """Run a read, retrying only when the index of the collection is not on disk yet."""
    for attempt in range(READ_RETRIES):
        try:
            return call()
        except InternalError as exc:
            if not not_on_disk(exc) or attempt == READ_RETRIES - 1:
                raise
            _sleep(READ_WAIT_SECONDS * 2 ** attempt)


def default_name() -> str:
    """Name of the default collection (the indexed base law, "normativa")."""
    return _COLLECTION


def client_path(path: str | Path | None = None) -> str:
    """Folder of a ChromaDB database (default: the base database, chroma_db)."""
    return str(path) if path is not None else _PATH


def client(path: str | Path | None = None):
    """The ChromaDB client of a folder (one per folder, reused); default: the base database."""
    key = client_path(path)
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(path=key)
    return _clients[key]


def exists(name: str, path: str | Path | None = None) -> bool:
    """True if the collection is already stored in that folder."""
    try:
        client(path).get_collection(name)
    except NotFoundError:
        return False
    return True


def get_collection(name: str | None = None, path: str | Path | None = None, *,
                   space: str | None = None, metadata: dict | None = None):
    """Open a collection by name, creating it if it does not exist.

    ``space`` and ``metadata`` only apply when the collection is CREATED: ChromaDB keeps the
    distance function and the metadata of an existing collection as they are."""
    name = name or _COLLECTION
    db = client(path)
    try:
        return db.get_collection(name)
    except NotFoundError:
        meta = dict(metadata or {})
        if space:
            meta[_SPACE_KEY] = space
        return db.create_collection(name, metadata=meta or None)


# The default collection, kept as a module attribute for the Sprint 2 code
# (retriever.py reads ``vector_store.collection``; reset() refreshes it).
collection = get_collection()


def _target(name: str | None, path: str | Path | None):
    if path is None and (name is None or name == _COLLECTION):
        return collection
    return get_collection(name, path)


def add(ids: list[str], docs: list[str], embeddings: list[list[float]], *,
        name: str | None = None, path: str | Path | None = None):
    _target(name, path).add(ids=ids, documents=docs, embeddings=embeddings)


def upsert(ids: list[str], docs: list[str], embeddings: list[list[float]],
           metadatas: list[dict] | None = None, *, name: str | None = None,
           path: str | Path | None = None):
    """Insert or overwrite documents by id, so re-indexing is idempotent."""
    _target(name, path).upsert(ids=ids, documents=docs, embeddings=embeddings,
                               metadatas=metadatas)


def count(*, name: str | None = None, path: str | Path | None = None) -> int:
    """Number of documents stored in the collection (0 if it does not exist yet)."""
    if path is None and (name is None or name == _COLLECTION):
        return collection.count()
    if not exists(name, path):
        return 0
    return get_collection(name, path).count()


def get(ids: list[str] | None = None, *, name: str | None = None,
        path: str | Path | None = None, where: dict | None = None,
        include: tuple[str, ...] = ("documents", "metadatas")) -> dict:
    """Read stored documents (all, by id or by metadata filter)."""
    return _read(lambda: _target(name, path).get(ids=ids, where=where, include=list(include)))


def query(query_embeddings: list[list[float]], n_results: int, *, name: str | None = None,
          path: str | Path | None = None, where: dict | None = None,
          include: tuple[str, ...] = ("documents", "metadatas", "distances")) -> dict:
    """Nearest documents for each query vector (optionally filtered by metadata)."""
    return _read(lambda: _target(name, path).query(query_embeddings=query_embeddings,
                                                   n_results=n_results, where=where,
                                                   include=list(include)))


def metadata(name: str | None = None, path: str | Path | None = None) -> dict:
    """Metadata stored with the collection ({} if it does not exist)."""
    name = name or _COLLECTION
    if not exists(name, path):
        return {}
    return dict(get_collection(name, path).metadata or {})


def reset(*, name: str | None = None, path: str | Path | None = None,
          space: str | None = None, metadata: dict | None = None):
    """Drop and recreate a collection, to rebuild it from scratch. Returns the new collection.

    Without arguments it resets "normativa", exactly as in Sprint 2."""
    global collection
    name = name or _COLLECTION
    db = client(path)
    try:
        db.delete_collection(name)
    except NotFoundError:
        pass
    new = get_collection(name, path, space=space, metadata=metadata)
    if path is None and name == _COLLECTION:
        collection = new
    return new