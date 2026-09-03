"""Adapter between the UI (uploaded bytes) and the src.ingest loaders.

Streamlit gives us uploaded files as bytes; the loaders expect a path. We persist the
upload to a temp file and call the same loaders used everywhere else, so there is a single
source of truth and no duplicated logic."""
from __future__ import annotations
from pathlib import Path
import tempfile

from src.ingest.ontology_loader import load_ontology, OntologyReport, characterization_summary
from src.ingest.text_loader import load_legal_text, TextReport
from src.ingest.dpv_loader import load_dpv, DpvReport


def _persist(name: str, data: bytes) -> Path:
    suffix = Path(name).suffix or ".dat"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def ingest_ontology(name: str, data: bytes) -> OntologyReport:
    """Load + characterize an uploaded ontology (RDF/XML, Turtle, JSON-LD)."""
    return load_ontology(_persist(name, data))


def ontology_summary(name: str, data: bytes) -> dict:
    """Same as ingest_ontology but returns the serializable summary (for the UI/JSON)."""
    return characterization_summary(load_ontology(_persist(name, data)))


def ingest_legal_text(name: str, data: bytes) -> TextReport:
    """Load an uploaded legal text (.txt or .pdf)."""
    return load_legal_text(_persist(name, data))


def dpv_status(path: str | Path = "vocab/dpv.ttl") -> DpvReport | None:
    """Load the DPV from disk if present (it is a fixed reference, not uploaded)."""
    p = Path(path)
    return load_dpv(p) if p.exists() else None