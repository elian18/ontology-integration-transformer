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



def _project_config() -> dict:
    import yaml
    root = Path(__file__).resolve().parents[2]
    cfg = root / "config" / "config.yaml"
    if cfg.exists():
        return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    return {}


def base_ontology():
    """Load the project's base ontology (OntoPriv) from the path in config.yaml."""
    inputs = _project_config().get("inputs", {})
    path = Path(inputs.get("ontology", "data/input/ontopriv.rdf"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return load_ontology(path) if path.exists() else None


def base_legal_text():
    """Load the project's base legal text (LOPDP) from the path in config.yaml."""
    inputs = _project_config().get("inputs", {})
    path = Path(inputs.get("legal_text", "data/input/lopdp.pdf"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return load_legal_text(path) if path.exists() else None