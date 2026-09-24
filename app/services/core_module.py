"""Bridge for the modular-core view (Sprint 3): structure, downloads and AI proposals."""
from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

from src.config import load_config
from src.ingest.ontology_loader import load_ontology
from src.core.split import assign_modules, split_summary
from src.core.emit import materialize_modules
from src.core.extract import PROPOSALS_FILE

# Same name the extractor writes; imported so both sides can never drift apart.
_PROPOSALS_FILE = PROPOSALS_FILE


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_ontology_path() -> Path:
    path = Path(load_config().get("inputs", {}).get("ontology", "data/input/ontopriv.rdf"))
    if not path.is_absolute():
        path = _root() / path
    return path


def _proposals_path() -> Path:
    out_dir = load_config().get("outputs", {}).get("dir", "data/output")
    p = Path(out_dir)
    if not p.is_absolute():
        p = _root() / p
    return p / _PROPOSALS_FILE


@lru_cache(maxsize=1)
def core_structure() -> dict | None:
    """Structure of the modular core: class/property counts and families per module.

    Returns None when the base ontology is missing. The view only reads this dict."""
    path = _base_ontology_path()
    if not path.exists():
        return None
    summary = split_summary(assign_modules(load_ontology(str(path))))
    return {
        "onto_path": summary["onto_path"],
        "counts": summary["counts"],
        "core_families": summary["core_families"],
        "profile_families": summary["profile_families"],
        "n_cross_refs": summary["n_cross_refs"],
        "n_flagged": summary["n_flagged"],
    }


@lru_cache(maxsize=1)
def build_downloads() -> dict | None:
    """Materialize the core and profile and return their file bytes for st.download_button.

    Returns None when the base ontology is missing. These are the REAL emitted files (with
    individuals and the autonomy move), so their counts differ slightly from core_structure()."""
    path = _base_ontology_path()
    if not path.exists():
        return None
    report = load_ontology(str(path))
    with tempfile.TemporaryDirectory() as tmp:
        result = materialize_modules(report, out_dir=tmp)
        core_bytes = Path(result.core_path).read_bytes()
        profile_bytes = Path(result.profile_path).read_bytes()
        manifest_bytes = Path(result.manifest_path).read_bytes()
    return {
        "core": {"filename": Path(result.core_path).name, "bytes": core_bytes,
                 "entities": result.counts["core"]["seeds"], "triples": result.counts["core"]["triples"]},
        "profile": {"filename": Path(result.profile_path).name, "bytes": profile_bytes,
                    "entities": result.counts["profile"]["seeds"], "triples": result.counts["profile"]["triples"]},
        "manifest": {"filename": Path(result.manifest_path).name, "bytes": manifest_bytes},
        "moved_to_profile": result.moved_to_profile,
        "core_iri": result.core_iri, "profile_iri": result.profile_iri,
    }


def proposed_concepts(path=None) -> dict | None:
    """Read the AI-proposed concepts written by S3-T05 (``PROPOSALS_FILE`` in data/output).

    Returns None when the extraction has not been run yet. Not cached: the file grows as the
    extraction resumes, and the view should always show the latest."""
    p = Path(path) if path is not None else _proposals_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "source": data.get("source", ""),
        "counts": data.get("counts", {}),
        "quota_exhausted": data.get("quota_exhausted", False),
        "processed_articles": data.get("processed_articles", []),
        "concepts": data.get("concepts", []),
    }