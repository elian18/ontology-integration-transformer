"""Bridge for the modular-core view (Sprint 3): structure of the core and the profile.

Wraps the split (S3-T03) so the Streamlit view stays pure presentation, mirroring how
services.segmentation backs the articles view. The counts and families come straight from
split_summary; the result is cached because it is deterministic and touches no network."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.config import load_config
from src.ingest.ontology_loader import load_ontology
from src.core.split import assign_modules, split_summary


def _base_ontology_path() -> Path:
    path = Path(load_config().get("inputs", {}).get("ontology", "data/input/ontopriv.rdf"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path      # app/services/ -> project root
    return path


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
        "counts": summary["counts"],                    # {"core": {...}, "profile": {...}}
        "core_families": summary["core_families"],      # {family: n_classes}
        "profile_families": summary["profile_families"],
        "n_cross_refs": summary["n_cross_refs"],
        "n_flagged": summary["n_flagged"],
    }