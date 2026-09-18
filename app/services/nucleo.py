"""Bridge for the modular-core view (Sprint 3): structure and downloadable modules.

Wraps the split (S3-T03) and the materialization (S3-T04) so the Streamlit view stays pure
presentation, mirroring how services.segmentation backs the articles view. Results are cached
because they are deterministic and touch no network."""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from src.config import load_config
from src.ingest.ontology_loader import load_ontology
from src.core.split import assign_modules, split_summary
from src.core.emit import materialize_modules


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


@lru_cache(maxsize=1)
def build_downloads() -> dict | None:
    """Materialize the core and profile and return their file bytes for st.download_button.

    Returns None when the base ontology is missing. Bytes are read into memory, so the temp
    files can be cleaned up immediately. These are the REAL emitted files (with individuals and
    the autonomy move), so their counts differ slightly from core_structure()."""
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