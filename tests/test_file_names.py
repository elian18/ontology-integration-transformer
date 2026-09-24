"""S4-T01: every project output file is named in English (no Spanish file names left)."""
from pathlib import Path

from src.core.extract import PROPOSALS_FILE
from src.core.emit import MANIFEST_FILE, CORE_FILE, PROFILE_FILE
from app.services import core_module

ROOT = Path(__file__).resolve().parents[1]
# Old Spanish names (Sprint 3). Kept only here, so the scan below can detect them.
OLD_NAMES = ("perfil-conceptos-" + "propuestos", "nucleo-" + "manifest")


def test_output_constants_are_english():
    assert PROPOSALS_FILE == "profile-proposed-concepts.json"
    assert MANIFEST_FILE == "core-manifest.json"
    assert CORE_FILE == "ontopriv-core.rdf"
    assert PROFILE_FILE == "profile-ecuador-lopdp.rdf"


def test_web_service_reads_the_same_file_the_extractor_writes():
    assert core_module._PROPOSALS_FILE == PROPOSALS_FILE
    assert core_module._proposals_path().name == PROPOSALS_FILE


def test_service_reads_proposals_from_renamed_file(tmp_path):
    f = tmp_path / PROPOSALS_FILE
    f.write_text('{"source": "lopdp", "concepts": [{"name": "DataSubject"}]}', encoding="utf-8")
    data = core_module.proposed_concepts(path=f)
    assert data is not None
    assert data["concepts"][0]["name"] == "DataSubject"


def test_no_old_spanish_file_names_in_code_or_config():
    targets = list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").rglob("*.py"))
    targets += [p for p in (ROOT / "config").glob("*.yaml")]
    offenders = []
    for path in targets:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for old in OLD_NAMES:
            if old in text:
                offenders.append(f"{path.relative_to(ROOT)}: {old}")
    assert not offenders, "Nombres viejos en español: " + ", ".join(offenders)