"""Package the aligned graph so it opens in Protégé with its imports resolved (S5-T07).

The alignment module imports the profile, and the profile imports the core, by IRI. Those IRIs
are not published on the web, so Protégé needs an XML catalog (``catalog-v001.xml``) that maps
each IRI to a local file. The zip holds the three modules, that catalog, the manifest, the CSV
and a short LEEME.txt; unzipped in any folder, opening ``ontopriv-dpv-alignment.rdf`` loads the
whole chain (alignment -> profile -> core). The DPV is not imported, so it is not included.

The catalog is generated here and only goes inside the zip: the ``catalog-v001.xml`` that
Protégé may have left in data/output is not touched.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

from src.core.emit import CORE_IRI, PROFILE_IRI, CORE_FILE, PROFILE_FILE
from src.alignment.write import ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV, alignment_iri

BUNDLE_FILE = "ontopriv-dpv-alignment.zip"
CATALOG_FILE = "catalog-v001.xml"
README_FILE = "LEEME.txt"


def catalog_xml(entries: dict[str, str]) -> str:
    """Protégé XML catalog: ontology IRI -> local file name."""
    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
             '<catalog prefer="public" xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">']
    for i, (iri, file_name) in enumerate(entries.items(), start=1):
        lines.append(f'    <uri id={quoteattr(f"modulo-{i}")} name={quoteattr(iri)} '
                     f'uri={quoteattr(file_name)}/>')
    lines.append("</catalog>")
    return "\n".join(lines) + "\n"


def _readme(included: list[str], missing: list[str]) -> str:
    text = [
        "Grafo alineado con el DPV (Sprint 5)",
        "",
        f"Abre {ALIGNMENT_FILE} en Protégé desde esta carpeta (File > Open).",
        f"El archivo {CATALOG_FILE} resuelve las importaciones:",
        "  alineación -> perfil de Ecuador (LOPDP) -> núcleo OntoPriv-Core.",
        "El DPV no se importa: sus términos se referencian por IRI (https://w3id.org/dpv#).",
        "",
        f"{ALIGNMENT_MANIFEST}: cifras, quién decidió y huella del registro de decisiones.",
        f"{ALIGNMENT_CSV}: una fila por correspondencia aprobada.",
        "",
        "Archivos incluidos: " + ", ".join(included),
    ]
    if missing:
        text.append("Faltan (genéralos con py -m src.core.emit): " + ", ".join(missing))
    return "\n".join(text) + "\n"


def build_bundle(out_dir, *, core_iri: str = CORE_IRI,
                 profile_iri: str = PROFILE_IRI) -> tuple[bytes, list[str], list[str]]:
    """The zip as bytes, the files it includes and the module files that were missing.

    Raises ValueError (in Spanish) when the alignment module itself has not been written."""
    out = Path(out_dir)
    if not (out / ALIGNMENT_FILE).exists():
        raise ValueError("Todavia no hay grafo alineado: escribelo primero.")
    modules = {alignment_iri(profile_iri): ALIGNMENT_FILE, profile_iri: PROFILE_FILE,
               core_iri: CORE_FILE}
    extras = [ALIGNMENT_MANIFEST, ALIGNMENT_CSV]
    included, missing = [], []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in list(modules.values()) + extras:
            path = out / name
            if path.exists():
                zf.write(path, arcname=name)
                included.append(name)
            elif name in (PROFILE_FILE, CORE_FILE):
                missing.append(name)
        present = {iri: f for iri, f in modules.items() if f in included}
        zf.writestr(CATALOG_FILE, catalog_xml(present))
        included.append(CATALOG_FILE)
        zf.writestr(README_FILE, _readme(included + [README_FILE], missing))
        included.append(README_FILE)
    return buf.getvalue(), included, missing