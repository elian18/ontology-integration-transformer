"""Package the aligned graph so it opens in Protégé with its imports resolved (S5-T07, S6-T03).

The alignment module imports the profile, and the profile imports the core, by IRI. Those IRIs
are not published on the web, so Protégé needs an XML catalog (``catalog-v001.xml``) that maps
each IRI to a local file. The zip holds the three modules, that catalog, the manifest, the CSV
and a short LEEME.txt; unzipped in any folder, opening the alignment module loads the whole
chain (alignment -> profile -> core). The DPV is not imported, so it is not included.

S6-T03: the package can be built in any output format (RDF/XML, OWL, Turtle, JSON-LD). The
modules are converted with ``src.transform.convert`` (same graph, checked in S6-T02) and the
catalog points to the files of that format. The RDF/XML package keeps its Sprint 5 name and
contents. The generic packaging lives in ``src.transform.package``.

Measured in S6-T03 with Protégé 5.6.9 (OWL API 4.5.29): the RDF/XML, OWL and Turtle packages
open with their imports resolved; JSON-LD does not open ("Not a valid (absolute) IRI: @context":
Protégé reads the file as RDF/JSON, another JSON format). The JSON-LD itself is valid (an
independent processor, pyld, reads the same triples), so the JSON-LD package carries a note
in its LEEME.txt and the web shows it too (``PROTEGE_JSONLD_NOTE``).

``write_base_catalog`` replaces the ``catalog-v001.xml`` that Protégé left in data/output (it
mapped the core IRI to the profile file) with a correct one for the RDF/XML modules there.

CLI: ``py -m src.alignment.bundle [--formato rdfxml|owl|turtle|jsonld|all]`` writes the zips
into the outputs folder and fixes the catalog there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.core.emit import CORE_IRI, PROFILE_IRI, CORE_FILE, PROFILE_FILE
from src.alignment.write import ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV, alignment_iri
from src.transform.convert import OUTPUT_FORMATS, ConversionError, OutputFormat, output_format
from src.transform.package import (CATALOG_FILE, README_FILE, Module, Package, catalog_xml,
                                   package_modules, write_catalog)

BUNDLE_FILE = "ontopriv-dpv-alignment.zip"
PROTEGE_JSONLD_NOTE = ("Protégé 5.6 no abre JSON-LD (lo lee como RDF/JSON, otro formato JSON). "
                       "Para Protégé usa el paquete RDF/XML, OWL o Turtle; el JSON-LD es para "
                       "aplicaciones web y de datos enlazados.")

__all__ = ["BUNDLE_FILE", "PROTEGE_JSONLD_NOTE", "CATALOG_FILE", "README_FILE", "base_modules", "build_bundle",
           "build_package", "bundle_name", "catalog_xml", "write_base_catalog"]


def bundle_name(fmt: str | OutputFormat = "rdfxml") -> str:
    """Zip name per format; RDF/XML keeps the Sprint 5 name."""
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    if spec.key == "rdfxml":
        return BUNDLE_FILE
    return f"ontopriv-dpv-alignment-{spec.key}.zip"


def base_modules(out_dir, *, core_iri: str = CORE_IRI,
                 profile_iri: str = PROFILE_IRI) -> list[Module]:
    """The three base modules in import order (the first one is the file to open)."""
    out = Path(out_dir)
    return [Module(alignment_iri(profile_iri), out / ALIGNMENT_FILE, Path(ALIGNMENT_FILE).stem),
            Module(profile_iri, out / PROFILE_FILE, Path(PROFILE_FILE).stem),
            Module(core_iri, out / CORE_FILE, Path(CORE_FILE).stem)]


def _readme(package: Package) -> str:
    main = package.main_file or ALIGNMENT_FILE
    text = [
        "Grafo alineado con el DPV (Sprint 5)",
        f"Formato: {package.format.label}",
        "",
        f"Abre {main} en Protégé desde esta carpeta (File > Open).",
        f"El archivo {CATALOG_FILE} resuelve las importaciones (IRI -> archivo):",
        "  alineación -> perfil de Ecuador (LOPDP) -> núcleo OntoPriv-Core.",
        "El DPV no se importa: sus términos se referencian por IRI (https://w3id.org/dpv#).",
        "",
        f"{ALIGNMENT_MANIFEST}: cifras, quién decidió y huella del registro de decisiones.",
        f"{ALIGNMENT_CSV}: una fila por correspondencia aprobada.",
        "",
        "Archivos incluidos: " + ", ".join(package.included),
    ]
    if package.format.key == "jsonld":      # Protégé cannot open it (measured, S6-T03)
        text[3] = f"Archivo principal: {main}."
        text[4:4] = ["Nota: " + PROTEGE_JSONLD_NOTE, ""]
    if package.missing:
        text.append("Faltan (genéralos con py -m src.core.emit): " + ", ".join(package.missing))
    return "\n".join(text) + "\n"


def build_package(out_dir, *, fmt: str | OutputFormat = "rdfxml", core_iri: str = CORE_IRI,
                  profile_iri: str = PROFILE_IRI) -> Package:
    """The base package in one format. Raises ValueError (Spanish) without the alignment."""
    out = Path(out_dir)
    if not (out / ALIGNMENT_FILE).exists():
        raise ValueError("Todavia no hay grafo alineado: escribelo primero.")
    extras = {ALIGNMENT_MANIFEST: out / ALIGNMENT_MANIFEST, ALIGNMENT_CSV: out / ALIGNMENT_CSV}
    return package_modules(base_modules(out, core_iri=core_iri, profile_iri=profile_iri),
                           fmt, extras=extras, readme=_readme)


def build_bundle(out_dir, *, fmt: str | OutputFormat = "rdfxml", core_iri: str = CORE_IRI,
                 profile_iri: str = PROFILE_IRI) -> tuple[bytes, list[str], list[str]]:
    """The zip as bytes, the files it includes and the module files that were missing
    (Sprint 5 signature; ``fmt`` added in S6-T03)."""
    package = build_package(out_dir, fmt=fmt, core_iri=core_iri, profile_iri=profile_iri)
    return package.data, package.included, package.missing


def write_base_catalog(out_dir, *, core_iri: str = CORE_IRI,
                       profile_iri: str = PROFILE_IRI) -> Path:
    """Correct catalog-v001.xml in the outputs folder for the RDF/XML modules there."""
    return write_catalog(out_dir, base_modules(out_dir, core_iri=core_iri,
                                               profile_iri=profile_iri), "rdfxml")


def _run(argv: list[str] | None = None) -> int:
    from src.config import ROOT, load_config

    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.bundle",
        description="Escribe el paquete para Protégé en uno o todos los formatos (S6-T03).")
    parser.add_argument("--formato", default="all",
                        help="rdfxml, owl, turtle, jsonld o all (por defecto, todos)")
    args = parser.parse_args(argv)

    cfg = load_config()
    out = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    out = out if out.is_absolute() else ROOT / out
    try:
        formats = (list(OUTPUT_FORMATS.values()) if args.formato.strip().lower() == "all"
                   else [output_format(args.formato)])
        lines = ["Paquetes para Protégé (S6-T03)"]
        for spec in formats:
            package = build_package(out, fmt=spec)
            name = bundle_name(spec)
            (out / name).write_bytes(package.data)
            modules = [f for f in package.included if f.endswith(spec.extension)]
            lines.append(f"  {name:<40} {spec.label:<14} abre {package.main_file}  "
                         f"({len(modules)} módulos)")
            if package.missing:
                lines.append(f"    faltan: {', '.join(package.missing)}")
    except (ValueError, ConversionError) as exc:
        print(f"Error: {exc}")
        return 1
    catalog = write_base_catalog(out)
    lines.append(f"  Catálogo corregido: {catalog.name} (IRI -> archivo .rdf en esa carpeta)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())