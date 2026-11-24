"""The files of the edge cases (S11-T10): empty, damaged, a scanned law, too big, a wrong
extension and a small privacy ontology for the cases without key, without Java or without quota.
They are made here, so nobody keeps them in git (one weighs more than the upload limit)."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from src.workspace.store import workspace_settings

FOLDER = "data/workspaces/_e2e/_material"
MINI = """@prefix : <http://example.org/mini-privacidad#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

<http://example.org/mini-privacidad> a owl:Ontology ;
    rdfs:label "Mini ontología de privacidad (casos límite S11-T10)"@es .
"""
CLASSES = (
    ("Titular", "titular de los datos", "Persona natural cuyos datos personales se tratan."),
    ("Consentimiento", "consentimiento",
     "Manifestación de voluntad libre, específica, informada e inequívoca del titular."),
    ("DatoSensible", "dato sensible",
     "Dato cuyo tratamiento indebido puede dar origen a discriminación."),
    ("RegistroDeActividades", "registro de actividades de tratamiento",
     "Documento donde el responsable anota los tratamientos que realiza y sus finalidades."),
    ("BrechaDeSeguridad", "brecha de seguridad",
     "Incidente que provoca la destrucción, pérdida o acceso no autorizado a datos personales."),
    ("Portabilidad", "portabilidad de los datos",
     "Derecho del titular a recibir sus datos en un formato estructurado y llevarlos a otro "
     "responsable."))
SCANNED_LINES = ("LEY DE PROTECCIÓN DE DATOS (copia escaneada)", "",
                 "Art. 1.- Objeto.- Esta ley protege los datos personales.",
                 "Art. 2.- Consentimiento.- El tratamiento requiere el consentimiento",
                 "del titular, libre e informado.")


def mini() -> str:
    lines = [MINI]
    for name, label, definition in CLASSES:
        lines.append(f':{name} a owl:Class ; rdfs:label "{label}"@es ;\n'
                     f'    skos:definition "{definition}"@es .\n')
    return "\n".join(lines)


def scanned_pdf(path: Path) -> None:
    """A law as a picture: the text is in the pixels, there is no text to read."""
    page = Image.new("L", (1240, 1754), color=255)
    draw = ImageDraw.Draw(page)
    for i, line in enumerate(SCANNED_LINES):
        draw.text((100, 150 + 40 * i), line, fill=0)
    page.save(path, "PDF", resolution=150)


def big(path: Path, megabytes: float) -> None:
    """Valid Turtle, one byte over ``megabytes``: only comments after a small ontology."""
    filler = "# relleno para pasar el límite de tamaño de una subida\n"
    head = mini()
    count = int(megabytes * 1024 * 1024 - len(head.encode("utf-8"))) // len(filler.encode()) + 1
    path.write_text(head + filler * count, encoding="utf-8", newline="\n")


def write_material(folder: Path, max_mb: float) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    files = {"vacia.ttl": b"",
             "danada.ttl": mini().encode("utf-8")[:-60],          # cut in the middle
             "mini-privacidad.ttl": mini().encode("utf-8"),
             "mi-ontologia.docx": mini().encode("utf-8")}         # Turtle, wrong extension
    for name, data in files.items():
        (folder / name).write_bytes(data)
    scanned_pdf(folder / "escaneada.pdf")
    big(folder / "grande.ttl", max_mb)
    return sorted(folder.iterdir())


def _run(argv: list[str] | None = None) -> int:
    from src.config import ROOT
    parser = argparse.ArgumentParser(
        prog="py -m src.e2e.edge",
        description="Genera los archivos de los casos límite del Sprint 11.")
    parser.add_argument("--carpeta", type=Path, default=ROOT / FOLDER)
    args = parser.parse_args(argv)
    max_mb = workspace_settings()["max_upload_mb"]
    print("Archivos de los casos límite (S11-T10)")
    for path in write_material(args.carpeta, max_mb):
        print(f"  {path.name}: {path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  Carpeta: {args.carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
