from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.transform.convert import ConversionError, input_format

MANIFEST_FILE = "workspace.json"
INPUT_DIR, WORK_DIR, OUTPUT_DIR = "input", "work", "output"
SOURCE_PREFIX = "upload:"
DEFAULT_ROOT = "data/workspaces"
DEFAULT_MAX_MB = 25

KIND_ONTOLOGY = "ontology"
KIND_LAW = "law"
LAW_SUFFIXES = (".txt", ".pdf")

STEPS = ("inherit", "candidates", "propose", "labels", "write", "package")
LAW_STEPS = ("segment", "index")
EXTRACT_STEPS = ("extract", "accept")
PROFILE_STEPS = ("profile",)                    # "solo ley": then the steps of STEPS
DEFINITION_STEPS = ("definitions",)             # "ley + ontología": right after inheriting
EVIDENCE_STEPS = ("evidence",)                  # "ley + ontología": before writing
ALL_STEPS = (STEPS + LAW_STEPS + EXTRACT_STEPS + PROFILE_STEPS + DEFINITION_STEPS
             + EVIDENCE_STEPS)
STEP_LABELS = {"inherit": "Heredar lo validado", "candidates": "Buscar candidatos del DPV",
               "propose": "Proponer correspondencias",
               "labels": "Completar las etiquetas en español e inglés",
               "write": "Escribir la ontología alineada",
               "package": "Preparar la descarga", "segment": "Separar la ley en artículos",
               "index": "Indexar los artículos de la ley",
               "extract": "Extraer los conceptos de la ley con la IA",
               "accept": "Aceptar los conceptos por regla",
               "profile": "Armar el perfil de la ley sobre OntoPriv-Core",
               "definitions": "Completar las definiciones con la ley",
               "evidence": "Buscar en la ley los artículos de cada concepto"}
PENDING, RUNNING, DONE, ERROR = "pending", "running", "done", "error"
STATUS_LABELS = {PENDING: "pendiente", RUNNING: "en curso", DONE: "hecho", ERROR: "error"}
ACCEPTED_ONTOLOGY = "RDF/XML (.rdf, .owl), Turtle (.ttl) o JSON-LD (.jsonld)"

# Folders of the system base: a workspace root may never be (or be inside) one of them.
BASE_DIRS = ("data/output", "data/review", "data/input", "data/reference", "vocab")


class WorkspaceError(ValueError):
    """The upload cannot be stored (message in Spanish, shown as is in the web)."""


def steps_for(mode: str | None) -> tuple[str, ...]:
    """The steps of a workspace, in order, by what it holds."""
    if mode == "law":
        return LAW_STEPS + EXTRACT_STEPS + PROFILE_STEPS + STEPS
    if mode == "ontology+law":
        return (LAW_STEPS + STEPS[:1] + DEFINITION_STEPS + STEPS[1:4] + EVIDENCE_STEPS
                + STEPS[4:])
    return STEPS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --------------------------------------------------------------------------- names

def slugify(name: str, max_len: int = 40) -> str:
    """Readable, ASCII, lowercase name for a folder: 'Ontología GDPR v2' -> 'ontologia-gdpr-v2'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:max_len].strip("-") or "carga"


def safe_file_name(name: str) -> str:
    """The uploaded name without folders or odd characters, keeping its extension."""
    path = Path(str(name).replace("\\", "/")).name
    stem, suffix = Path(path).stem, Path(path).suffix.lower()
    return f"{slugify(stem, 60)}{suffix}"


def content_hash(files: dict[str, tuple[str, bytes]]) -> str:
    """SHA-256 over every uploaded file (kind, then bytes), independent of the file names."""
    digest = hashlib.sha256()
    for kind in sorted(files):
        digest.update(kind.encode("utf-8") + b"\0")
        digest.update(files[kind][1])
    return digest.hexdigest()


def workspace_id(files: dict[str, tuple[str, bytes]]) -> str:
    """'<slug of the first file name>-<8 hex of the content hash>'."""
    first = files.get(KIND_ONTOLOGY) or next(iter(files.values()))
    return f"{slugify(Path(first[0]).stem)}-{content_hash(files)[:8]}"


# --------------------------------------------------------------------------- settings

def workspace_settings(cfg: dict | None = None) -> dict:
    from src.config import ROOT, load_config
    cfg = cfg if cfg is not None else load_config()
    ws = cfg.get("workspaces", {}) or {}
    root = Path(ws.get("dir", DEFAULT_ROOT))
    root = root if root.is_absolute() else ROOT / root
    return {"root": root, "max_upload_mb": float(ws.get("max_upload_mb", DEFAULT_MAX_MB))}


def check_root(root: str | Path, project_root: Path | None = None) -> Path:
    """Refuse a workspace root that is, or is inside, a folder of the system base."""
    from src.config import ROOT
    project_root = project_root or ROOT
    root = Path(root).resolve()
    for base in BASE_DIRS:
        base_path = (project_root / base).resolve()
        if root == base_path or base_path in root.parents:
            raise WorkspaceError(
                f"La carpeta de espacios de trabajo no puede estar dentro de {base}: "
                "ahí están los archivos de la base del sistema.")
    return root


# --------------------------------------------------------------------------- validation

def _check_upload(kind: str, name: str, data: bytes, max_mb: float) -> str:
    """Validate one uploaded file; returns its format label."""
    if kind not in (KIND_ONTOLOGY, KIND_LAW):
        raise WorkspaceError(f"Tipo de archivo desconocido: {kind}")
    if not data:
        raise WorkspaceError(f"El archivo {name} está vacío.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise WorkspaceError(f"El archivo {name} pesa {size_mb:.1f} MB; el máximo es "
                             f"{max_mb:g} MB.")
    suffix = Path(name).suffix.lower()
    if kind == KIND_LAW:
        if suffix not in LAW_SUFFIXES:
            raise WorkspaceError(f"El texto de ley debe ser .txt o .pdf (llegó '{suffix}').")
        return suffix.lstrip(".").upper()
    try:
        fmt = input_format(name)
    except ConversionError as exc:
        hint = "" if suffix == ".owx" else f" Sube la ontología en {ACCEPTED_ONTOLOGY}."
        raise WorkspaceError(f"{exc}{hint}") from exc
    return {"xml": "RDF/XML", "turtle": "Turtle", "json-ld": "JSON-LD"}[fmt]


# --------------------------------------------------------------------------- workspace

@dataclass
class Workspace:
    path: Path
    manifest: dict = field(default_factory=dict)
    reused: bool = False            # True when the same upload already had a workspace

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def source_id(self) -> str:
        return self.manifest["source_id"]

    @property
    def input_dir(self) -> Path:
        return self.path / INPUT_DIR

    @property
    def work_dir(self) -> Path:
        return self.path / WORK_DIR

    @property
    def output_dir(self) -> Path:
        return self.path / OUTPUT_DIR

    def input_path(self, kind: str = KIND_ONTOLOGY) -> Path | None:
        entry = self.manifest.get("inputs", {}).get(kind)
        return self.input_dir / entry["stored_as"] if entry else None

    def step(self, name: str) -> dict:
        return self.manifest.setdefault("steps", {}).get(name, {"status": PENDING})

    def mark_step(self, name: str, status: str, detail: str | None = None,
                  **extra) -> None:
        """Record the state of a step (pending/running/done/error) and save the manifest."""
        if name not in ALL_STEPS:
            raise WorkspaceError(f"Paso desconocido: {name}")
        if status not in (PENDING, RUNNING, DONE, ERROR):
            raise WorkspaceError(f"Estado desconocido: {status}")
        entry = {"status": status, "at": _now()}
        if detail:
            entry["detail"] = detail
        entry.update(extra)
        self.manifest.setdefault("steps", {})[name] = entry
        self.save()

    def progress(self) -> list[dict]:
        """One row per step, in order, for the web and the console."""
        return [{"step": s, "label": STEP_LABELS[s], **self.step(s)}
                for s in steps_for(self.manifest.get("mode"))]

    def save(self) -> Path:
        self.manifest["updated"] = _now()
        path = self.path / MANIFEST_FILE
        path.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path


def _characterize(path: Path) -> dict:
    """Load the stored ontology once to reject unreadable files early and keep its profile."""
    from src.ingest.ontology_loader import characterization_summary, load_ontology
    try:
        summary = characterization_summary(load_ontology(path))
    except Exception as exc:
        raise WorkspaceError(f"No se pudo leer la ontología {path.name}: {exc}") from exc
    summary.pop("path", None)          # absolute path of this machine: not useful in the json
    return summary


def open_upload(files: dict[str, tuple[str, bytes]], *, cfg: dict | None = None,
                root: str | Path | None = None) -> Workspace:
    """Create the workspace of an upload, or reopen it if the same files were uploaded before.

    ``files`` maps a kind (``"ontology"``, ``"law"``) to ``(file name, bytes)``. Raises
    WorkspaceError (Spanish) for empty, too large, unsupported or unreadable files."""
    if not files:
        raise WorkspaceError("No se subió ningún archivo.")
    settings = workspace_settings(cfg)
    base_root = check_root(root if root is not None else settings["root"])
    labels = {kind: _check_upload(kind, name, data, settings["max_upload_mb"])
              for kind, (name, data) in files.items()}

    ws_id = workspace_id(files)
    path = base_root / ws_id
    manifest_path = path / MANIFEST_FILE
    if manifest_path.exists():
        ws = Workspace(path, json.loads(manifest_path.read_text(encoding="utf-8")), reused=True)
        ws.manifest["opened"] = ws.manifest.get("opened", 1) + 1
        ws.save()
        return ws

    for sub in (INPUT_DIR, WORK_DIR, OUTPUT_DIR):
        (path / sub).mkdir(parents=True, exist_ok=True)
    inputs = {}
    for kind, (name, data) in files.items():
        stored = safe_file_name(name)
        target = path / INPUT_DIR / stored
        target.write_bytes(data)
        entry = {"original_name": Path(str(name).replace("\\", "/")).name,
                 "stored_as": stored, "format": labels[kind], "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
        if kind == KIND_ONTOLOGY:
            try:
                entry["characterization"] = _characterize(target)
            except WorkspaceError:
                shutil.rmtree(path, ignore_errors=True)   # new folder: nothing else lives here
                raise
        inputs[kind] = entry
    mode = ("ontology+law" if len(inputs) == 2
            else "ontology" if KIND_ONTOLOGY in inputs else "law")
    ws = Workspace(path, {"id": ws_id, "source_id": SOURCE_PREFIX + ws_id, "mode": mode,
                          "created": _now(), "opened": 1, "inputs": inputs,
                          "steps": {s: {"status": PENDING} for s in steps_for(mode)}})
    ws.save()
    return ws


def open_ontology(name: str, data: bytes, **kwargs) -> Workspace:
    """Shortcut for the "solo ontología" mode (Sprint 6)."""
    return open_upload({KIND_ONTOLOGY: (name, data)}, **kwargs)


def open_law(name: str, data: bytes, **kwargs) -> Workspace:
    """Shortcut for the "solo ley" mode (Sprint 7)."""
    return open_upload({KIND_LAW: (name, data)}, **kwargs)


def load_workspace(ws_id: str, *, cfg: dict | None = None,
                   root: str | Path | None = None) -> Workspace:
    base_root = Path(root) if root is not None else workspace_settings(cfg)["root"]
    path = base_root / ws_id
    if not (path / MANIFEST_FILE).exists():
        raise WorkspaceError(f"No existe el espacio de trabajo {ws_id}.")
    return Workspace(path, json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8")),
                     reused=True)


def list_workspaces(*, cfg: dict | None = None, root: str | Path | None = None) -> list[dict]:
    """Every workspace (most recent first): id, original names, mode, dates."""
    base_root = Path(root) if root is not None else workspace_settings(cfg)["root"]
    rows = []
    for manifest in base_root.glob(f"*/{MANIFEST_FILE}"):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rows.append({"id": data["id"], "mode": data.get("mode"),
                     "files": [i["original_name"] for i in data.get("inputs", {}).values()],
                     "created": data.get("created"), "updated": data.get("updated")})
    return sorted(rows, key=lambda r: r.get("updated") or "", reverse=True)


# --------------------------------------------------------------------------- console

def render_console(ws: Workspace) -> str:
    lines = ["Espacio de trabajo (S6-T04)",
             f"  id: {ws.id}  ({'reabierto: ya existía' if ws.reused else 'nuevo'})",
             f"  carpeta: {ws.path}",
             f"  source_id: {ws.source_id}"]
    for kind, entry in ws.manifest.get("inputs", {}).items():
        label = "Ontología" if kind == KIND_ONTOLOGY else "Ley"
        size = f"{entry['bytes']:,}".replace(",", ".")
        lines.append(f"  {label}: {entry['original_name']} ({entry['format']}, {size} bytes)")
        c = entry.get("characterization")
        if c:
            s = c["structure"]
            lines.append(f"    {s['triples']} tripletas · {s['classes']} clases · "
                         f"{s['object_properties']} propiedades de objeto · "
                         f"{s['data_properties']} de datos · {s['individuals']} individuos")
            lines.append(f"    {c['flavor_detail']}")
    lines.append("  Pasos:")
    lines.extend(f"    {r['label']}: {STATUS_LABELS[r['status']]}" for r in ws.progress())
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m src.workspace.store",
        description="Crea (o reabre) el espacio de trabajo de una ontología subida (S6-T04).")
    parser.add_argument("file", help="ontología .rdf, .owl, .ttl o .jsonld")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        ws = open_ontology(path.name, path.read_bytes())
    except WorkspaceError as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(ws))
    return 0


__all__ = ["Workspace", "WorkspaceError", "open_upload", "open_ontology", "open_law",
           "load_workspace", "list_workspaces", "workspace_id", "slugify", "STEPS", "LAW_STEPS",
           "EXTRACT_STEPS", "PROFILE_STEPS", "DEFINITION_STEPS", "EVIDENCE_STEPS", "steps_for"]

if __name__ == "__main__":
    raise SystemExit(_run())