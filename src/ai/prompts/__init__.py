"""Versioned prompts of the AI layer (S7-T02).

Until Sprint 6 every prompt lived as a constant inside the module that used it
(``src/alignment/justify.py``, ``src/core/extract.py``) and this folder was empty since
Sprint 0. Now each prompt is a plain text file here, named ``<name>.v<version>.txt``
(e.g. ``justify.v2.txt``), and the code loads it by name and version:

    from src.ai.prompts import load_prompt
    prompt = load_prompt("justify", 2)      # prompt.text, prompt.version, prompt.sha256

Why files: a prompt is part of the method (it changes the results), so it is versioned in git
like the code, readable without opening Python, and every run records WHICH prompt produced
it: its name, version and sha256 travel to the manifests and, in Sprint 10, to PROV-O.

Rules: a published version is never edited; a change is a new file with the next version.
The text is read as UTF-8 (Python already turns Windows line endings into LF) and ONE final
newline is removed (editors add or strip it), so the fingerprint is the same on every machine.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
_FILE_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)\.v(?P<version>\d+)\.txt$")


class PromptNotFound(FileNotFoundError):
    """The requested prompt (name or version) does not exist in src/ai/prompts/."""


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str
    sha256: str
    path: Path

    def manifest(self) -> dict:
        """What a run records about the prompt it used."""
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


def _normalize(text: str) -> str:
    """Drop ONE final newline (``read_text`` already turned Windows line endings into LF)."""
    return text[:-1] if text.endswith("\n") else text


def available(directory: Path | None = None) -> dict[str, list[int]]:
    """Every prompt in the folder with its versions, sorted: {'justify': [2], ...}."""
    out: dict[str, list[int]] = {}
    for f in sorted((directory or PROMPTS_DIR).glob("*.txt")):
        m = _FILE_RE.match(f.name)
        if m:
            out.setdefault(m["name"], []).append(int(m["version"]))
    return {name: sorted(versions) for name, versions in out.items()}


def load_prompt(name: str, version: int | None = None, directory: Path | None = None) -> Prompt:
    """Load a prompt by name; without ``version`` the latest one is returned."""
    directory = directory or PROMPTS_DIR
    versions = available(directory).get(name)
    if not versions:
        raise PromptNotFound(f"No existe el prompt '{name}' en {directory}.")
    version = versions[-1] if version is None else int(version)
    if version not in versions:
        raise PromptNotFound(f"El prompt '{name}' no tiene la versión {version} "
                             f"(hay: {', '.join(map(str, versions))}).")
    path = directory / f"{name}.v{version}.txt"
    text = _normalize(path.read_text(encoding="utf-8"))
    if not text.strip():
        raise ValueError(f"El prompt {path.name} está vacío.")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Prompt(name=name, version=version, text=text, sha256=digest, path=path)


def prompts_manifest(*names_and_versions: tuple[str, int]) -> dict[str, dict]:
    """Manifest block for several prompts: {'justify': {'name', 'version', 'sha256'}, ...}."""
    return {name: load_prompt(name, version).manifest() for name, version in names_and_versions}