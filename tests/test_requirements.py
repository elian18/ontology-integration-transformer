"""requirements.txt has every library the code imports and none it no longer uses (S11-T02)."""

import ast
import re
import sys
from importlib.metadata import packages_distributions

from tests.conftest import ROOT

LOCAL = {"src", "app", "tests", "services", "views"}
REMOVED = {"anthropic", "owlready2", "prov", "sparqlwrapper", "docstring-parser"}


def _name(text: str) -> str:
    return re.sub(r"[-_.]+", "-", text).lower()


def _requirements() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {_name(re.split(r"[=<>!~\[; ]", line, maxsplit=1)[0]) for line in lines
            if line.strip() and not line.startswith("#")}


def _imports() -> set[str]:
    found = set()
    for folder in ("src", "app", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    found |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return found - LOCAL - set(sys.stdlib_module_names)


def test_every_imported_library_is_in_requirements():
    required, dists = _requirements(), packages_distributions()
    missing = {m for m in _imports()
               if not ({_name(d) for d in dists.get(m, [])} | {_name(m)}) & required}
    assert not missing, f"faltan en requirements.txt: {sorted(missing)}"


def test_the_unused_libraries_are_gone():
    assert not REMOVED & _requirements()
