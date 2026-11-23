"""The repository carries the base, its license and nothing of the developer's machine (S11-T01)."""

import json
import os
import shutil
import subprocess
import sys

import pytest
from rdflib import Graph

from src.alignment.inherit import base_paths
from src.core.law_profile import core_path

from tests.conftest import ROOT

MODULES = ("ontopriv-core", "profile-ecuador-lopdp", "ontopriv-dpv-alignment",
           "ontopriv-gdpr-alignment", "ontopriv-provenance")
FORMATS = ("rdf", "owl", "ttl", "jsonld")
BASE_FILES = [f"data/output/{m}.{f}" for m in MODULES for f in FORMATS] + [
    "data/output/catalog-v001.xml", "data/output/alignment-candidates.json"]
# What the "Herramientas de construcción" pages show (S11-T02: a clean clone showed them empty).
TOOLS_FILES = [f"data/output/{f}" for f in (
    "profile-proposed-concepts.json", "alignment-approved.csv", "alignment-evaluation.json",
    "alignment-manifest.json", "ontopriv-gdpr-bridge.json", "ontopriv-gdpr-bridge.csv",
    "questions-report.json")]
NOT_IN_GIT = ["data/output/ontopriv-dpv-alignment.zip", "data/output/shacl-report.json",
              "data/output/decider-evaluation.json", "data/output/bridge-cache.jsonl",
              "data/output/e2e/13-suite.txt", "data/output/decider/cache.jsonl",
              "data/workspaces/gdprtext-ontology-d35361ae/workspace.json",
              "chroma_db/chroma.sqlite3", "tools/apache-jena-6.2.0.zip", ".env",
              ".agents/skills/developing-with-streamlit",
              ".claude/skills/developing-with-streamlit"]


def _git(*args) -> subprocess.CompletedProcess:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("sin git o fuera del repositorio")
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _ignored(path: str) -> bool:
    """What .gitignore says about the path, whether or not it is already in git."""
    return _git("check-ignore", "--no-index", "-q", path).returncode == 0


@pytest.mark.parametrize("path", BASE_FILES + TOOLS_FILES)
def test_the_base_goes_in_git(path):
    assert not _ignored(path), f"{path} quedaría fuera del repositorio"


@pytest.mark.parametrize("path", NOT_IN_GIT)
def test_measurements_caches_and_secrets_stay_out(path):
    assert _ignored(path), f"{path} se subiría al repositorio"


def test_what_the_assistant_reads_from_the_base_is_in_git():
    """Inheritance reads the candidates file and the law mode reads the core."""
    tracked = {(ROOT / p).resolve() for p in BASE_FILES}
    assert base_paths()["candidates"].resolve() in tracked
    assert core_path().resolve() in tracked


# The pages run in another process: an audit hook cannot be removed and slows every later test.
_PAGE_READS = """
import json, os, sys
root, page = sys.argv[1], sys.argv[2]
sys.path[:0] = [root, os.path.join(root, "app")]
os.chdir(root)
reads = []
sys.addaudithook(lambda e, a: e == "open" and isinstance(a[0], (str, os.PathLike))
                 and reads.append(os.path.abspath(os.fsdecode(a[0]))))
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(root, "app", "views", page + ".py"), default_timeout=60).run()
print(json.dumps({"errors": len(at.exception), "reads": reads}))
"""
TOOLS_PAGES = ("core_module", "ai_proposals", "concept_review", "alignment", "base_questions")


@pytest.mark.parametrize("page", TOOLS_PAGES)
def test_the_tools_pages_only_read_from_the_base_what_is_in_git(page):
    run = subprocess.run([sys.executable, "-c", _PAGE_READS, str(ROOT), page],
                         capture_output=True, text=True, encoding="utf-8", timeout=180)
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["errors"] == 0
    output = str(ROOT / "data" / "output")
    read = {os.path.relpath(r, ROOT).replace(os.sep, "/") for r in result["reads"]
            if r.startswith(output)}
    assert read <= set(BASE_FILES + TOOLS_FILES), sorted(read - set(BASE_FILES + TOOLS_FILES))


@pytest.mark.parametrize("path", BASE_FILES + TOOLS_FILES)
def test_every_file_of_the_base_is_there(path):
    assert (ROOT / path).stat().st_size > 0


@pytest.mark.parametrize("module", MODULES)
def test_every_module_loads(module):
    g = Graph().parse(ROOT / f"data/output/{module}.ttl")
    assert len(g) > 0


def test_the_assistant_folders_are_out_of_git():
    assert _git("ls-files", ".agents", ".claude").stdout.strip() == ""


def test_apache_license_and_notice_credit_ontopriv():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text and "Version 2.0, January 2004" in license_text
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for name in ("Elian Alexander Gavilanes Miranda", "Escuela Politécnica Nacional",
                 "Gabriela Suntaxi", "Kelvin Ojeda", "Francisco Rodríguez",
                 "github.com/kelvinojedaepn/OntoPriv", "10.1109/CLEI64178.2024.10700326",
                 "R1", "R2", "DPV 2.3"):
        assert name in notice, name
