from __future__ import annotations

import re
from pathlib import Path

import yaml
from dotenv import dotenv_values

from src.config import load_config
from src.validation import logic_inspector as li

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICE = COMPOSE["services"]["app"]
IGNORED = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8")
           .splitlines() if line.strip() and not line.startswith("#")]


def _arg(name: str) -> str:
    return re.search(rf"^ARG {name}=(\S+)$", DOCKERFILE, re.M).group(1)


def _env(name: str) -> str:
    return re.search(rf"\b{name}=(\S+)", DOCKERFILE).group(1)


def test_the_image_has_the_java_and_the_jena_of_the_config():
    s = li.settings()
    assert Path(s.jena_home).name == f"apache-jena-{_arg('JENA_VERSION')}"
    assert re.fullmatch(r"[0-9a-f]{128}", _arg("JENA_SHA512"))        # the download is checked
    assert "sha512sum -c" in DOCKERFILE
    java = re.search(r"^FROM eclipse-temurin:(\d+)-jdk AS java$", DOCKERFILE, re.M)
    assert int(java.group(1)) >= li.MIN_JAVA                     # a JDK: JenaCheck.java is compiled
    assert "assert engine['name'] == 'jena'" in DOCKERFILE        # the build stops without Jena


def test_the_image_has_the_python_and_the_torch_of_the_project():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    python = re.search(r"^FROM python:(\d+\.\d+)-slim$", DOCKERFILE, re.M).group(1)
    assert f"Python {python}" in readme
    pinned = re.findall(r"^torch==\S+$", (ROOT / "requirements.txt").read_text(), re.M)
    assert len(pinned) == 1 and "sed -n 's/^torch==//p' requirements.txt" in DOCKERFILE
    assert "--index-url https://download.pytorch.org/whl/cpu" in DOCKERFILE


def test_the_model_and_chromadb_of_the_image_are_the_ones_the_program_uses():
    example = dotenv_values(ROOT / ".env.example")
    model = _arg("EMBEDDING_MODEL")
    assert model == example["EMBEDDING_MODEL"] == SERVICE["environment"]["EMBEDDING_MODEL"]
    assert _env("CHROMA_PATH") == SERVICE["environment"]["CHROMA_PATH"] == "/app/chroma_db"
    assert DOCKERFILE.index("SentenceTransformer(") < DOCKERFILE.index("HF_HUB_OFFLINE=1") \
        < DOCKERFILE.index("python -m src.alignment.knowledge")   # downloaded, then offline


def test_the_deployed_version_shows_only_the_assistant_with_the_base_read_only():
    env = SERVICE["environment"]
    assert (env["BUILD_TOOLS"], env["EDITABLE_BASE"]) == ("0", "0")      # they win over .env
    assert SERVICE["env_file"] == ".env"


def test_the_workspaces_live_in_a_volume_the_program_can_write():
    folder = load_config()["workspaces"]["dir"]
    assert SERVICE["volumes"] == [f"workspaces:/app/{folder}"]
    assert "workspaces" in COMPOSE["volumes"]                       # named: Docker keeps it
    assert f"mkdir -p /app/chroma_db /app/{folder}" in DOCKERFILE
    assert f"chown app:app /app/chroma_db /app/data /app/{folder}" in DOCKERFILE
    assert DOCKERFILE.index("USER app") < DOCKERFILE.index("COPY --chown=app:app . .")


def test_secrets_and_local_data_stay_out_of_the_image():
    for path in (".env", ".git", ".venv", "chroma_db", "tools", "data/workspaces"):
        assert path in IGNORED, path                 # the key comes only with env_file, at start
