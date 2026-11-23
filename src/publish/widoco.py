"""The page of the base (S11-T06): OOPS! on OntoPriv and on the base, the file of the base with
that comparison, and WIDOCO, which writes the page in Spanish and English in docs/."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import urllib.error
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.console import counted
from src.publish import base_doc, oops

LANGS = "es-en"
GENERATED = ("index.html", "index-es.html", "index-en.html", "ontology.ttl", "ontology.owl",
             "ontology.jsonld", "ontology.nt", "resources", ".nojekyll")
LEFT_OUT = ("provenance", "readme.md")     # its provenance page shows a folder of this computer


@dataclass(frozen=True)
class Settings:
    jar: Path
    java: str
    docs: Path
    ontopriv: Path
    output: Path                    # data/output: the modules of the base
    oops_timeout: int
    widoco_timeout: int


def settings(cfg: dict | None = None) -> Settings:
    from src.alignment.inherit import base_paths
    from src.config import ROOT, load_config
    cfg = cfg if cfg is not None else load_config()
    block = cfg.get("publish") or {}

    def path(value: str) -> Path:
        return Path(value) if Path(value).is_absolute() else ROOT / value
    return Settings(
        jar=path(block.get("widoco_jar", "tools/widoco/widoco.jar")),
        java=(cfg.get("validation") or {}).get("java", "java"),
        docs=path(block.get("docs_dir", "docs")),
        ontopriv=path((cfg.get("inputs") or {}).get("ontology", "data/input/ontopriv.rdf")),
        output=base_paths(cfg)["candidates"].parent,
        oops_timeout=int(block.get("oops_timeout_seconds", 300)),
        widoco_timeout=int(block.get("widoco_timeout_seconds", 600)))


def evaluate(s: Settings, graphs, *, post=oops.post, today: str | None = None) -> dict:
    """OOPS! on OntoPriv as its authors published it and on the base as it is in data/output
    (with its header, without the sentences written for the page)."""
    base, _ = base_doc.build(graphs, document=False)
    answers = {"ontopriv": post(oops.request_body(s.ontopriv.read_text(encoding="utf-8")),
                                s.oops_timeout),
               "base": post(oops.request_body(base.serialize(format="xml")), s.oops_timeout)}
    return oops.save(s.docs / "oops", answers, today or date.today().isoformat())


def widoco_command(s: Settings, ontology: Path, out: Path) -> list[str]:
    return [s.java, "-jar", str(s.jar), "-ontFile", str(ontology), "-outFolder", str(out),
            "-lang", LANGS, "-rewriteAll", "-uniteSections", "-noPlaceHolderText",
            "-excludeProvenance"]


def run_widoco(s: Settings, ontology: Path, work: Path, run=subprocess.run) -> Path:
    out = work / "widoco"
    done = run(widoco_command(s, ontology, out), cwd=work, capture_output=True, text=True,
               timeout=s.widoco_timeout)
    page = out / "doc"                     # with an IRI that ends in "/", WIDOCO writes it there
    page = page if (page / "index-es.html").exists() else out
    if done.returncode != 0 or not (page / "index-es.html").exists():
        tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-5:])
        raise RuntimeError(f"WIDOCO no generó la página:\n{tail}")
    return page


def publish(out: Path, docs: Path) -> None:
    """What WIDOCO wrote replaces the page before it; the rest of docs/ stays."""
    docs.mkdir(parents=True, exist_ok=True)
    for name in (*GENERATED, *LEFT_OUT):
        old = docs / name
        if old.is_dir():
            shutil.rmtree(old)
        elif old.exists():
            old.unlink()
        if name in LEFT_OUT:
            continue
        if (out / name).is_dir():
            shutil.copytree(out / name, old)
        elif (out / name).exists():
            shutil.copy2(out / name, old)
    shutil.copy2(out / "index-es.html", docs / "index.html")      # GitHub Pages opens this one
    (docs / ".nojekyll").write_text("", encoding="utf-8")         # the files as they are


def build(s: Settings, *, ask_oops: bool, post=oops.post, run=subprocess.run,
          today: str | None = None) -> list[str]:
    graphs = base_doc.load_modules(s.output)
    lines = ["Página de la base con WIDOCO (S11-T06)"]
    if ask_oops:
        try:
            evaluate(s, graphs, post=post, today=today)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            lines.append(f"  OOPS! no respondió ({exc}): se usa la evaluación guardada")
    result = oops.load(s.docs / "oops")
    if result:
        lines.append(f"  OOPS!: evaluación del {result['date']} (OntoPriv: "
                     f"{counted(len(result['ontopriv']), 'problema', 'problemas')}; base: "
                     f"{counted(len(result['base']), 'problema', 'problemas')})")
    else:
        lines.append("  OOPS!: sin evaluación guardada (corre con --oops)")
    extra = {lang: oops.table(result, lang) for lang in ("es", "en")} if result else None
    g, _ = base_doc.build(graphs, extra)
    ontology = s.output / base_doc.DOC_FILE
    ontology.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(ontology, format="turtle")
    if not s.jar.exists():
        raise FileNotFoundError(f"falta WIDOCO en {s.jar} (ver el README)")
    with tempfile.TemporaryDirectory() as work:
        publish(run_widoco(s, ontology, Path(work), run=run), s.docs)
    lines += [f"  Archivo de la base: {ontology}",
              f"  Página: {s.docs / 'index.html'} (español) y {s.docs / 'index-en.html'} (inglés)"]
    return lines


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m src.publish.widoco",
        description="Genera la página de la base con WIDOCO en docs/, con la comparación de "
                    "OOPS! entre OntoPriv original y la base.")
    parser.add_argument("--oops", action="store_true",
                        help="vuelve a pedir la evaluación a OOPS! (necesita internet)")
    args = parser.parse_args(argv)
    try:
        lines = build(settings(), ask_oops=args.oops)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Error: {exc}")
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
