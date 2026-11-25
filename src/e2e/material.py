"""The new material of Sprint 11 (S11-T11): four ontologies and three laws, downloaded from
their sources into data/input/s11. Not in git: not every source lets it be shared again."""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FOLDER = "data/input/s11"


@dataclass(frozen=True)
class Source:
    name: str
    urls: tuple[str, ...]            # official sources, tried in order
    note: str
    page: bool = False               # a web page: only its text is kept (.txt)


SOURCES = (
    Source("gconsent.ttl", ("https://raw.githubusercontent.com/coolharsh55/GConsent/master/"
                            "gconsent.ttl",), "GConsent (H. J. Pandit), CC BY 4.0"),
    Source("gdprov.ttl", ("https://raw.githubusercontent.com/coolharsh55/GDPRov/master/"
                          "gdprov.ttl",), "GDPRov (H. J. Pandit), CC BY 4.0"),
    Source("duo.owl", ("https://raw.githubusercontent.com/EBISPOT/DUO/master/duo.owl",),
           "Data Use Ontology (GA4GH), CC BY 4.0"),
    Source("pizza.owl", ("https://raw.githubusercontent.com/owlcs/pizza-ontology/master/"
                         "pizza.owl",), "Pizza (Universidad de Manchester), de control"),
    Source("mexico-lfpdppp-2025.txt", (       # the PDF of diputados.gob.mx does not answer
        "https://www.ordenjuridico.gob.mx/Documentos/Federal/html/wo125102.html",
        "https://www.te.gob.mx/legislacion/media/files/f3ca59b75a57b6f.htm"),
           "LFPDPPP de México (2025), texto oficial", page=True),
    Source("argentina-ley-25326.txt", ("https://www.argentina.gob.ar/normativa/nacional/"
                                       "ley-25326-2000-64790/actualizacion",),
           "Ley 25.326 de Argentina, texto actualizado", page=True),
    Source("kenya-dpa-2019.pdf", (
        "https://interior.go.ke/sites/default/files/2024-09/"
        "27.%20Data%20Protection%20Act%202019.pdf",
        "http://kenyalaw.org/kl/fileadmin/pdfdownloads/Acts/2019/"
        "TheDataProtectionAct__No24of2019.pdf",
        "https://www.odpc.go.ke/wp-content/uploads/2024/02/TheDataProtectionAct__No24of2019.pdf"),
        "Data Protection Act 2019 de Kenia"))


def page_text(raw: bytes, charset: str | None = None) -> str:
    """The text of a web page, a block per line (what a person would copy from it). The line
    breaks of the HTML source are spaces, as in the browser: "Artículo\\n1." is "Artículo 1."."""
    found = re.search(rb'charset=["\']?([\w-]+)', raw[:2000])
    text = raw.decode(charset or (found.group(1).decode() if found else "utf-8"),
                      errors="replace")
    text = re.sub(r"(?is)<(script|style|head|nav|header|footer)\b.*?</\1>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|h\d|li|tr)>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def fetch(url: str) -> tuple[bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TIC EPN)"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        return answer.read(), answer.headers.get_content_charset()


def first_answer(source: Source, get) -> tuple[bytes, str | None, str]:
    """What the first source that answers with something gives (an empty answer is a failure)."""
    failures = []
    for url in source.urls:
        try:
            raw, charset = get(url)
        except OSError as exc:
            failures.append(f"{urllib.parse.urlsplit(url).netloc}: {exc}")
            continue
        host = urllib.parse.urlsplit(url).netloc
        if not raw.strip():
            failures.append(f"{host}: respuesta vacía")
        elif source.name.endswith(".pdf") and not raw.lstrip().startswith(b"%PDF"):
            failures.append(f"{host}: respondió una página web, no un PDF")
        else:
            return raw, charset, url
    raise OSError("; ".join(failures))


def download(folder: Path, *, get=fetch, only: set[str] | None = None) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for source in SOURCES:
        if only and source.name not in only:
            continue
        try:
            raw, charset, url = first_answer(source, get)
        except OSError as exc:
            lines.append(f"  {source.name}: no se pudo descargar ({exc})")
            continue
        data = page_text(raw, charset).encode("utf-8") if source.page else raw
        (folder / source.name).write_bytes(data)
        lines.append(f"  {source.name}: {len(data) / 1024:.0f} KB, sha256 "
                     f"{hashlib.sha256(data).hexdigest()[:12]} — {source.note} "
                     f"({urllib.parse.urlsplit(url).netloc})")
    return lines


def _run(argv: list[str] | None = None) -> int:
    from src.config import ROOT
    parser = argparse.ArgumentParser(
        prog="py -m src.e2e.material",
        description="Descarga el material nuevo del Sprint 11 desde sus fuentes.")
    parser.add_argument("--carpeta", type=Path, default=ROOT / FOLDER)
    parser.add_argument("--solo", nargs="+", metavar="ARCHIVO")
    args = parser.parse_args(argv)
    print("Material nuevo del Sprint 11 (S11-T11)")
    print("\n".join(download(args.carpeta, only=set(args.solo) if args.solo else None)))
    print(f"  Carpeta: {args.carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
