"""Load the raw legal text (LOPDP or another law) from .txt or .pdf.

For PDF we only extract the raw text (pypdf); PDF is an input-only source document,
never an ontology format. No article segmentation here: that is Sprint 2."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass
class TextReport:
    path: str
    source: str            # "txt" | "pdf"
    sha256: str
    n_chars: int
    n_lines: int
    n_nonempty_lines: int
    encoding: str          # "utf-8" | "latin-1" | "pdf-extract"
    candidate_articles: int   # tentative count, NOT segmentation
    text: str = ""

    @property
    def loaded(self) -> bool:
        return self.n_chars > 0


def _count_articles(text: str) -> int:
    """Tentative count: the articles the segmenter would find (S7-T07: every heading style)."""
    from src.ingest.legal_segmenter import count_articles
    return count_articles(text)


def _newlines(text: str) -> str:
    """Windows (CRLF) and old Mac (CR) line endings become LF, as with any text file read
    by Python, so the article headings are found on every system (S7-T07)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _load_txt(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    try:
        return _newlines(data.decode("utf-8")), "utf-8"
    except UnicodeDecodeError:
        return _newlines(data.decode("latin-1")), "latin-1"


def _load_pdf(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages), "pdf-extract"


def load_legal_text(path: str | Path) -> TextReport:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Texto normativo no encontrado: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, encoding = _load_pdf(path)
        source = "pdf"
    elif suffix in {".txt", ".text"}:
        text, encoding = _load_txt(path)
        source = "txt"
    else:
        raise ValueError(f"Fuente de texto no soportada: {suffix} (usa .txt o .pdf)")

    lines = text.splitlines()
    return TextReport(
        path=str(path),
        source=source,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        n_chars=len(text),
        n_lines=len(lines),
        n_nonempty_lines=sum(1 for ln in lines if ln.strip()),
        encoding=encoding,
        candidate_articles=_count_articles(text),
        text=text,
    )