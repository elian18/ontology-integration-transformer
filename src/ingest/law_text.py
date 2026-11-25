from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from src.ingest.legal_segmenter import BARE, candidates

EDGE_LINES = 2
MIN_PAGES = 3
MIN_SHARE = 0.5
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
             "ﬅ": "st", "ﬆ": "st"}
_MARKER_LINE = re.compile(r"^\s*[▼►◄]\s*[A-Z]?\d*\s*$")          # ▼B, ▼C1, ►M2
_INLINE_MARKER = re.compile(r"►\s*[A-Z]\d+\s*|\s*◄")              # ►C1 text ◄ -> text
_SPACED = re.compile(r"^\s*(?:\S ){4,}\S?\s*$")                    # S e c t i o n 1


def load_law_pages(path: str | Path) -> list[str]:
    """The text of each page of a PDF; a .txt is one page."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [(page.extract_text() or "") for page in PdfReader(str(path)).pages]
    from src.ingest.text_loader import _load_txt
    return [_load_txt(path)[0]]


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def _edges(page: str) -> list[str]:
    lines = [ln for ln in page.splitlines() if ln.strip()]
    return lines[:EDGE_LINES] + lines[-EDGE_LINES:]


def _is_heading(line: str) -> bool:
    """An article heading with keyword ("1. Text" may be a page footer, S11-T11)."""
    return any(c["family"] != BARE for c in candidates(line.strip() + "\n"))


def furniture(pages: list[str]) -> set[str]:
    """Normalized lines repeated at the top or bottom of most pages (never a heading)."""
    seen: Counter = Counter()
    headings: set[str] = set()
    for page in pages:
        edges = _edges(page)
        seen.update({_norm(ln) for ln in edges})
        headings.update(_norm(ln) for ln in edges if _is_heading(ln))
    needed = max(MIN_PAGES, MIN_SHARE * len(pages))
    return {n for n, k in seen.items() if k >= needed and n and n not in headings}


def clean_pages(pages: list[str]) -> tuple[str, dict]:
    """Joined text without page furniture, EUR-Lex markers, spaced headings or ligatures."""
    repeated = furniture(pages)
    removed = markers = spaced = 0
    out_pages = []
    for page in pages:
        lines = page.splitlines()
        nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(nonempty[:EDGE_LINES] + nonempty[-EDGE_LINES:])
        kept = []
        for i, line in enumerate(lines):
            if i in edge_idx and _norm(line) in repeated:
                removed += 1
                continue
            if _MARKER_LINE.match(line):
                markers += 1
                continue
            if _SPACED.match(line):
                spaced += 1
                continue
            new = _INLINE_MARKER.sub("", line) if ("►" in line or "◄" in line) else line
            markers += new != line
            kept.append(new)
        out_pages.append("\n".join(kept))
    text = "\n".join(out_pages)
    ligatures = sum(text.count(k) for k in LIGATURES)
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    return text, {"pages": len(pages), "furniture": sorted(repeated),
                  "removed_lines": removed, "markers": markers, "ligatures": ligatures,
                  "spaced_headings": spaced}


def clean_law_file(path: str | Path) -> tuple[str, dict]:
    """Load an uploaded law (.pdf or .txt) and clean it (see the module doc)."""
    return clean_pages(load_law_pages(path))