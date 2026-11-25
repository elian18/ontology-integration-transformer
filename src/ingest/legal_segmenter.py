"""Split a law into numbered articles, without AI (deterministic and explainable).

A heading is a line that starts with a number, with or without a keyword before it (Artículo,
Art., Article, Section, Sección, §) and a separator after it (. ° — : or the end of the line).
The headings of one family whose numbers increase form a chain; the longest chain is the
list of articles. An index (one heading per line, followed by the body) is not taken for the
body, and the bare numbers ("1. Text") are used only when no keyword forms a chain.
"""
from __future__ import annotations

import re
from statistics import median
from typing import TypedDict

SEGMENTER_VERSION = 3              # S11-T11 (2: S7-T07, every heading style; 1: Sprint 2)
MAX_GAP = 3                        # the chain may skip at most 2 numbers (badly extracted PDF)
INDEX_GAP = 120                    # an index: its headings are this close (median, characters)
MIN_CHAIN = 2                      # fewer keyword headings: the bare numbers are tried
MAX_TITLE = 100                    # a longer "title" is body text
BARE = "bare"

# Sprint 2 pattern, unchanged (text_loader counts with the same chains).
_SPRINT2_RE = re.compile(r"(?im)^\s*art[íi]?(?:culo|\.)\s*(\d+)\s*\.?-")
_ES_WORD = r"(?:ART[ÍI]CULO|Art[íi]culo)"
_ES_PERIOD_RE = re.compile(rf"(?m)^[ \t]*{_ES_WORD}[ \t]+(\d+)[ \t]*"
                           r"(?:[°º]|o(?=[ \t]*\.))?[ \t]*\.(?!-)")
_ES_DASH_RE = re.compile(rf"(?m)^[ \t]*{_ES_WORD}[ \t]+(\d+)[ \t]*"
                         r"(?:[°º]|o(?=[ \t]*[.:–—-]))?[ \t]*\.?[ \t]*[:–—-]")
# "Artículo" at the end of a line and its number on the next one (a cut heading).
_ES_SPLIT_RE = re.compile(rf"(?m)^[ \t]*{_ES_WORD}[ \t]*\r?\n[ \t]*(\d+)[ \t]*(?:[°º])?[ \t]*"
                          r"[.:–—-]")
_ES_ALONE_RE = re.compile(rf"(?m)^[ \t]*{_ES_WORD}[ \t]+(\d+)[ \t]*(?:[°º])?[ \t]*\r?$")
_EN_ALONE_RE = re.compile(r"(?m)^[ \t]*(?:ARTICLE|Article)[ \t]+(\d+)[ \t]*\r?$")
_EN_INLINE_RE = re.compile(r"(?m)^[ \t]*(?:ARTICLE|Article)[ \t]+(\d+)[ \t]*(?:[.:]|[-–—])"
                           r"[ \t]*(?=[A-ZÁÉÍÓÚÑ])")
_SECTION_WORD = r"(?:SECTION|Section|SEC\.|Sec\.|SECCI[ÓO]N|Secci[óo]n|§)"
_SECTION_ALONE_RE = re.compile(rf"(?m)^[ \t]*{_SECTION_WORD}[ \t]*(\d+)[ \t]*\r?$")
_SECTION_INLINE_RE = re.compile(rf"(?m)^[ \t]*{_SECTION_WORD}[ \t]*(\d+)[ \t]*[°º]?[ \t]*"
                                r"(?:[.:]|[-–—])[ \t]*(?=[(A-ZÁÉÍÓÚÑ])")
# "1. Text", "1.- Text", "1. —Text", "1 — Text": a number without keyword (Kenya, 2019).
_BARE_RE = re.compile(r"(?m)^[ \t]*(\d{1,3})[ \t]*(?:\.[ \t]*[–—-]?|[–—])[ \t]*"
                      r"(?=[(\"“A-ZÁÉÍÓÚÑ])")
# (style, family, pattern), in priority order when two match at the same place.
PATTERNS = (("sprint2", "sprint2", _SPRINT2_RE),
            ("es_period", "es", _ES_PERIOD_RE), ("es_dash", "es", _ES_DASH_RE),
            ("es_split", "es", _ES_SPLIT_RE), ("es_alone", "es", _ES_ALONE_RE),
            ("en_alone", "en", _EN_ALONE_RE), ("en_inline", "en", _EN_INLINE_RE),
            ("section_alone", "section", _SECTION_ALONE_RE),
            ("section_inline", "section", _SECTION_INLINE_RE),
            (BARE, BARE, _BARE_RE))
_ALONE_STYLES = ("es_alone", "en_alone", "section_alone")
_TOC_RE = re.compile(r"(?:\.\s?){4,}\s*\d*\s*$|…\s*\d*\s*$")

_STOPWORDS = {
    "es": {"de", "la", "que", "el", "en", "los", "del", "las", "por", "para", "con", "una",
           "se", "su", "al", "lo", "datos", "sus", "como", "ley"},
    "en": {"the", "of", "and", "to", "in", "shall", "be", "or", "for", "by", "with", "that",
           "this", "which", "data", "on", "is", "as", "an", "are"},
}


class Article(TypedDict):
    number: int
    title: str
    text: str
    char_start: int
    char_end: int


class SegmentationSummary(TypedDict, total=False):
    source: str
    n_articles: int
    articles: list[Article]
    language: str                  # es | en
    style: str | None              # family of the headings used (sprint2|es|en|section|bare)
    missing: list[int]             # numbers skipped by the chain
    rejected: int                  # heading-like lines left out (cross references, index...)


class _Candidate(TypedDict):
    start: int
    end: int
    number: int
    style: str
    family: str
    marker: str


# --------------------------------------------------------------------------- finding headings

def _clean_start(match: re.Match[str]) -> int:
    """Offset of the article header itself, skipping leading whitespace."""
    raw = match.group(0)
    return match.start() + (len(raw) - len(raw.lstrip()))


def _line_at(text: str, start: int) -> str:
    end = text.find("\n", start)
    return text[start:] if end == -1 else text[start:end]


def candidates(text: str) -> list[_Candidate]:
    """Every heading-like line, one per position (the first pattern wins), in text order."""
    found: dict[int, _Candidate] = {}
    for style, family, pattern in PATTERNS:
        for m in pattern.finditer(text):
            start = _clean_start(m)
            if start in found or _TOC_RE.search(_line_at(text, start)):
                continue
            found[start] = {"start": start, "end": m.end(), "number": int(m.group(1)),
                            "style": style, "family": family, "marker": m.group(0).strip()}
    return [found[k] for k in sorted(found)]


def _chain(cands: list[_Candidate], first: int) -> list[_Candidate]:
    chain = [cands[first]]
    for c in cands[first + 1:]:
        if chain[-1]["number"] < c["number"] <= chain[-1]["number"] + MAX_GAP:
            chain.append(c)
    return chain


def _chains(cands: list[_Candidate]) -> list[list[_Candidate]]:
    """Per family, the chain that starts at each article 1 (or at the first heading)."""
    out = []
    for family in dict.fromkeys(c["family"] for c in cands):
        own = [c for c in cands if c["family"] == family]
        starts = [i for i, c in enumerate(own) if c["number"] == 1] or [0]
        out += [_chain(own, i) for i in starts]
    return out


def _is_index(chain: list[_Candidate], chains: list[list[_Candidate]]) -> bool:
    """Headings one per line, and after them another chain at least half as long: an index."""
    gaps = [b["start"] - a["start"] for a, b in zip(chain, chain[1:])]
    return bool(gaps) and median(gaps) < INDEX_GAP and any(
        other[0]["start"] > chain[-1]["start"] and 2 * len(other) >= len(chain)
        for other in chains)


def _longest(chains: list[list[_Candidate]]) -> list[_Candidate]:
    """The longest chain that is not an index; on a tie, the one that starts later."""
    kept = [ch for ch in chains if not _is_index(ch, chains)]
    return max(kept, key=lambda ch: (len(ch), ch[0]["start"]), default=[])


def select_headings(cands: list[_Candidate]) -> list[_Candidate]:
    """The articles: the longest chain of increasing numbers of one family (see module doc)."""
    chains = _chains(cands)
    best = _longest([ch for ch in chains if ch[0]["family"] != BARE])
    return best if len(best) >= MIN_CHAIN else _longest(chains)


# --------------------------------------------------------------------------- titles

def _caption(rest: str) -> str:
    """The title written after the number: "(Objeto)." or "Objeto. La ley..." (a sentence
    that ends on the heading line); body text is no title."""
    rest = rest.strip().lstrip(".:–—-").strip()
    if re.match(r"\(\w{1,4}\)", rest):        # (1), (a), (iv): the body starts
        return ""
    paren = re.match(r"\(([^()]+)\)", rest)
    end = re.search(r"\.(?:\s|$)", rest)
    title = paren.group(1) if paren else rest[:end.start()] if end else ""
    title = title.strip(" .\t")
    return title if len(title) <= MAX_TITLE else ""


def _next_line(block: str) -> str:
    lines = [ln.strip() for ln in block.splitlines()[1:] if ln.strip()]
    if not lines:
        return ""
    line = lines[0]
    if len(line) > MAX_TITLE or re.match(r"^(?:\(?\d+[.)]|\([a-z]\))", line):
        return ""                              # body text, not a title
    return line.rstrip(" .")


def _title(c: _Candidate, block: str) -> str:
    rest = block[len(c["marker"]):].split("\n", 1)[0] if block.startswith(c["marker"]) else ""
    if c["style"] == "sprint2":                # Sprint 2 rule, unchanged
        return rest.lstrip(" .–-\t").strip()
    if c["style"] in _ALONE_STYLES:
        return _next_line(block)
    return _caption(rest)


# --------------------------------------------------------------------------- language

def detect_language(text: str) -> str:
    """``es`` or ``en``: which list of common words appears more often."""
    words = re.findall(r"[a-záéíóúñü]+", text[:200_000].lower())
    counts = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORDS.items()}
    return "en" if counts["en"] > counts["es"] else "es"


# --------------------------------------------------------------------------- segmenting

def segment_articles(text: str, source: str = "") -> SegmentationSummary:
    """Split a legal text into numbered articles.

    Args:
        text: Plain text of the law, as returned by ``load_legal_text`` (for an uploaded law,
            after ``law_text.clean_law_file``).
        source: Optional label of where the text came from (file path, name).

    Returns:
        A JSON-serializable summary: the articles (number, title, full text and character
        offsets into ``text``), the language, the family of headings used, the numbers the
        chain skipped and how many heading-like lines were left out.
    """
    cands = candidates(text)
    chosen = select_headings(cands)
    articles: list[Article] = []
    for index, c in enumerate(chosen):
        char_start = c["start"]
        char_end = chosen[index + 1]["start"] if index + 1 < len(chosen) else len(text)
        block = text[char_start:char_end].strip()
        articles.append({
            "number": c["number"],
            "title": _title(c, block),
            "text": block,
            "char_start": char_start,
            "char_end": char_end,
        })
    numbers = [a["number"] for a in articles]
    missing = [n for a, b in zip(numbers, numbers[1:]) for n in range(a + 1, b)]
    family = chosen[0]["family"] if chosen else None
    counted = [c for c in cands if c["family"] != BARE or family == BARE]    # not inner lists
    return {
        "source": source,
        "n_articles": len(articles),
        "articles": articles,
        "language": detect_language(text),
        "style": family,
        "missing": missing,
        "rejected": len(counted) - len(chosen),
    }


def count_articles(text: str) -> int:
    """How many articles ``segment_articles`` finds (the tentative count of text_loader)."""
    return len(select_headings(candidates(text)))