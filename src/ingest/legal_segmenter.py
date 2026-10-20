"""Split a law into its numbered articles (S2-T01, generalized in S7-T07).

Sprint 2 only knew the heading of the LOPDP, ``Art. 1.-Objeto y finalidad.-...``. Since Sprint 7
a person can upload another data-protection law, in Spanish or in English, so three families of
headings are recognized, always at the start of a line:

- ``sprint2``: ``Art. 5.-``, ``Artículo 5.-``, ``Articulo 5-`` (the LOPDP). Its title keeps the
  rule of Sprint 2 (the rest of the heading line), so the base LOPDP is segmented EXACTLY as
  before: same 77 articles, same titles, texts and offsets; its index "normativa" and every
  cache built on it stay valid.
- ``es``: ``Artículo 1°. Objeto.``, ``ARTÍCULO 1o. OBJETO.``, ``Artículo 10. Casos ...``
  (Colombia, Peru, ...), or ``Artículo 5`` alone on its line with the title on the next one.
  The title is the first sentence after the number.
- ``en``: ``Article 5`` alone on its line with the title on the next one (the GDPR as EUR-Lex
  publishes it), or ``Article 5 - Title`` / ``Article 5. Title`` on one line (the title is
  then its first sentence).

A text with Windows line endings (CRLF) is read the same way. A heading must start with a
capital letter (``artículo 2°.`` at the start of a line is a cross
reference, not a heading) and must not be a table-of-contents line (dot leaders). Then the
NUMBERS decide: within each family, a chain starts at every article 1 and grows with headings
whose number is higher than the previous one by at most ``MAX_GAP``; the longest chain wins
(the later one on a tie: a table of contents comes before the body). Every heading left out is
counted (``rejected``) and every number skipped by the chain is reported (``missing``), so a
badly extracted PDF is visible instead of silent.

The summary also says in which language the law is written (``es`` or ``en``, by counting
common words), which the extraction of Sprint 7 needs.
"""
from __future__ import annotations

import re
from typing import TypedDict

SEGMENTER_VERSION = 2              # S7-T07 (1: Sprint 2, only the LOPDP heading)
MAX_GAP = 3                        # the chain may skip at most 2 numbers (badly extracted PDF)

# Sprint 2 pattern, unchanged (text_loader counts with the same chains).
_SPRINT2_RE = re.compile(r"(?im)^\s*art[íi]?(?:culo|\.)\s*(\d+)\s*\.?-")
_ES_PERIOD_RE = re.compile(r"(?m)^[ \t]*(?:ART[ÍI]CULO|Art[íi]culo)[ \t]+(\d+)[ \t]*"
                           r"(?:[°º]|o(?=[ \t]*\.))?[ \t]*\.(?!-)")
_ES_ALONE_RE = re.compile(r"(?m)^[ \t]*(?:ART[ÍI]CULO|Art[íi]culo)[ \t]+(\d+)[ \t]*(?:[°º])?"
                          r"[ \t]*\r?$")
_EN_ALONE_RE = re.compile(r"(?m)^[ \t]*(?:ARTICLE|Article)[ \t]+(\d+)[ \t]*\r?$")
_EN_INLINE_RE = re.compile(r"(?m)^[ \t]*(?:ARTICLE|Article)[ \t]+(\d+)[ \t]*(?:[.:]|[-–—])"
                           r"[ \t]*(?=[A-ZÁÉÍÓÚÑ])")
# (style, family, pattern), in priority order when two match at the same place.
PATTERNS = (("sprint2", "sprint2", _SPRINT2_RE),
            ("es_period", "es", _ES_PERIOD_RE), ("es_alone", "es", _ES_ALONE_RE),
            ("en_alone", "en", _EN_ALONE_RE), ("en_inline", "en", _EN_INLINE_RE))
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
    style: str | None              # family of the headings used (sprint2 | es | en)
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


def select_headings(cands: list[_Candidate]) -> list[_Candidate]:
    """The articles: the longest chain of increasing numbers of one family (see module doc)."""
    best: list[_Candidate] = []
    for family in dict.fromkeys(c["family"] for c in cands):
        own = [c for c in cands if c["family"] == family]
        starts = [i for i, c in enumerate(own) if c["number"] == 1] or [0]
        for i in starts:
            chain = _chain(own, i)
            if (len(chain), chain[0]["start"]) > (len(best), best[0]["start"] if best else -1):
                best = chain
    return best


# --------------------------------------------------------------------------- titles

def _first_sentence(rest: str) -> str:
    rest = rest.strip()
    m = re.search(r"\.(?:\s|$)", rest)
    return (rest[:m.start()] if m else rest).strip(" .\t")


def _next_line(block: str) -> str:
    lines = [ln.strip() for ln in block.splitlines()[1:] if ln.strip()]
    if not lines:
        return ""
    line = lines[0]
    if len(line) > 150 or re.match(r"^(?:\(?\d+[.)]|\([a-z]\))", line):
        return ""                              # body text, not a title
    return line.rstrip(" .")


def _title(c: _Candidate, block: str) -> str:
    header_line = block.split("\n", 1)[0]
    rest = header_line[len(c["marker"]):] if header_line.startswith(c["marker"]) else ""
    if c["style"] == "sprint2":                # Sprint 2 rule, unchanged
        return rest.lstrip(" .–-\t").strip()
    if c["style"] in ("es_alone", "en_alone"):
        return _next_line(block)
    return _first_sentence(rest)                # es_period, en_inline


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
    return {
        "source": source,
        "n_articles": len(articles),
        "articles": articles,
        "language": detect_language(text),
        "style": chosen[0]["family"] if chosen else None,
        "missing": missing,
        "rejected": len(cands) - len(chosen),
    }


def count_articles(text: str) -> int:
    """How many articles ``segment_articles`` finds (the tentative count of text_loader)."""
    return len(select_headings(candidates(text)))