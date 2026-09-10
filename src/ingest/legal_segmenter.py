from __future__ import annotations

import re
from typing import TypedDict

# Same header pattern as text_loader._count_articles, with the article number
# captured so the text can be split at each header. Keeping both patterns in
# sync guarantees n_articles == candidate_articles.
_ARTICLE_HEADER_RE = re.compile(r"(?im)^\s*art[íi]?(?:culo|\.)\s*(\d+)\s*\.?-")


class Article(TypedDict):
    number: int
    title: str
    text: str
    char_start: int
    char_end: int


class SegmentationSummary(TypedDict):
    source: str
    n_articles: int
    articles: list[Article]


def _clean_start(match: re.Match[str]) -> int:
    """Offset of the article header itself, skipping leading whitespace."""
    raw = match.group(0)
    return match.start() + (len(raw) - len(raw.lstrip()))


def segment_articles(text: str, source: str = "") -> SegmentationSummary:
    """Split a legal text into numbered articles.

    Args:
        text: Plain text of the law, as returned by ``load_legal_text``.
        source: Optional label of where the text came from (file path, name).

    Returns:
        A JSON-serializable summary with the article count and, for each
        article, its number, title, full text and character offsets into
        ``text``.
    """
    matches = list(_ARTICLE_HEADER_RE.finditer(text))
    starts = [_clean_start(m) for m in matches]
    articles: list[Article] = []

    for index, match in enumerate(matches):
        char_start = starts[index]
        char_end = starts[index + 1] if index + 1 < len(matches) else len(text)

        block = text[char_start:char_end].strip()
        marker = match.group(0).strip()

        newline = block.find("\n")
        header_line = block if newline == -1 else block[:newline]
        title = header_line[len(marker):].lstrip(" .\u2013-\t").strip()

        articles.append({
            "number": int(match.group(1)),
            "title": title,
            "text": block,
            "char_start": char_start,
            "char_end": char_end,
        })

    return {
        "source": source,
        "n_articles": len(articles),
        "articles": articles,
    }