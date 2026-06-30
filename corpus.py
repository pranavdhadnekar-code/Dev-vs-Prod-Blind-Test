"""Per-language test corpus for the Voice Arena.

All arena languages (en-US, en-UK, en-IN) share the same fixed sentence list in
`arena_comparison_corpus.py`. The same item text is used for both clips in a
battle. Each item gets a stable id (`shared:<index>`) for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import config
from arena_comparison_corpus import ARENA_COMPARISON_TEXTS

_CORPUS_LOCALE = "en-shared"


@dataclass(frozen=True)
class CorpusItem:
    id: str          # stable: "shared:<index>"
    text: str
    language: str    # the requesting arena language (e.g. en-IN)
    corpus_locale: str  # always en-shared


def get_items(language: str) -> List[CorpusItem]:
    """All corpus items for an arena language (stable order + ids)."""
    return [
        CorpusItem(f"shared:{i}", text, language, _CORPUS_LOCALE)
        for i, text in enumerate(ARENA_COMPARISON_TEXTS)
    ]


def get_item(language: str, item_id: str) -> Optional[CorpusItem]:
    for it in get_items(language):
        if it.id == item_id:
            return it
    return None


def item_count(language: str) -> int:
    return len(get_items(language))


def all_counts() -> Dict[str, int]:
    return {lang: item_count(lang) for lang in config.LANGUAGES}
