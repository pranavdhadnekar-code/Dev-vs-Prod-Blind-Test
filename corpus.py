"""Per-language test corpus for the Voice Arena.

English locales (en-US, en-UK, en-IN) share the same fixed sentence list in
`arena_comparison_corpus.py`. Hindi, Bangla, and Tamil use their own lists.
The same item text is used for both clips in a battle. Each item gets a stable
id for reproducibility (`shared:<index>` or `{locale}:<index>`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import config
from arena_comparison_corpus import (
    ARENA_COMPARISON_TEXTS,
    BANGLA_COMPARISON_TEXTS,
    HINDI_COMPARISON_TEXTS,
    TAMIL_COMPARISON_TEXTS,
)

_CORPUS_LOCALE = "en-shared"
_NATIVE_CORPORA: Dict[str, Sequence[str]] = {
    "hi-IN": HINDI_COMPARISON_TEXTS,
    "bn-IN": BANGLA_COMPARISON_TEXTS,
    "ta-IN": TAMIL_COMPARISON_TEXTS,
}


@dataclass(frozen=True)
class CorpusItem:
    id: str          # stable: "shared:<index>" or "hi-IN:<index>"
    text: str
    language: str    # the requesting arena language (e.g. en-IN)
    corpus_locale: str


def _texts_for(language: str) -> Tuple[Sequence[str], str, str]:
    """Return (texts, item-id prefix, corpus locale tag)."""
    native = _NATIVE_CORPORA.get(language)
    if native is not None:
        return native, language, language
    return ARENA_COMPARISON_TEXTS, "shared", _CORPUS_LOCALE


def get_items(language: str) -> List[CorpusItem]:
    """All corpus items for an arena language (stable order + ids)."""
    texts, prefix, bucket = _texts_for(language)
    return [
        CorpusItem(f"{prefix}:{i}", text, language, bucket)
        for i, text in enumerate(texts)
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
