"""Per-language test corpus for the Voice Arena.

Items live in `voice_battle_corpus.tsv` (two columns: BCP locale tag <TAB> text).
The same item text is used for both clips in a battle. Each item gets a stable
id derived from its corpus bucket + index so battles remain reproducible and
auditable. Add languages/items by editing the TSV.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import config
from voice_battle_locale_defaults import _bundle_mtime_ns, _lines_by_bcp_locale


@dataclass(frozen=True)
class CorpusItem:
    id: str          # stable: "<corpus_locale>:<index>"
    text: str
    language: str    # the requesting arena language (e.g. en-IN)
    corpus_locale: str  # the TSV bucket actually used (e.g. en-US)


@lru_cache(maxsize=64)
def _items_for_corpus_locale(corpus_locale: str, _mtime_key: int) -> List[tuple]:
    by_lc = _lines_by_bcp_locale(_mtime_key)
    lines = by_lc.get(corpus_locale) or by_lc.get("en-US") or []
    return [(f"{corpus_locale}:{i}", text) for i, text in enumerate(lines)]


def get_items(language: str) -> List[CorpusItem]:
    """All corpus items for an arena language (stable order + ids)."""
    corpus_locale = config.corpus_locale_for_language(language)
    raw = _items_for_corpus_locale(corpus_locale, _bundle_mtime_ns())
    return [CorpusItem(iid, text, language, corpus_locale) for iid, text in raw]


def get_item(language: str, item_id: str) -> Optional[CorpusItem]:
    for it in get_items(language):
        if it.id == item_id:
            return it
    return None


def item_count(language: str) -> int:
    return len(get_items(language))


def all_counts() -> Dict[str, int]:
    return {lang: item_count(lang) for lang in config.LANGUAGES}
