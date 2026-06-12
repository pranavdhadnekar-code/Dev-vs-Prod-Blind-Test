"""Voice Battle default sentences loaded from bundled TSV (`voice_battle_corpus.tsv`).

Extend or replace sentences by editing that file (two columns: locale\\t<text>, one sentence per row).
Locales **en-IN** and **en-UK** use the **en-US** lines until separate rows exist in the corpus.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_VOICE_BATTLE_TSV = Path(__file__).resolve().parent / "voice_battle_corpus.tsv"

_FALLBACK_LINES = """The quick brown fox jumps over the lazy dog.
The wine glass fills again and laughter breaks through the pressure that had been building quietly for hours.
Just to confirm, the co-applicant's name is spelled M-A-R-I-S-A, correct?
Scientists have made a groundbreaking discovery that could revolutionize renewable energy.
Hello, how can I assist you today with your account inquiry?"""

# Maps `blind_test_2_locale_filter` UI keys → BCP locale tag rows in voice_battle_corpus.tsv.
_VOICE_BATTLE_UI_TO_BCP_LOCALE: Dict[str, str] = {
    "US": "en-US",
    "IN": "en-US",
    "UK": "en-US",
    "HI": "hi-IN",
    "BN": "bn-IN",
    "TA": "ta-IN",
    "FR": "fr-FR",
    "ES": "es-ES",
    "MR": "mr-IN",
    "ML": "ml-IN",
}


def _bundle_mtime_ns() -> int:
    try:
        return int(_VOICE_BATTLE_TSV.stat().st_mtime_ns)
    except OSError:
        return -1


@lru_cache(maxsize=4)
def _lines_by_bcp_locale(_mtime_key: int) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    if not _VOICE_BATTLE_TSV.is_file():
        return {}
    for raw in _VOICE_BATTLE_TSV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tab = line.find("\t")
        if tab == -1:
            continue
        loc = line[:tab].strip()
        sentence = line[tab + 1 :].strip()
        if loc and sentence:
            buckets[loc].append(sentence)
    return dict(buckets)


def bundled_default_sentences_voice_battle(ui_locale: str) -> str:
    """Newline-separated default script lines for the Voice Battle textarea."""
    tag = _VOICE_BATTLE_UI_TO_BCP_LOCALE.get(ui_locale, "en-US")
    by_lc = _lines_by_bcp_locale(_bundle_mtime_ns())
    seq = by_lc.get(tag) or by_lc.get("en-US")
    if not seq:
        return _FALLBACK_LINES
    return "\n".join(seq)
