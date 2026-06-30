"""Per-language voice pools for Falcon dev vs prod blind battles.

Edit FALCON_BATTLE_VOICES below. Dev and prod always use the **same** voice id
per battle (see config.share_voice_across_providers and scheduler).

Format per language: lists of Murf voice ids under ``male`` / ``female``.
"""
from __future__ import annotations

from typing import Dict, List

VoicePool = Dict[str, List[str]]


def v(m1: str, m2: str, f1: str, f2: str) -> VoicePool:
    return {"male": [m1, m2], "female": [f1, f2]}


# BCP-47 keys used across the arena UI, scheduler, and corpus.
ARENA_LANGUAGES: Dict[str, str] = {
    "en-US": "English (US)",
    "en-IN": "English (India)",
    "en-UK": "English (UK)",
}

# --- Add voices here (shared by falcon_dev and falcon_prod) -------------------
FALCON_BATTLE_VOICES: Dict[str, VoicePool] = {
    "en-US": {
        "male": ["en-US-tyler", "en-US-gordon", "en-US-caleb", "en-US-matthew"],
        "female": [
            "en-US-luna",
            "en-US-alicia",
            "en-US-natalie",
            "en-US-ariana",
        ],
    },
    "en-UK": {
        "male": ["en-UK-benedict", "en-UK-joshua", "en-UK-jake"],
        "female": ["en-UK-lydia", "en-UK-lucy", "en-UK-sharon"],
    },
    "en-IN": {
        "male": ["en-IN-nikhil", "en-IN-samar", "en-IN-abhinav"],
        "female": ["en-IN-anisha", "en-IN-anusha", "en-IN-pooja"],
    },
}

FALCON_PROVIDER_IDS = frozenset({"falcon_dev", "falcon_prod"})


def _languages_with_voices() -> set[str]:
    out: set[str] = set()
    for lang, pool in FALCON_BATTLE_VOICES.items():
        if lang not in ARENA_LANGUAGES:
            continue
        if pool.get("male") or pool.get("female"):
            out.add(lang)
    return out


PROVIDER_SUPPORTED_LANGUAGES: Dict[str, set] = {
    "falcon_dev": _languages_with_voices(),
    "falcon_prod": _languages_with_voices(),
}


def build_provider_languages(omni_dev: bool = True) -> Dict[str, Dict[str, VoicePool]]:
    """Identical pools for dev and prod — scheduler picks one id for both clips."""
    shared = {
        lang: dict(FALCON_BATTLE_VOICES[lang])
        for lang in FALCON_BATTLE_VOICES
        if lang in ARENA_LANGUAGES
    }
    return {
        "falcon_dev": shared,
        "falcon_prod": shared,
    }


def flatten_voice_ids(pools: Dict[str, VoicePool]) -> List[str]:
    out: List[str] = []
    for pool in pools.values():
        for ids in pool.values():
            out.extend(ids)
    return sorted(set(out))


def voice_gender_lookup(*pool_maps: Dict[str, VoicePool]) -> Dict[str, str]:
    """Map voice id -> male|female across language pools."""
    out: Dict[str, str] = {}
    for pools in pool_maps:
        for pool in pools.values():
            for gender, ids in pool.items():
                for vid in ids:
                    out[vid] = gender
    return out


MURF_VOICE_GENDERS = voice_gender_lookup(FALCON_BATTLE_VOICES)

# Back-compat alias used by config.py
_FALCON_BATTLE = FALCON_BATTLE_VOICES
