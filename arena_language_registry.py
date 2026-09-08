"""Per-language voice pools for Falcon dev vs prod blind battles.

Edit FALCON_BATTLE_VOICES below. Dev and prod always use the **same** registry id
per battle (see config.share_voice_across_providers and scheduler).

Internal ids are ``{language}-{voice_id}`` (e.g. ``en-US-Joshua``) so the same
Falcon voice name can appear in more than one locale. Synthesis looks up
FALCON_VOICE_CONFIGS to send the API ``voiceId``, ``style``, and optional
``multiNativeLocale``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

VoicePool = Dict[str, List[str]]
VoiceSpec = Union[Tuple[str, str], Tuple[str, str, str]]


@dataclass(frozen=True)
class FalconVoiceConfig:
    """API fields for one battle voice in a language."""

    language: str
    voice_id: str
    style: str
    multi_native_locale: Optional[str] = None
    model: str = "Falcon"

    @property
    def registry_id(self) -> str:
        return f"{self.language}-{self.voice_id}"


# BCP-47 keys used across the arena UI, scheduler, and corpus.
ARENA_LANGUAGES: Dict[str, str] = {
    "en-US": "English (US)",
    "en-IN": "English (India)",
    "en-UK": "English (UK)",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
}

# Blind-UI locale key (US/IN/UK/HI/BN) for catalog accent matching.
LANGUAGE_UI_ACCENT: Dict[str, str] = {
    "en-US": "US",
    "en-IN": "IN",
    "en-UK": "UK",
    "hi-IN": "HI",
    "bn-IN": "BN",
}

FALCON_VOICE_CONFIGS: Dict[str, FalconVoiceConfig] = {}


def _register(
    language: str,
    voice_id: str,
    style: str,
    locale: Optional[str] = None,
) -> str:
    cfg = FalconVoiceConfig(language, voice_id, style, locale)
    FALCON_VOICE_CONFIGS[cfg.registry_id] = cfg
    return cfg.registry_id


def _ids(language: str, specs: Sequence[VoiceSpec]) -> List[str]:
    out: List[str] = []
    for spec in specs:
        if len(spec) == 3:
            voice_id, style, locale = spec
            out.append(_register(language, voice_id, style, locale))
        else:
            voice_id, style = spec
            out.append(_register(language, voice_id, style))
    return out


def _pool(
    language: str,
    male: Sequence[VoiceSpec],
    female: Sequence[VoiceSpec],
) -> VoicePool:
    return {"male": _ids(language, male), "female": _ids(language, female)}


# --- Add voices here (shared by falcon_dev and falcon_prod) -------------------
# Each spec is (voice_id, style) or (voice_id, style, multiNativeLocale).
FALCON_BATTLE_VOICES: Dict[str, VoicePool] = {
    "en-US": _pool(
        "en-US",
        male=[
            ("Joshua", "Conversational", "en-US"),
            ("Bertie", "Conversational", "en-US"),
            ("Gordon", "Conversational"),
            ("Carlos", "Conversational", "en-US"),
        ],
        female=[
            ("Nimisha", "Conversational", "en-US"),
            ("Heidi", "Conversational", "en-US"),
            ("Madison", "Conversational"),
            ("Abirami", "Conversational", "en-US"),
        ],
    ),
    "en-UK": _pool(
        "en-UK",
        male=[
            ("Joshua", "Conversational"),
            ("Bertie", "Conversational"),
            ("Benedict", "Conversational"),
            ("Freddie", "Conversational"),
        ],
        female=[
            ("Lydia", "Conversational"),
            ("Ruby", "Conversational"),
            ("Sharon", "Conversational"),
            ("Heidi", "Conversational"),
        ],
    ),
    "en-IN": _pool(
        "en-IN",
        male=[
            ("Abhinav", "Conversational"),
            ("Nikhil", "Conversational"),
            ("Samar", "Conversational"),
            ("Aarav", "Conversational"),
        ],
        female=[
            ("Anisha", "Conversation"),
            ("Pooja", "Conversational"),
            ("Anusha", "Conversational"),
            ("Arohi", "Conversational"),
        ],
    ),
    "hi-IN": _pool(
        "hi-IN",
        male=[
            ("Karthikeyan", "Conversation", "hi-IN"),
            ("Abhinav", "Conversational", "hi-IN"),
            ("Hardik", "Conversational", "hi-IN"),
            ("Madhavan", "Conversational", "hi-IN"),
        ],
        female=[
            ("Ayushi", "Conversation"),
            ("Namrita", "Conversational"),
            ("Alia", "Conversational", "hi-IN"),
            ("Pooja", "Conversational", "hi-IN"),
        ],
    ),
    "bn-IN": _pool(
        "bn-IN",
        male=[
            ("Subhankar", "Conversational"),
            ("Abhik", "Conversational"),
            ("Arnab", "Conversational"),
        ],
        female=[
            ("Debarati", "Conversational"),
            ("Anisha", "Conversational", "bn-IN"),
            ("Ishani", "Conversational"),
        ],
    ),
}

FALCON_PROVIDER_IDS = frozenset({"falcon_dev", "falcon_prod"})


def get_falcon_voice_config(registry_id: str) -> Optional[FalconVoiceConfig]:
    return FALCON_VOICE_CONFIGS.get(registry_id)


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
