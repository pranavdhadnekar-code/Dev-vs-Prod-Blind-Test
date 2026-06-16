"""Per-provider language voice pools for the Voice Arena (2 male + 2 female each)."""
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
    "hi-IN": "Hindi (India)",
    "bn-IN": "Bengali (India)",
    "ta-IN": "Tamil (India)",
    "fr-FR": "French (France)",
    "es-ES": "Spanish (Spain)",
    "mr-IN": "Marathi (India)",
    "ml-IN": "Malayalam (India)",
}

# Multilingual competitors (English voice pool reused; models read other scripts).
_EL_M = ("Liam", "Dan")
_EL_F = ("Laura", "Jessica")
_DG_M = ("aura-2-apollo-en", "aura-2-orion-en")
_DG_F = ("aura-2-thalia-en", "aura-2-helena-en")
# Deepgram Aura-2 native voice ids per language (es = Peninsular/Spain,
# fr = France). French currently ships a single voice per gender.
_DEEPGRAM: Dict[str, VoicePool] = {
    "en-US": v(*_DG_M, *_DG_F),
    "en-IN": v(*_DG_M, *_DG_F),
    "en-UK": v(*_DG_M, *_DG_F),
    "es-ES": v("aura-2-nestor-es", "aura-2-alvaro-es", "aura-2-carina-es", "aura-2-diana-es"),
    "fr-FR": {"male": ["aura-2-hector-fr"], "female": ["aura-2-agathe-fr"]},
}
_CT_M = ("Professional Man", "Classy British Man")
_CT_F = ("Conversational Lady", "British Lady")
_OAI_M = ("onyx", "echo")
_OAI_F = ("nova", "shimmer")

_MURF_PROD: Dict[str, VoicePool] = {
    "en-US": v("en-US-terrell", "en-US-ronnie", "en-US-natalie", "en-US-samantha"),
    "en-IN": v("en-IN-aarav", "en-IN-eashwar", "en-IN-alia", "en-IN-isha"),
    "en-UK": v("en-UK-aiden", "en-UK-theo", "en-UK-heidi", "en-UK-amber"),
    "hi-IN": v("hi-IN-shaan", "hi-IN-rahul", "hi-IN-shweta", "hi-IN-ayushi"),
    "bn-IN": v("bn-IN-abhik", "bn-IN-arnab", "bn-IN-anwesha", "bn-IN-ishani"),
    "ta-IN": v("ta-IN-suresh", "ta-IN-sarvesh", "ta-IN-iniya", "ta-IN-abirami"),
    "fr-FR": v("fr-FR-maxime", "fr-FR-axel", "fr-FR-justine", "fr-FR-louise"),
    "es-ES": v("es-ES-enrique", "es-MX-alejandro", "es-ES-carla", "es-ES-carmen"),
    "mr-IN": v("mr-IN-abhishek", "mr-IN-harshad", "mr-IN-prajakta", "mr-IN-rujuta"),
    "ml-IN": v("ml-IN-vishnu", "ml-IN-madhavan", "ml-IN-sreelakshmi", "ml-IN-nimisha"),
}

_MURF_DEV_FALCON: Dict[str, VoicePool] = {
    "en-US": v("en-US-wyatt", "en-US-grant", "en-US-isabelle", "en-US-cc-033-f"),
    "en-IN": v("en-IN-samar", "en-IN-cc-050-m", "en-IN-cc-006-f", "en-IN-anisha"),
    "en-UK": v("en-UK-reuben", "en-UK-leonard", "en-UK-cc-035-f", "en-UK-lucy"),
    "hi-IN": v("hi-IN-cc-028-m", "hi-IN-cc-026-m", "hi-IN-khyati", "hi-IN-cc-059-f"),
    "bn-IN": v("bn-IN-anirban", "bn-IN-debashis", "bn-IN-debarati", "bn-IN-shreyasi"),
    "ta-IN": v("ta-IN-velmurugan", "ta-IN-santhosh", "ta-IN-janaki", "ta-IN-gayathri"),
    "fr-FR": v("fr-FR-maxime", "fr-FR-axel", "fr-FR-justine", "fr-FR-louise"),
    "es-ES": v("es-ES-enrique", "es-MX-alejandro", "es-ES-carla", "es-ES-carmen"),
    "mr-IN": v("mr-IN-abhishek", "mr-IN-harshad", "mr-IN-prajakta", "mr-IN-rujuta"),
    "ml-IN": v("ml-IN-vishnu", "ml-IN-madhavan", "ml-IN-sreelakshmi", "ml-IN-nimisha"),
}

_GOOGLE: Dict[str, VoicePool] = {
    "en-US": v("en-US-Neural2-D", "en-US-Neural2-J", "en-US-Neural2-F", "en-US-Neural2-H"),
    "en-IN": v("en-IN-Neural2-B", "en-IN-Neural2-C", "en-IN-Neural2-A", "en-IN-Neural2-D"),
    "en-UK": v("en-GB-Neural2-B", "en-GB-Neural2-D", "en-GB-Neural2-A", "en-GB-Neural2-C"),
    "hi-IN": v("hi-IN-Neural2-B", "hi-IN-Neural2-C", "hi-IN-Neural2-A", "hi-IN-Neural2-D"),
    "bn-IN": v("bn-IN-Standard-B", "bn-IN-Standard-C", "bn-IN-Standard-A", "bn-IN-Standard-D"),
    "ta-IN": v("ta-IN-Standard-D", "ta-IN-Standard-E", "ta-IN-Standard-C", "ta-IN-Standard-A"),
    "fr-FR": v("fr-FR-Neural2-D", "fr-FR-Neural2-B", "fr-FR-Neural2-C", "fr-FR-Neural2-E"),
    "es-ES": v("es-ES-Neural2-G", "es-ES-Neural2-F", "es-ES-Neural2-E", "es-ES-Neural2-H"),
    "mr-IN": v("mr-IN-Wavenet-B", "mr-IN-Wavenet-A", "mr-IN-Wavenet-A", "mr-IN-Wavenet-C"),
    "ml-IN": v("ml-IN-Wavenet-B", "ml-IN-Wavenet-C", "ml-IN-Wavenet-A", "ml-IN-Wavenet-D"),
}

# Fix mr-IN duplicate in google - use distinct voices
_GOOGLE["mr-IN"] = v("mr-IN-Wavenet-B", "mr-IN-Wavenet-C", "mr-IN-Wavenet-A", "mr-IN-Standard-A")

_AZURE: Dict[str, VoicePool] = {
    "en-US": v("en-US-GuyNeural", "en-US-DavisNeural", "en-US-JennyNeural", "en-US-AriaNeural"),
    "en-IN": v("en-IN-PrabhatNeural", "en-IN-AaravNeural", "en-IN-NeerjaNeural", "en-IN-AnanyaNeural"),
    "en-UK": v("en-GB-RyanNeural", "en-GB-ThomasNeural", "en-GB-SoniaNeural", "en-GB-LibbyNeural"),
    "hi-IN": v("hi-IN-MadhurNeural", "hi-IN-AaravNeural", "hi-IN-SwaraNeural", "hi-IN-AnanyaNeural"),
    "bn-IN": v("bn-IN-BashkarNeural", "bn-IN-SamirNeural", "bn-IN-TanishaaNeural", "bn-IN-NabanitaNeural"),
    "ta-IN": v("ta-IN-ValluvarNeural", "ta-IN-SuryaNeural", "ta-IN-PallaviNeural", "ta-IN-SnehaNeural"),
    "fr-FR": v("fr-FR-HenriNeural", "fr-FR-AlainNeural", "fr-FR-DeniseNeural", "fr-FR-EloiseNeural"),
    "es-ES": v("es-ES-AlvaroNeural", "es-ES-ArnauNeural", "es-ES-ElviraNeural", "es-ES-AbrilNeural"),
    "mr-IN": v("mr-IN-ManoharNeural", "mr-IN-AjayNeural", "mr-IN-AarohiNeural", "mr-IN-SnehaNeural"),
    "ml-IN": v("ml-IN-MidhunNeural", "ml-IN-SobhanaNeural", "ml-IN-SobhanaNeural", "ml-IN-MidhunNeural"),
}

_AZURE["ml-IN"] = v("ml-IN-MidhunNeural", "ml-IN-ArjunNeural", "ml-IN-SobhanaNeural", "ml-IN-MayaNeural")

_POLLY: Dict[str, VoicePool] = {
    "en-US": v("Matthew", "Stephen", "Joanna", "Ruth"),
    "en-IN": v("Matthew", "Stephen", "Joanna", "Ruth"),
    "en-UK": v("Brian", "Arthur", "Amy", "Emma"),
    "hi-IN": v("Matthew", "Stephen", "Aditi", "Kajal"),
    "bn-IN": v("Matthew", "Stephen", "Aditi", "Kajal"),
    "ta-IN": v("Matthew", "Stephen", "Aditi", "Kajal"),
    "fr-FR": v("Mathieu", "Remi", "Celine", "Lea"),
    "es-ES": v("Enrique", "Sergio", "Lucia", "Mia"),
    "mr-IN": v("Matthew", "Stephen", "Aditi", "Kajal"),
    "ml-IN": v("Matthew", "Stephen", "Aditi", "Kajal"),
}

_SARVAM: Dict[str, VoicePool] = {
    "en-IN": v("en-IN-male", "en-IN-male-2", "en-IN-female", "en-IN-female-2"),
    "hi-IN": v("hi-IN-male", "hi-IN-male-2", "hi-IN-female", "hi-IN-female-2"),
    "bn-IN": v("bn-IN-male", "bn-IN-male-2", "bn-IN-female", "bn-IN-female-2"),
    "ta-IN": v("ta-IN-male", "ta-IN-male-2", "ta-IN-female", "ta-IN-female-2"),
    "mr-IN": v("mr-IN-male", "mr-IN-male-2", "mr-IN-female", "mr-IN-female-2"),
    "ml-IN": v("ml-IN-male", "ml-IN-male-2", "ml-IN-female", "ml-IN-female-2"),
    "en-US": v("en-IN-male", "en-IN-male-2", "en-IN-female", "en-IN-female-2"),
    "en-UK": v("en-IN-male", "en-IN-male-2", "en-IN-female", "en-IN-female-2"),
    "fr-FR": v("en-IN-male", "en-IN-male-2", "en-IN-female", "en-IN-female-2"),
    "es-ES": v("en-IN-male", "en-IN-male-2", "en-IN-female", "en-IN-female-2"),
}

def _multilingual(pool: VoicePool) -> Dict[str, VoicePool]:
    return {lang: pool for lang in ARENA_LANGUAGES}


_ALL_LANGS = set(ARENA_LANGUAGES)

# Genuine per-provider language support for the arena's 10 languages.
# Two cases drive this:
#   * Voice-agnostic models (ElevenLabs/Cartesia/OpenAI) auto-detect language
#     from text, so coverage = the MODEL's supported languages.
#   * Voice-locked providers (Deepgram/Google/Azure/Polly/Sarvam/Murf) can only
#     do a language if we have a NATIVE voice id for it; reused English/Hindi
#     voices reading another script are excluded.
# Sources (Jun 2026): ElevenLabs Flash v2.5 (32 langs), Deepgram Aura-2
# (en/es/de/fr/nl/it/ja — en/fr/es voice ids configured here), Cartesia
# Sonic 3.5 (42 langs incl. all Indian languages), OpenAI gpt-4o-mini-tts
# (Whisper language set), Sarvam Bulbul v3 (Indian languages + Indian English).
PROVIDER_SUPPORTED_LANGUAGES: Dict[str, set] = {
    "omni_tts": set(_ALL_LANGS),            # Murf native voices for all
    "murf_gen2": set(_ALL_LANGS),           # Murf native voices for all
    "elevenlabs_v3": {"en-US", "en-IN", "en-UK", "hi-IN", "ta-IN", "fr-FR", "es-ES"},
    "deepgram_aura2": {"en-US", "en-IN", "en-UK", "fr-FR", "es-ES"},  # native en/fr/es voices
    "cartesia_sonic3": set(_ALL_LANGS),     # Sonic 3.5 covers all 10
    "openai": set(_ALL_LANGS),              # multilingual, voice-agnostic
    "sarvam_bulbul_v3": {"en-IN", "hi-IN", "bn-IN", "ta-IN", "mr-IN", "ml-IN"},
    "google_tts": set(_ALL_LANGS),          # native voices per language
    "azure_tts": set(_ALL_LANGS),           # native voices per language
    "amazon_polly": {"en-US", "en-IN", "en-UK", "hi-IN", "fr-FR", "es-ES"},  # no native bn/ta/mr/ml
}


def build_provider_languages(omni_dev: bool) -> Dict[str, Dict[str, VoicePool]]:
    murf_omni = _MURF_DEV_FALCON if omni_dev else _MURF_PROD
    el = v(*_EL_M, *_EL_F)
    ct = v(*_CT_M, *_CT_F)
    oai = v(*_OAI_M, *_OAI_F)
    raw = {
        "omni_tts": dict(murf_omni),
        "murf_gen2": dict(_MURF_PROD),
        "elevenlabs_v3": _multilingual(el),
        "deepgram_aura2": dict(_DEEPGRAM),
        "cartesia_sonic3": _multilingual(ct),
        "openai": _multilingual(oai),
        "sarvam_bulbul_v3": dict(_SARVAM),
        "google_tts": dict(_GOOGLE),
        "azure_tts": dict(_AZURE),
        "amazon_polly": dict(_POLLY),
    }
    # Restrict each provider to genuinely supported languages.
    out: Dict[str, Dict[str, VoicePool]] = {}
    for pid, pools in raw.items():
        allowed = PROVIDER_SUPPORTED_LANGUAGES.get(pid, _ALL_LANGS)
        out[pid] = {lang: pool for lang, pool in pools.items() if lang in allowed}
    return out


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


MURF_VOICE_GENDERS = voice_gender_lookup(_MURF_PROD, _MURF_DEV_FALCON)
