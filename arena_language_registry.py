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

# Per-language ElevenLabs voice ids (Flash 2.5).
_ELEVENLABS: Dict[str, VoicePool] = {
    "en-US": v("S9GPGBaMND8XWwwzxQXp", "bfGb7JTLUnZebZRiFYyq", "r1KmysJdVYZjJCm4mL3b", "hA4zGnmTwX2NQiTRMt7o"),
    "en-IN": v("37frHvUllvzviJDpT2Qa", "u7bRcYbD7visSINTyAT8", "ecp3DWciuUyW7BYM7II1", "6qL48o1LBmtR94hIYAQh"),
    "en-UK": v("auq43ws1oslv0tO4BDa7", "lUTamkMw7gOzZbFIwmq4", "4CrZuIW9am7gYAxgo2Af", "Gv42yFG3G6CHLsU5y8g6"),
    "hi-IN": v("3AMU7jXQuQa3oRvRqUmb", "iWNf11sz1GrUE4ppxTOL", "JS6C6yu2x9Byh4i1a8lX", "Ek86tj0PS0XTYchY9Ody"),
    "ta-IN": v("ConvFtidCOyEp2P5a1nK", "yIFUVClxedWzoMYhk15k", "mGboHvCVOXWYeFL8KTR0", "Nda4CxqYPMJ65wadFnhJ"),
    "fr-FR": v("NyxenPOqNyllHIzSoPbJ", "IbbR6Av0dWuQJS0b8JVT", "OhWejZm6c7D8CIm5epRM", "lvQdCgwZfBuOzxyV5pxu"),
    "es-ES": v("ByVRQtaK1WDOvTmP1PKO", "LlZr3QuzbW4WrPjgATHG", "dNjJKg63Fr5AXwIdkATa", "ewn5JTa3lNPY8QVuZJi6"),
}
_DG_M = ("aura-2-apollo-en", "aura-2-orion-en")
_DG_F = ("aura-2-thalia-en", "aura-2-helena-en")
# Deepgram Aura-2 native voice ids per language (es = Peninsular/Spain,
# fr = France). French currently ships a single voice per gender.
_DEEPGRAM: Dict[str, VoicePool] = {
    "en-US": v(*_DG_M, *_DG_F),
    "en-UK": {"male": ["aura-2-draco-en"], "female": ["aura-2-pandora-en"]},
    "es-ES": v("aura-2-nestor-es", "aura-2-alvaro-es", "aura-2-carina-es", "aura-2-diana-es"),
    "fr-FR": {"male": ["aura-2-hector-fr"], "female": ["aura-2-agathe-fr"]},
}
# Per-language Cartesia Sonic voice ids (UUID).
_CARTESIA: Dict[str, VoicePool] = {
    "en-US": {
        "male": [
            "630ed21c-2c5c-41cf-9d82-10a7fd668370",
            "47c38ca4-5f35-497b-b1a3-415245fb35e1",
            "f786b574-daa5-4673-aa0c-cbe3e8534c02",
        ],
        "female": ["db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"],
    },
    "en-IN": {
        "male": [
            "638efaaa-4d0c-442e-b701-3fae16aad012",
            "1259b7e3-cb8a-43df-9446-30971a46b8b0",
        ],
        "female": [
            "3b554273-4299-48b9-9aaf-eefd438e3941",
            "7ea5e9c2-b719-4dc3-b870-5ba5f14d31d8",
        ],
    },
    "en-UK": {
        "male": [
            "ef191366-f52f-447a-a398-ed8c0f2943a1",
            "4bc3cb8c-adb9-4bb8-b5d5-cbbef950b991",
        ],
        "female": [
            "62ae83ad-4f6a-430b-af41-a9bede9286ca",
            "2f251ac3-89a9-4a77-a452-704b474ccd01",
        ],
    },
    "hi-IN": {
        "male": [
            "4877b818-c7fe-4c89-b1cf-eadf8e23da72",
            "098fb15d-2597-4186-8b74-25340050b6e7",
        ],
        "female": [
            "faf0731e-dfb9-4cfc-8119-259a79b27e12",
            "95d51f79-c397-46f9-b49a-23763d3eaa2d",
        ],
    },
    "bn-IN": {
        "male": ["2ba861ea-7cdc-43d1-8608-4045b5a41de5"],
        "female": ["59ba7dee-8f9a-432f-a6c0-ffb33666b654"],
    },
    "ta-IN": {
        "male": [],
        "female": [
            "80e4e2b3-ec54-4930-97ac-667eba950352",
            "4014f0c9-d3eb-4eca-af2b-fd6004f526be",
        ],
    },
    "fr-FR": {
        "male": [
            "0418348a-0ca2-4e90-9986-800fb8b3bbc0",
            "7345dfa5-ee04-44d2-abf4-29262b880ab4",
        ],
        "female": [
            "7c58f4a4-a72c-42fa-a503-41b9408820f3",
            "faa75703-00e3-4a57-9955-0703001e3231",
        ],
    },
    "es-ES": {
        "male": [
            "13ff5deb-2591-42ad-a356-63a04e524411",
            "02aeee94-c02b-456e-be7a-659672acf82d",
        ],
        "female": [
            "9d8c6b2e-0a23-4a15-ae1b-121d5b5af417",
            "538a8872-3799-4df5-b373-b78493b766c6",
        ],
    },
    "mr-IN": {
        "male": ["f227bc18-3704-47fe-b759-8c78a450fdfa"],
        "female": ["5c32dce6-936a-4892-b131-bafe474afe5f"],
    },
    "ml-IN": {
        "male": ["374b80da-e622-4dfc-90f6-1eeb13d331c9"],
        "female": ["b426013c-002b-4e89-8874-8cd20b68373a"],
    },
}
_OAI_M = ("onyx", "echo")
_OAI_F = ("nova", "shimmer")

_MURF_PROD: Dict[str, VoicePool] = {
    "en-US": v("en-US-terrell", "en-US-ronnie", "en-US-natalie", "en-US-samantha"),
    "en-IN": v("en-IN-aarav", "en-IN-eashwar", "en-IN-alia", "en-IN-isha"),
    "en-UK": v("en-UK-aiden", "en-UK-theo", "en-UK-heidi", "en-UK-amber"),
    "hi-IN": v("hi-IN-shaan", "hi-IN-rahul", "hi-IN-shweta", "hi-IN-ayushi"),
    "bn-IN": {"male": ["bn-IN-abhik"], "female": ["bn-IN-anwesha", "bn-IN-ishani"]},
    "ta-IN": {"male": ["ta-IN-sarvesh"], "female": ["ta-IN-iniya", "ta-IN-abirami"]},
    "fr-FR": v("fr-FR-maxime", "fr-FR-axel", "fr-FR-justine", "fr-FR-louise"),
    "es-ES": v("es-ES-enrique", "es-ES-javier", "es-ES-carla", "es-ES-carmen"),
}

_MURF_DEV_FALCON: Dict[str, VoicePool] = {
    "en-US": v("en-US-will", "en-US-gordon", "en-US-olivia", "en-US-luna"),
    "en-IN": v("en-IN-samar", "en-IN-nikhil", "en-IN-anusha", "en-IN-anisha"),
    "en-UK": {"male": ["en-UK-benedict", "en-UK-jake"], "female": ["en-UK-lydia"]},
    "hi-IN": v("hi-IN-aman", "hi-IN-karan", "hi-IN-namrita", "hi-IN-khyati"),
    "bn-IN": {"male": ["bn-IN-subhankar"], "female": ["bn-IN-debarati"]},
    "ta-IN": {"male": ["ta-IN-romesh"], "female": ["ta-IN-latika"]},
    "fr-FR": {"male": ["fr-FR-guillaume", "fr-FR-axel"], "female": ["fr-FR-justine"]},
    "es-ES": {"male": ["es-ES-javier"], "female": ["es-ES-carmen"]},
    "mr-IN": {"male": ["mr-IN-prathamesh", "mr-IN-vaibhav"], "female": ["mr-IN-prajakta"]},
    "ml-IN": {"male": ["ml-IN-madhavan"], "female": ["ml-IN-nimisha"]},
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
    # Marathi / Malayalam: Azure only ships one neural voice per gender (Jun 2026).
    "mr-IN": {"male": ["mr-IN-ManoharNeural"], "female": ["mr-IN-AarohiNeural"]},
    "ml-IN": {"male": ["ml-IN-MidhunNeural"], "female": ["ml-IN-SobhanaNeural"]},
}

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
    "deepgram_aura2": {"en-US", "en-UK", "fr-FR", "es-ES"},  # native en/fr/es voices
    "cartesia_sonic3": set(_ALL_LANGS),     # Sonic 3.5 covers all 10
    "openai": set(_ALL_LANGS),              # multilingual, voice-agnostic
    "sarvam_bulbul_v3": {"en-IN", "hi-IN", "bn-IN", "ta-IN", "mr-IN", "ml-IN"},
    "google_tts": set(_ALL_LANGS),          # native voices per language
    "azure_tts": set(_ALL_LANGS),           # native voices per language
    "amazon_polly": {"en-US", "en-IN", "en-UK", "hi-IN", "fr-FR", "es-ES"},  # no native bn/ta/mr/ml
}


def build_provider_languages(omni_dev: bool = True) -> Dict[str, Dict[str, VoicePool]]:
    # Falcon 2 (omni_tts) always uses its own voice catalog, independent of Murf Gen2.
    murf_omni = dict(_MURF_DEV_FALCON)
    oai = v(*_OAI_M, *_OAI_F)
    raw = {
        "omni_tts": dict(murf_omni),
        "murf_gen2": dict(_MURF_PROD),
        "elevenlabs_v3": dict(_ELEVENLABS),
        "deepgram_aura2": dict(_DEEPGRAM),
        "cartesia_sonic3": dict(_CARTESIA),
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
