"""
Configuration settings for the TTS Benchmarking Tool
"""
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from arena_language_registry import (
    ARENA_LANGUAGES,
    MURF_VOICE_GENDERS,
    _MURF_DEV_FALCON,
    _MURF_PROD,
    build_provider_languages,
    flatten_voice_ids,
)

@dataclass
class VoiceInfo:
    """Voice metadata with gender information"""
    id: str
    name: str
    gender: str  # "male" or "female"
    accent: str = "US"  # e.g. US, UK, IN, HI, BN, TA

@dataclass
class TTSConfig:
    """Configuration for TTS providers"""
    name: str
    api_key_env: str
    base_url: str
    supported_voices: List[str]
    max_chars: int
    supports_streaming: bool
    model_name: str = ""  # Full model name for display
    voice_info: Dict[str, VoiceInfo] = field(default_factory=dict)  # Voice metadata with gender


def get_omni_tts_url() -> str:
    """POST target for NewModel TTS (override with OMNI_BASE_URL or OMNI_HOST)."""
    u = (os.getenv("OMNI_BASE_URL") or "").strip().rstrip("/")
    if u:
        return u
    host = (os.getenv("OMNI_HOST") or "").strip()
    if not host:
        return ""
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}/tts"


def get_falcon_api_url() -> str:
    """Murf Falcon 2 (FALCON) streaming endpoint for the arena anchor."""
    return (
        _clean_env("MURF_FALCON_URL")
        or "https://api.murf.ai/v1/speech/stream"
    )


def is_falcon_dev_stream() -> bool:
    """True when Falcon 2 is pointed at the Murf dev stream endpoint."""
    return "dev.murf.ai" in get_falcon_api_url()


def _clean_env(name: str) -> str:
    return (os.getenv(name) or "").strip().strip('"').strip("'")


def is_jwt_token(key: str) -> bool:
    """True when key looks like a three-part JWT (dev stream auth)."""
    parts = key.split(".")
    return len(parts) == 3 and all(parts)


def is_murf_api_key(key: str) -> bool:
    return key.startswith("ap2_")


def falcon_auth_headers(api_key: str) -> Dict[str, str]:
    """Murf FALCON accepts api-key (ap2_…) or token (JWT) — not legacy Omni host keys."""
    if is_jwt_token(api_key):
        return {"token": api_key}
    return {"api-key": api_key}


def get_omni_falcon_api_key() -> str:
    """API key for Falcon 2 anchor (OMNI_API_KEY env var).

    Must be a Murf ``ap2_`` key or a JWT for dev stream endpoints. Legacy Omni /
    NewModel host keys are not accepted by the Murf speech API.
    """
    omni = _clean_env("OMNI_API_KEY")
    if omni and (is_murf_api_key(omni) or is_jwt_token(omni)):
        return omni
    murf = _clean_env("MURF_API_KEY")
    if murf:
        return murf
    if omni:
        return omni
    raise ValueError(
        "Set OMNI_API_KEY to a Murf API key (ap2_…) or JWT for Falcon 2. "
        "Legacy Omni host keys do not work with the Murf FALCON endpoint."
    )


# --- Single voice catalog (Murf voiceId is the key) ---------------------
# en-US + en-IN + en-UK: Murf Gen2 English catalogs. hi-IN: "Hindi - India". bn-IN / ta-IN as below.
# kn-IN / mr-IN / te-IN native rows from partner sheets are omitted: Gen2 library tables show those locales via
# multilingual EN voices, not primary kn-IN-*/mr-IN-*/te-IN-* IDs (avoids GEN2 400). en-US-clint omitted by request.
_VOICE_CATALOG = [
    # en-US (Gen2 US English catalog)
    VoiceInfo("en-US-terrell",   "Terrell",   "male",   "US"),
    VoiceInfo("en-US-natalie",   "Natalie",   "female", "US"),
    VoiceInfo("en-US-charles",   "Charles",   "male",   "US"),
    VoiceInfo("en-US-samantha",  "Samantha",  "female", "US"),
    VoiceInfo("en-US-alicia",    "Alicia",    "female", "US"),
    VoiceInfo("en-US-ronnie",    "Ronnie",    "male",   "US"),
    VoiceInfo("en-US-cooper",    "Cooper",    "male",   "US"),
    VoiceInfo("en-US-michelle",  "Michelle",  "female", "US"),
    VoiceInfo("en-US-miles",     "Miles",     "male",   "US"),
    VoiceInfo("en-US-marcus",    "Marcus",    "male",   "US"),
    VoiceInfo("en-US-lucas",     "Lucas",     "male",   "US"),
    VoiceInfo("en-US-ken",       "Ken",       "male",   "US"),
    VoiceInfo("en-US-daisy",     "Daisy",     "female", "US"),
    VoiceInfo("en-US-edmund",    "Edmund",    "male",   "US"),
    VoiceInfo("en-US-wayne",     "Wayne",     "male",   "US"),
    VoiceInfo("en-US-iris",      "Iris",      "female", "US"),
    VoiceInfo("en-US-ryan",      "Ryan",      "male",   "US"),
    VoiceInfo("en-US-claire",    "Claire",    "female", "US"),
    VoiceInfo("en-US-naomi",     "Naomi",     "female", "US"),
    VoiceInfo("en-US-charlotte", "Charlotte", "female", "US"),
    VoiceInfo("en-US-dylan",     "Dylan",     "male",   "US"),
    VoiceInfo("en-US-julia",     "Julia",     "female", "US"),
    VoiceInfo("en-US-carter",    "Carter",    "male",   "US"),
    VoiceInfo("en-US-daniel",    "Daniel",    "male",   "US"),
    VoiceInfo("en-US-june",      "June",      "female", "US"),
    VoiceInfo("en-US-amara",     "Amara",     "female", "US"),
    VoiceInfo("en-US-river",     "River",     "male",   "US"),
    VoiceInfo("en-US-evander",   "Evander",   "male",   "US"),
    VoiceInfo("en-US-caleb",     "Caleb",     "male",   "US"),
    VoiceInfo("en-US-josie",     "Josie",     "female", "US"),
    VoiceInfo("en-US-molly",     "Molly",     "female", "US"),
    VoiceInfo("en-US-delilah",   "Delilah",   "female", "US"),
    VoiceInfo("en-US-imani",     "Imani",     "female", "US"),
    VoiceInfo("en-US-jayden",    "Jayden",    "male",   "US"),
    VoiceInfo("en-US-denzel",    "Denzel",    "male",   "US"),
    VoiceInfo("en-US-angela",    "Angela",    "female", "US"),
    VoiceInfo("en-US-phoebe",    "Phoebe",    "female", "US"),
    VoiceInfo("en-US-riley",     "Riley",     "female", "US"),
    VoiceInfo("en-US-abigail",   "Abigail",   "female", "US"),
    VoiceInfo("en-US-zion",      "Zion",      "male",   "US"),
    VoiceInfo("en-US-ariana",    "Ariana",    "female", "US"),
    VoiceInfo("en-US-paul",      "Paul",      "male",   "US"),
    VoiceInfo("en-US-maverick",  "Maverick",  "male",   "US"),
    # hi-IN
    VoiceInfo("hi-IN-shaan",     "Shaan",     "male",   "HI"),
    VoiceInfo("hi-IN-rahul",     "Rahul",     "male",   "HI"),
    VoiceInfo("hi-IN-shweta",    "Shweta",    "female", "HI"),
    VoiceInfo("hi-IN-ayushi",    "Ayushi",    "female", "HI"),
    VoiceInfo("hi-IN-amit",      "Amit",      "male",   "HI"),
    VoiceInfo("hi-IN-kabir",     "Kabir",     "male",   "HI"),
    # bn-IN (Gen2 Bangla - India)
    VoiceInfo("bn-IN-anwesha",   "Anwesha",   "female", "BN"),
    VoiceInfo("bn-IN-ishani",    "Ishani",    "female", "BN"),
    VoiceInfo("bn-IN-abhik",     "Abhik",     "male",   "BN"),
    VoiceInfo("bn-IN-arnab",     "Arnab",     "male",   "BN"),
    # ta-IN (Gen2 Tamil - India)
    VoiceInfo("ta-IN-iniya",     "Iniya",     "female", "TA"),
    VoiceInfo("ta-IN-suresh",    "Suresh",    "male",   "TA"),
    VoiceInfo("ta-IN-sarvesh",   "Sarvesh",   "male",   "TA"),
    VoiceInfo("ta-IN-abirami",   "Abirami",   "female", "TA"),
    # en-IN (English - India, Gen2)
    VoiceInfo("en-IN-alia",      "Alia",      "female", "IN"),
    VoiceInfo("en-IN-isha",      "Isha",      "female", "IN"),
    VoiceInfo("en-IN-aarav",     "Aarav",     "male",   "IN"),
    VoiceInfo("en-IN-eashwar",   "Eashwar",   "male",   "IN"),
    # en-UK (English - UK / British English, Gen2)
    VoiceInfo("en-UK-aiden",     "Aiden",     "male",   "UK"),
    VoiceInfo("en-UK-theo",      "Theo",      "male",   "UK"),
    VoiceInfo("en-UK-heidi",     "Heidi",     "female", "UK"),
    VoiceInfo("en-UK-amber",     "Amber",     "female", "UK"),
]


def _infer_murf_voice_info(voice_id: str) -> VoiceInfo:
    parts = voice_id.split("-")
    accent_map = {
        "US": "US", "UK": "UK", "IN": "IN", "HI": "HI", "BN": "BN", "TA": "TA",
        "FR": "FR", "ES": "ES", "MR": "MR", "ML": "ML", "PH": "TL", "MX": "ES",
    }
    accent = accent_map.get(parts[1], parts[1].upper()) if len(parts) >= 2 else "US"
    slug = parts[-1].split("[")[0].replace("cc-", "")
    name = slug[:1].upper() + slug[1:] if slug else voice_id
    gender = MURF_VOICE_GENDERS.get(voice_id, "male")
    return VoiceInfo(voice_id, name, gender, accent)


_murf_arena_ids = set(flatten_voice_ids(_MURF_PROD)) | set(flatten_voice_ids(_MURF_DEV_FALCON))
_catalog_ids = {v.id for v in _VOICE_CATALOG}
for _vid in sorted(_murf_arena_ids):
    if _vid not in _catalog_ids:
        _VOICE_CATALOG.append(_infer_murf_voice_info(_vid))
        _catalog_ids.add(_vid)

# Murf voiceId  ->  NewModel source_voice_id  (vendor mapping table)
MURF_TO_OMNI_VOICE: Dict[str, str] = {
    "en-US-terrell":   "en-US-terrell",
    "en-US-natalie":   "en-US-natalie",
    "en-US-charles":   "en-US-003",
    "en-US-samantha":  "en-US-samantha",
    "en-US-alicia":    "en-US-009-conversational",
    "en-US-ronnie":    "en-US-ronnie",
    "en-US-cooper":    "en-US-cooper",
    "en-US-michelle":  "en-US-michelle",
    "en-US-miles":     "en-US-miles",
    "en-US-marcus":    "en-US-marcus",
    "en-US-lucas":     "en-US-lucas",
    "en-US-ken":       "en-US-ken",
    "en-US-daisy":     "en-US-daisy",
    "en-US-edmund":    "en-US-edmund",
    "en-US-wayne":     "en-US-wayne",
    "en-US-iris":      "en-US-iris",
    "en-US-ryan":      "en-US-ryan",
    "en-US-claire":    "en-US-claire",
    "en-US-naomi":     "en-US-naomi",
    "en-US-charlotte": "en-US-022-conversational",
    "en-US-dylan":     "en-US-011-conversational",
    "en-US-julia":     "en-US-002",
    "en-US-carter":    "en-US-006-documentary",
    "en-US-daniel":    "en-US-012-conversational",
    "en-US-june":      "en-US-026-advertising",
    "en-US-amara":     "en-US-028-podcast",
    "en-US-river":     "en-US-029-podcast",
    "en-US-evander":   "en-US-024-conversational",
    "en-US-caleb":     "en-US-023-advertising",
    "en-US-josie":     "en-US-036-podcast",
    "en-US-molly":     "en-US-031-podcast",
    "en-US-delilah":   "en-US-032-podcast",
    "en-US-imani":     "en-US-034-podcast",
    "en-US-jayden":    "en-US-035-podcast",
    "en-US-denzel":    "en-US-033-podcast",
    "en-US-angela":    "en-US-038-promotional",
    "en-US-phoebe":    "en-US-043-conversational",
    "en-US-riley":     "en-US-040-promotional",
    "en-US-abigail":   "en-US-037-podcast",
    "en-US-zion":      "en-US-042-promotional",
    "en-US-ariana":    "en-US-041-podcast",
    "en-US-paul":      "en-US-045-audiobook",
    "en-US-maverick":  "en-US-048-M-Audiobook",
    "hi-IN-shaan":     "hi-IN-010-conversational",
    "hi-IN-rahul":     "hi-IN-013-conversational",
    "hi-IN-shweta":    "hi-IN-005-conversational",
    "hi-IN-ayushi":    "hi-IN-017-conversational",
    "hi-IN-amit":      "hi-IN-012-podcast",
    "hi-IN-kabir":     "hi-IN-016-conversational",
    "bn-IN-anwesha":   "bn-IN-001-conversational",
    "bn-IN-ishani":    "bn-IN-002-conversational",
    "bn-IN-abhik":     "bn-IN-003-conversational",
    "bn-IN-arnab":     "bn-IN-Pratik-004-M-Conversational",
    "ta-IN-iniya":     "ta-IN-001-podcast",
    "ta-IN-suresh":    "ta-IN-004-conversational",
    "ta-IN-sarvesh":   "ta-IN-003-conversational",
    "ta-IN-abirami":   "ta-IN-002-conversational",
    # en-IN
    "en-IN-alia":      "en-IN-014-documentary",
    "en-IN-isha":      "en-IN-005-conversational",
    "en-IN-aarav":     "en-IN-006-conversational",
    "en-IN-eashwar":   "en-IN-009-conversational",
    # en-UK
    "en-UK-aiden":     "en-UK-aiden",
    "en-UK-theo":      "en-UK-theo",
    "en-UK-heidi":     "en-UK-018-conversational",
    "en-UK-amber":     "en-UK-022-documentary",
}
for _vid in _murf_arena_ids:
    MURF_TO_OMNI_VOICE.setdefault(_vid, _vid)

_VOICE_INFO: Dict[str, VoiceInfo] = {v.id: v for v in _VOICE_CATALOG}
if set(MURF_TO_OMNI_VOICE.keys()) != set(_VOICE_INFO.keys()):
    _only_v = set(_VOICE_INFO.keys()) - set(MURF_TO_OMNI_VOICE.keys())
    _only_m = set(MURF_TO_OMNI_VOICE.keys()) - set(_VOICE_INFO.keys())
    raise ValueError(f"MURF_TO_OMNI_VOICE must match voice catalog keys. Only in catalog: {_only_v}. Only in map: {_only_m}.")
# Both providers share the same Murf voiceId catalog (so blind A/B uses identical voices).
_SHARED_VOICES: List[str] = list(_VOICE_INFO.keys())

_FALCON_DEV_VOICES: List[str] = flatten_voice_ids(_MURF_DEV_FALCON)


def get_falcon_supported_voices() -> List[str]:
    """Voice IDs accepted for Falcon 2 validation."""
    return list(_FALCON_DEV_VOICES)


def get_falcon_voice_info() -> Dict[str, VoiceInfo]:
    return {vid: _VOICE_INFO[vid] for vid in _FALCON_DEV_VOICES if vid in _VOICE_INFO}

def _voice_info_from(pairs: List[tuple], accent: str = "US") -> Dict[str, VoiceInfo]:
    """Build a {voice_id: VoiceInfo} map from (id, name, gender) tuples."""
    out: Dict[str, VoiceInfo] = {}
    for voice_id, name, gender in pairs:
        out[voice_id] = VoiceInfo(voice_id, name, gender, accent)
    return out


# --- Competitor voice catalogs (id, display name, gender) --------------------
_ELEVENLABS_VOICES = _voice_info_from([
    ("Laura", "Laura", "female"),
    ("Jessica", "Jessica", "female"),
    ("Elizabeth", "Elizabeth", "female"),
    ("Liam", "Liam", "male"),
    ("Jarnathan", "Jarnathan", "male"),
    ("Dan", "Dan", "male"),
    ("Nathaniel", "Nathaniel", "male"),
])

_DEEPGRAM_AURA1_VOICES = _voice_info_from([
    ("aura-asteria-en", "Asteria", "female"),
    ("aura-luna-en", "Luna", "female"),
    ("aura-stella-en", "Stella", "female"),
    ("aura-athena-en", "Athena", "female"),
    ("aura-hera-en", "Hera", "female"),
    ("aura-orion-en", "Orion", "male"),
    ("aura-arcas-en", "Arcas", "male"),
    ("aura-perseus-en", "Perseus", "male"),
    ("aura-angus-en", "Angus", "male"),
    ("aura-orpheus-en", "Orpheus", "male"),
    ("aura-helios-en", "Helios", "male"),
    ("aura-zeus-en", "Zeus", "male"),
])

_DEEPGRAM_AURA2_VOICES = _voice_info_from([
    ("aura-2-thalia-en", "Thalia", "female"),
    ("aura-2-andromeda-en", "Andromeda", "female"),
    ("aura-2-helena-en", "Helena", "female"),
    ("aura-2-hera-en", "Hera", "female"),
    ("aura-2-apollo-en", "Apollo", "male"),
    ("aura-2-arcas-en", "Arcas", "male"),
    ("aura-2-aries-en", "Aries", "male"),
    ("aura-2-orion-en", "Orion", "male"),
])

_OPENAI_VOICES = _voice_info_from([
    ("nova", "Nova", "female"),
    ("shimmer", "Shimmer", "female"),
    ("alloy", "Alloy", "female"),
    ("onyx", "Onyx", "male"),
    ("echo", "Echo", "male"),
    ("fable", "Fable", "male"),
])

_CARTESIA_VOICES = _voice_info_from([
    ("British Lady", "British Lady", "female"),
    ("Conversational Lady", "Conversational Lady", "female"),
    ("Midwestern Woman", "Midwestern Woman", "female"),
    ("Classy British Man", "Classy British Man", "male"),
    ("Friendly Reading Man", "Friendly Reading Man", "male"),
    ("Professional Man", "Professional Man", "male"),
    ("Newsman", "Newsman", "male"),
])

_SARVAM_VOICES = _voice_info_from([
    ("en-IN-female", "Female (English-India)", "female"),
    ("en-IN-female-2", "Female 2 (English-India)", "female"),
    ("hi-IN-female", "Female (Hindi-India)", "female"),
    ("hi-IN-female-2", "Female 2 (Hindi-India)", "female"),
    ("en-IN-male", "Male (English-India)", "male"),
    ("en-IN-male-2", "Male 2 (English-India)", "male"),
    ("hi-IN-male", "Male (Hindi-India)", "male"),
    ("hi-IN-male-2", "Male 2 (Hindi-India)", "male"),
    ("bn-IN-female", "Female (Bengali-India)", "female"),
    ("bn-IN-female-2", "Female 2 (Bengali-India)", "female"),
    ("bn-IN-male", "Male (Bengali-India)", "male"),
    ("bn-IN-male-2", "Male 2 (Bengali-India)", "male"),
    ("ta-IN-female", "Female (Tamil-India)", "female"),
    ("ta-IN-female-2", "Female 2 (Tamil-India)", "female"),
    ("ta-IN-male", "Male (Tamil-India)", "male"),
    ("ta-IN-male-2", "Male 2 (Tamil-India)", "male"),
    ("mr-IN-female", "Female (Marathi-India)", "female"),
    ("mr-IN-female-2", "Female 2 (Marathi-India)", "female"),
    ("mr-IN-male", "Male (Marathi-India)", "male"),
    ("mr-IN-male-2", "Male 2 (Marathi-India)", "male"),
    ("ml-IN-female", "Female (Malayalam-India)", "female"),
    ("ml-IN-female-2", "Female 2 (Malayalam-India)", "female"),
    ("ml-IN-male", "Male (Malayalam-India)", "male"),
    ("ml-IN-male-2", "Male 2 (Malayalam-India)", "male"),
], accent="IN")

# Google Cloud TTS: voice name encodes the BCP-47 languageCode (e.g. en-US-Neural2-D).
_GOOGLE_VOICES = _voice_info_from([
    ("en-US-Neural2-F", "Neural2-F (US)", "female"),
    ("en-US-Neural2-H", "Neural2-H (US)", "female"),
    ("en-US-Neural2-D", "Neural2-D (US)", "male"),
    ("en-US-Neural2-J", "Neural2-J (US)", "male"),
    ("en-IN-Neural2-A", "Neural2-A (IN)", "female"),
    ("en-IN-Neural2-D", "Neural2-D (IN)", "female"),
    ("en-IN-Neural2-B", "Neural2-B (IN)", "male"),
    ("en-IN-Neural2-C", "Neural2-C (IN)", "male"),
    ("en-GB-Neural2-A", "Neural2-A (GB)", "female"),
    ("en-GB-Neural2-C", "Neural2-C (GB)", "female"),
    ("en-GB-Neural2-B", "Neural2-B (GB)", "male"),
    ("en-GB-Neural2-D", "Neural2-D (GB)", "male"),
    ("hi-IN-Neural2-A", "Neural2-A (HI)", "female"),
    ("hi-IN-Neural2-D", "Neural2-D (HI)", "female"),
    ("hi-IN-Neural2-B", "Neural2-B (HI)", "male"),
    ("hi-IN-Neural2-C", "Neural2-C (HI)", "male"),
    ("bn-IN-Standard-A", "Standard-A (BN)", "female"),
    ("bn-IN-Standard-D", "Standard-D (BN)", "female"),
    ("bn-IN-Standard-B", "Standard-B (BN)", "male"),
    ("bn-IN-Standard-C", "Standard-C (BN)", "male"),
    ("ta-IN-Standard-C", "Standard-C (TA)", "female"),
    ("ta-IN-Standard-A", "Standard-A (TA)", "female"),
    ("ta-IN-Standard-D", "Standard-D (TA)", "male"),
    ("ta-IN-Standard-E", "Standard-E (TA)", "male"),
    ("fr-FR-Neural2-C", "Neural2-C (FR)", "female"),
    ("fr-FR-Neural2-E", "Neural2-E (FR)", "female"),
    ("fr-FR-Neural2-D", "Neural2-D (FR)", "male"),
    ("fr-FR-Neural2-B", "Neural2-B (FR)", "male"),
    ("es-ES-Neural2-E", "Neural2-E (ES)", "female"),
    ("es-ES-Neural2-H", "Neural2-H (ES)", "female"),
    ("es-ES-Neural2-G", "Neural2-G (ES)", "male"),
    ("es-ES-Neural2-F", "Neural2-F (ES)", "male"),
    ("mr-IN-Wavenet-A", "Wavenet-A (MR)", "female"),
    ("mr-IN-Standard-A", "Standard-A (MR)", "female"),
    ("mr-IN-Wavenet-B", "Wavenet-B (MR)", "male"),
    ("mr-IN-Wavenet-C", "Wavenet-C (MR)", "male"),
    ("ml-IN-Wavenet-A", "Wavenet-A (ML)", "female"),
    ("ml-IN-Wavenet-D", "Wavenet-D (ML)", "female"),
    ("ml-IN-Wavenet-B", "Wavenet-B (ML)", "male"),
    ("ml-IN-Wavenet-C", "Wavenet-C (ML)", "male"),
])

# Azure: voice name encodes the locale (e.g. en-US-JennyNeural).
_AZURE_VOICES = _voice_info_from([
    ("en-US-JennyNeural", "Jenny (US)", "female"),
    ("en-US-AriaNeural", "Aria (US)", "female"),
    ("en-US-GuyNeural", "Guy (US)", "male"),
    ("en-US-DavisNeural", "Davis (US)", "male"),
    ("en-IN-NeerjaNeural", "Neerja (IN)", "female"),
    ("en-IN-AnanyaNeural", "Ananya (IN)", "female"),
    ("en-IN-PrabhatNeural", "Prabhat (IN)", "male"),
    ("en-IN-AaravNeural", "Aarav (IN)", "male"),
    ("en-GB-SoniaNeural", "Sonia (GB)", "female"),
    ("en-GB-LibbyNeural", "Libby (GB)", "female"),
    ("en-GB-RyanNeural", "Ryan (GB)", "male"),
    ("en-GB-ThomasNeural", "Thomas (GB)", "male"),
    ("hi-IN-SwaraNeural", "Swara (HI)", "female"),
    ("hi-IN-AnanyaNeural", "Ananya (HI)", "female"),
    ("hi-IN-MadhurNeural", "Madhur (HI)", "male"),
    ("hi-IN-AaravNeural", "Aarav (HI)", "male"),
    ("bn-IN-TanishaaNeural", "Tanishaa (BN)", "female"),
    ("bn-IN-NabanitaNeural", "Nabanita (BN)", "female"),
    ("bn-IN-BashkarNeural", "Bashkar (BN)", "male"),
    ("bn-IN-SamirNeural", "Samir (BN)", "male"),
    ("ta-IN-PallaviNeural", "Pallavi (TA)", "female"),
    ("ta-IN-SnehaNeural", "Sneha (TA)", "female"),
    ("ta-IN-ValluvarNeural", "Valluvar (TA)", "male"),
    ("ta-IN-SuryaNeural", "Surya (TA)", "male"),
    ("fr-FR-DeniseNeural", "Denise (FR)", "female"),
    ("fr-FR-EloiseNeural", "Eloise (FR)", "female"),
    ("fr-FR-HenriNeural", "Henri (FR)", "male"),
    ("fr-FR-AlainNeural", "Alain (FR)", "male"),
    ("es-ES-ElviraNeural", "Elvira (ES)", "female"),
    ("es-ES-AbrilNeural", "Abril (ES)", "female"),
    ("es-ES-AlvaroNeural", "Alvaro (ES)", "male"),
    ("es-ES-ArnauNeural", "Arnau (ES)", "male"),
    ("mr-IN-AarohiNeural", "Aarohi (MR)", "female"),
    ("mr-IN-SnehaNeural", "Sneha (MR)", "female"),
    ("mr-IN-ManoharNeural", "Manohar (MR)", "male"),
    ("mr-IN-AjayNeural", "Ajay (MR)", "male"),
    ("ml-IN-SobhanaNeural", "Sobhana (ML)", "female"),
    ("ml-IN-MayaNeural", "Maya (ML)", "female"),
    ("ml-IN-MidhunNeural", "Midhun (ML)", "male"),
    ("ml-IN-ArjunNeural", "Arjun (ML)", "male"),
])

# Amazon Polly (neural engine). VoiceId implies the language.
_POLLY_VOICES = _voice_info_from([
    ("Joanna", "Joanna (US)", "female"),
    ("Ruth", "Ruth (US)", "female"),
    ("Matthew", "Matthew (US)", "male"),
    ("Stephen", "Stephen (US)", "male"),
    ("Amy", "Amy (GB)", "female"),
    ("Emma", "Emma (GB)", "female"),
    ("Brian", "Brian (GB)", "male"),
    ("Arthur", "Arthur (GB)", "male"),
    ("Aditi", "Aditi (IN)", "female"),
    ("Kajal", "Kajal (IN)", "female"),
    ("Celine", "Celine (FR)", "female"),
    ("Lea", "Lea (FR)", "female"),
    ("Mathieu", "Mathieu (FR)", "male"),
    ("Remi", "Remi (FR)", "male"),
    ("Lucia", "Lucia (ES)", "female"),
    ("Mia", "Mia (ES)", "female"),
    ("Enrique", "Enrique (ES)", "male"),
    ("Sergio", "Sergio (ES)", "male"),
])

_DEEPGRAM_URL = "https://api.deepgram.com/v1/speak"
_ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
_CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
_GOOGLE_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

TTS_PROVIDERS = {
    "omni_tts": TTSConfig(
        name="Falcon 2",
        api_key_env="OMNI_API_KEY",
        base_url=get_falcon_api_url(),
        supported_voices=get_falcon_supported_voices(),
        max_chars=5000,
        supports_streaming=True,
        model_name="FALCON",
        voice_info=get_falcon_voice_info(),
    ),
    "murf_gen2": TTSConfig(
        name="Murf Gen 2",
        api_key_env="MURF_API_KEY",
        base_url="https://api.murf.ai/v1/speech/stream",
        supported_voices=list(_SHARED_VOICES),
        max_chars=3000,
        supports_streaming=True,
        model_name="Gen2",
        voice_info=dict(_VOICE_INFO),
    ),
    "elevenlabs_v3": TTSConfig(
        name="ElevenLabs Flash 2.5",
        api_key_env="ELEVENLABS_API_KEY",
        base_url=_ELEVENLABS_URL,
        supported_voices=list(_ELEVENLABS_VOICES.keys()),
        max_chars=5000,
        supports_streaming=False,
        model_name="eleven_flash_v2_5",
        voice_info=dict(_ELEVENLABS_VOICES),
    ),
    "deepgram_aura2": TTSConfig(
        name="Deepgram Aura 2",
        api_key_env="DEEPGRAM_API_KEY",
        base_url=_DEEPGRAM_URL,
        supported_voices=list(_DEEPGRAM_AURA2_VOICES.keys()),
        max_chars=2000,
        supports_streaming=True,
        model_name="aura-2",
        voice_info=dict(_DEEPGRAM_AURA2_VOICES),
    ),
    "cartesia_sonic3": TTSConfig(
        name="Cartesia Sonic 3.5",
        api_key_env="CARTESIA_API_KEY",
        base_url=_CARTESIA_URL,
        supported_voices=list(_CARTESIA_VOICES.keys()),
        max_chars=3000,
        supports_streaming=True,
        model_name="sonic-3.5",
        voice_info=dict(_CARTESIA_VOICES),
    ),
    "openai": TTSConfig(
        name="OpenAI",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1/audio/speech",
        supported_voices=list(_OPENAI_VOICES.keys()),
        max_chars=4096,
        supports_streaming=False,
        model_name="gpt-4o-mini-tts",
        voice_info=dict(_OPENAI_VOICES),
    ),
    "sarvam_bulbul_v3": TTSConfig(
        name="Sarvam Bulbul v3",
        api_key_env="SARVAM_API_KEY",
        base_url="https://api.sarvam.ai/text-to-speech",
        supported_voices=list(_SARVAM_VOICES.keys()),
        max_chars=2000,
        supports_streaming=False,
        model_name="bulbul:v3",
        voice_info=dict(_SARVAM_VOICES),
    ),
    "google_tts": TTSConfig(
        name="Google Cloud TTS",
        api_key_env="GOOGLE_TTS_API_KEY",
        base_url=_GOOGLE_URL,
        supported_voices=list(_GOOGLE_VOICES.keys()),
        max_chars=5000,
        supports_streaming=False,
        model_name="neural2",
        voice_info=dict(_GOOGLE_VOICES),
    ),
    "azure_tts": TTSConfig(
        name="Azure TTS",
        api_key_env="AZURE_SPEECH_KEY",
        base_url="",  # region-dependent, built at request time from AZURE_SPEECH_REGION
        supported_voices=list(_AZURE_VOICES.keys()),
        max_chars=5000,
        supports_streaming=False,
        model_name="neural",
        voice_info=dict(_AZURE_VOICES),
    ),
    "amazon_polly": TTSConfig(
        name="Amazon Polly",
        api_key_env="AWS_ACCESS_KEY_ID",
        base_url="",  # signed via boto3 using AWS_* env credentials
        supported_voices=list(_POLLY_VOICES.keys()),
        max_chars=3000,
        supports_streaming=False,
        model_name="neural",
        voice_info=dict(_POLLY_VOICES),
    ),
}

def get_voice_gender(provider_id: str, voice_id: str) -> str:
    """Get the gender of a voice for a provider"""
    if provider_id in TTS_PROVIDERS:
        voice_info = TTS_PROVIDERS[provider_id].voice_info.get(voice_id)
        if voice_info:
            return voice_info.gender
    return "unknown"

def voice_matches_blind_locale(voice_id: str, info: VoiceInfo, locale: Optional[str]) -> bool:
    """Strict locale match for the blind listening test (catalog accent + voice id prefix)."""
    if locale is None:
        return True
    vl = voice_id.lower().replace("_", "-")
    if locale == "US":
        # Only en-US Murf-style voice IDs; must match catalog accent (no hi-IN leakage).
        return vl.startswith("en-us-") and info.accent == "US"
    if locale == "UK":
        return vl.startswith("en-uk-") and info.accent == "UK"
    if locale == "HI":
        return vl.startswith("hi-in-") and info.accent == "HI"
    if locale == "BN":
        return vl.startswith("bn-in-") and info.accent == "BN"
    if locale == "IN":
        return vl.startswith("en-in-") or (info.accent == "IN" and "en-in" in vl)
    if locale == "TA":
        return vl.startswith("ta-in-") and info.accent == "TA"
    if locale == "FR":
        return vl.startswith("fr-fr-") and info.accent == "FR"
    if locale == "ES":
        return vl.startswith(("es-es-", "es-mx-")) and info.accent == "ES"
    if locale == "MR":
        return vl.startswith("mr-in-") and info.accent == "MR"
    if locale == "ML":
        return vl.startswith("ml-in-") and info.accent == "ML"
    return False


def get_voices_by_gender(provider_id: str, gender: str) -> List[str]:
    """Get voices filtered by gender for a provider - returns only voices matching the gender"""
    if provider_id in TTS_PROVIDERS:
        voices = []
        supported_voices_set = set(TTS_PROVIDERS[provider_id].supported_voices)
        for voice_id, info in TTS_PROVIDERS[provider_id].voice_info.items():
            # Only include voices that match gender AND are in supported_voices
            if info.gender == gender and voice_id in supported_voices_set:
                voices.append(voice_id)
        # Return only matching voices - don't fall back to all voices
        return voices
    return []

def get_voices_by_gender_and_locale(provider_id: str, gender: str, locale: str = None) -> List[str]:
    """Get voices filtered by gender and locale/language for a provider.

    Locale keys (blind UI): US = en-US; IN = en-IN; UK = en-UK; HI = hi-IN; BN = bn-IN; TA = ta-IN.
    Gender must be male or female (lowercase) as set by the UI.
    """
    if provider_id in TTS_PROVIDERS:
        voices = []
        supported_voices_set = set(TTS_PROVIDERS[provider_id].supported_voices)
        for voice_id, info in TTS_PROVIDERS[provider_id].voice_info.items():
            # Only include voices that match gender AND are in supported_voices
            if info.gender == gender and voice_id in supported_voices_set:
                if voice_matches_blind_locale(voice_id, info, locale):
                    voices.append(voice_id)
        return voices
    return []

# Benchmarking Configuration
BENCHMARK_CONFIG = {
    "default_iterations": 3,
    "timeout_seconds": 30,
    "quality_metrics": ["duration", "file_size", "sample_rate"],
    "latency_percentiles": [50, 90, 95, 99],
    "elo_k_factor": 32,
    "initial_elo_rating": 1000
}

# Test Dataset Configuration  
DATASET_CONFIG = {
    "sentence_lengths": {
        "short": (10, 30),    # 10-30 words
        "medium": (31, 80),   # 31-80 words  
        "long": (81, 150),    # 81-150 words
        "very_long": (151, 200) # 151-200 words
    },
    "categories": ["news", "literature", "conversation", "technical", "narrative"],
    "total_samples": 100
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Murf Gen 2 vs NewModel — Listening Test",
    "page_icon": None,
    "layout": "wide",
    "sidebar_width": 300,
    "chart_height": 400,
    "max_file_size_mb": 10
}

def get_api_key(provider: str) -> str:
    """Get API key for a provider from environment variables"""
    if provider not in TTS_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    if provider == "omni_tts":
        return get_omni_falcon_api_key()

    env_var = TTS_PROVIDERS[provider].api_key_env
    api_key = _clean_env(env_var)

    if not api_key:
        raise ValueError(f"API key not found for {provider}. Please set {env_var} environment variable.")

    return api_key

def validate_config() -> Dict[str, Any]:
    """Validate configuration and return status"""
    status = {
        "providers": {},
        "valid": False,
        "errors": [],
        "configured_count": 0
    }
    
    for provider_id, config in TTS_PROVIDERS.items():
        try:
            api_key = get_api_key(provider_id)
            status["providers"][provider_id] = {
                "configured": True,
                "api_key_length": len(api_key) if api_key else 0
            }
            status["configured_count"] += 1
        except ValueError as e:
            status["providers"][provider_id] = {
                "configured": False,
                "error": str(e)
            }
            status["errors"].append(str(e))
    
    # The arena is "ready" when the anchor (Omni) plus at least one competitor are
    # configured. Unconfigured optional competitors are expected and not fatal.
    anchor_ok = status["providers"].get(ANCHOR_PROVIDER, {}).get("configured", False)
    competitors_ok = [
        pid for pid, p in status["providers"].items()
        if pid != ANCHOR_PROVIDER and p.get("configured")
    ]
    status["anchor_ok"] = anchor_ok
    status["configured_competitors"] = competitors_ok
    status["valid"] = bool(anchor_ok and competitors_ok)

    return status


# =============================================================================
# Voice Arena registry
# -----------------------------------------------------------------------------
# Multi-provider blind benchmark. Every battle is Omni (the anchor) vs. one
# competitor. This section declares: the anchor, the pairing strategy, the
# languages, each provider's supported languages + representative voice per
# (language, gender), the business weights for the overall rating, the common
# audio-normalization target, and the rating-engine settings.
# =============================================================================
import math

# --- Anchor + pairing --------------------------------------------------------
# Locked decision: Omni / NewModel is the anchor; its strength is pinned to 0.
# Swap this single value (behind the PairingStrategy interface in scheduler.py)
# to re-anchor without code changes.
ANCHOR_PROVIDER: str = os.getenv("ARENA_ANCHOR", "omni_tts")

# "anchor_only" (default) | "anchor_plus" | "all_vs_all" (see scheduler.py).
PAIRING_STRATEGY: str = os.getenv("ARENA_PAIRING_STRATEGY", "anchor_only")

# --- Languages ---------------------------------------------------------------
# BCP-47-style locale keys -> human display name. Extensible: add a row here,
# add provider support below, and add corpus lines for the language.
LANGUAGES: Dict[str, str] = dict(ARENA_LANGUAGES)

# Which corpus bucket (voice_battle_corpus.tsv locale tag) feeds each language.
# en-IN / en-UK reuse the en-US corpus lines until dedicated rows exist.
LANGUAGE_TO_CORPUS: Dict[str, str] = {
    "en-US": "en-US",
    "en-IN": "en-US",
    "en-UK": "en-US",
    "hi-IN": "hi-IN",
    "bn-IN": "bn-IN",
    "ta-IN": "ta-IN",
    "fr-FR": "fr-FR",
    "es-ES": "es-ES",
    "mr-IN": "mr-IN",
    "ml-IN": "ml-IN",
}

# Legacy blind-UI locale key for each language (voice helpers + corpus loader).
LANGUAGE_TO_UI_LOCALE: Dict[str, str] = {
    "en-US": "US",
    "en-IN": "IN",
    "en-UK": "UK",
    "hi-IN": "HI",
    "bn-IN": "BN",
    "ta-IN": "TA",
    "fr-FR": "FR",
    "es-ES": "ES",
    "mr-IN": "MR",
    "ml-IN": "ML",
}

# Per-provider languages with 2 male + 2 female voice ids per language.
# Scheduler uses the first id for each gender (see representative_voice).
PROVIDER_LANGUAGES = build_provider_languages(is_falcon_dev_stream())

# --- Overall rating: business weights per language (DISCLOSED) ----------------
# Equal for now (tunable later). Normalized at use-time over active languages.
LANGUAGE_WEIGHTS: Dict[str, float] = {lang: 1.0 for lang in LANGUAGES}

# --- Common audio-normalization target (blindness integrity) -----------------
# Consumed by audio_norm.NormalizationParams; kept as a plain dict here so
# importing config never pulls in the heavy audio stack.
NORMALIZATION: Dict[str, Any] = {
    "target_lufs": -23.0,
    "true_peak_dbfs": -1.0,
    "sample_rate": 24000,
    "channels": 1,
    "container": "wav",
    "codec": "pcm_s16le",
}

# --- Rating engine settings (deterministic) ----------------------------------
ENGINE_CONFIG: Dict[str, Any] = {
    # Bradley-Terry + Davidson tie MLE.
    "anchor_strength": 0.0,                # Omni pinned to 0 for identifiability.
    "regularization": 1e-6,                # tiny ridge for numerical stability.
    # Bootstrap confidence intervals (fixed seed => reproducible).
    "bootstrap_samples": 400,
    "bootstrap_seed": 12345,
    "ci_percentiles": [2.5, 97.5],         # 95% CI.
    # Elo-like display rescale: elo = anchor + scale * strength.
    "elo_display_anchor": 1000.0,
    "elo_display_scale": 400.0 / math.log(10.0),  # ~173.72 (logistic->Elo).
    # Adaptive sampling (PRD 7). ON, allocating toward the target CI width.
    # Target is +-10 Elo-display points => full 95% CI width of 20 points.
    "adaptive_sampling": True,
    "target_ci_halfwidth_elo": 10.0,
    "target_ci_width_elo": 20.0,
}


# --- Registry helpers --------------------------------------------------------
def anchor_provider() -> str:
    """The pinned anchor provider id (Omni)."""
    return ANCHOR_PROVIDER


def get_language_display(language: str) -> str:
    return LANGUAGES.get(language, language)


def corpus_locale_for_language(language: str) -> str:
    """Corpus TSV bucket tag for a language (en-IN/en-UK -> en-US)."""
    return LANGUAGE_TO_CORPUS.get(language, "en-US")


def ui_locale_for_language(language: str) -> str:
    """Legacy blind-UI locale key (US/IN/UK/HI/BN/TA) for a language."""
    return LANGUAGE_TO_UI_LOCALE.get(language, "US")


def provider_supports_language(provider_id: str, language: str) -> bool:
    return language in PROVIDER_LANGUAGES.get(provider_id, {})


def representative_voice(provider_id: str, language: str, gender: str) -> Optional[str]:
    """Pinned representative voice id for a (provider, language, gender)."""
    raw = PROVIDER_LANGUAGES.get(provider_id, {}).get(language, {}).get(gender)
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


def language_voices(provider_id: str, language: str, gender: str) -> List[str]:
    """All configured voice ids (up to 2) for a provider/language/gender."""
    raw = PROVIDER_LANGUAGES.get(provider_id, {}).get(language, {}).get(gender)
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    return [raw]


def is_provider_configured(provider_id: str) -> bool:
    """True if the provider's API key is present."""
    try:
        get_api_key(provider_id)
        return True
    except ValueError:
        return False


def competitors_for_language(language: str, configured_only: bool = True) -> List[str]:
    """Competitors (excluding the anchor) that support `language`.

    When configured_only, also requires the competitor's API key to be present.
    Ordered by the PROVIDER_LANGUAGES declaration order for stability.
    """
    out: List[str] = []
    for pid in PROVIDER_LANGUAGES:
        if pid == ANCHOR_PROVIDER:
            continue
        if not provider_supports_language(pid, language):
            continue
        if configured_only and not is_provider_configured(pid):
            continue
        out.append(pid)
    return out


def languages_available(configured_only: bool = True) -> List[str]:
    """Languages where the anchor supports the language and (when
    configured_only) the anchor is configured and at least one supporting
    competitor is configured."""
    out: List[str] = []
    anchor = ANCHOR_PROVIDER
    anchor_ready = (not configured_only) or is_provider_configured(anchor)
    for language in LANGUAGES:
        if not provider_supports_language(anchor, language):
            continue
        if not anchor_ready:
            continue
        comps = competitors_for_language(language, configured_only=configured_only)
        if comps:
            out.append(language)
    return out


def language_weights(languages: Optional[List[str]] = None) -> Dict[str, float]:
    """Normalized business weights over the given (or active) languages."""
    langs = languages if languages is not None else languages_available()
    raw = {l: float(LANGUAGE_WEIGHTS.get(l, 1.0)) for l in langs}
    total = sum(raw.values())
    if total <= 0:
        return {l: 1.0 / len(raw) for l in raw} if raw else {}
    return {l: w / total for l, w in raw.items()}