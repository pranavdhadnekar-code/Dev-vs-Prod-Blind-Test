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
    FALCON_BATTLE_VOICES,
    FALCON_PROVIDER_IDS,
    MURF_VOICE_GENDERS,
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


def get_falcon_dev_url() -> str:
    """Murf Falcon dev stream endpoint."""
    return (
        _clean_env("FALCON_DEV_URL")
        or _clean_env("MURF_FALCON_URL")
        or "https://api.dev.murf.ai/v1/speech/stream"
    )


def get_falcon_prod_url() -> str:
    """Murf Falcon production endpoint."""
    return (
        _clean_env("FALCON_PROD_URL")
        or "https://api.murf.ai/v1/speech/stream"
    )


def get_falcon_api_url(provider_id: str = "falcon_dev") -> str:
    """Resolve Falcon streaming URL for a provider id."""
    if provider_id == "falcon_prod":
        return get_falcon_prod_url()
    return get_falcon_dev_url()


def is_falcon_dev_stream(url: Optional[str] = None) -> bool:
    """True when the URL points at the Murf dev stream endpoint."""
    target = url if url is not None else get_falcon_dev_url()
    return "dev.murf.ai" in target


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


def falcon_synthesis_timeout(provider_id: str) -> int:
    """Client-side aiohttp total timeout (seconds) for Falcon synthesis."""
    if provider_id == "falcon_dev":
        raw = _clean_env("FALCON_DEV_TIMEOUT") or "180"
    elif provider_id == "falcon_prod":
        raw = _clean_env("FALCON_PROD_TIMEOUT") or "120"
    else:
        raw = "25"
    return int(float(raw))


def health_check_timeout(provider_id: str) -> float:
    """asyncio.wait_for limit for sidebar health probes (per provider)."""
    if provider_id == "falcon_dev":
        raw = _clean_env("FALCON_DEV_HEALTH_TIMEOUT")
        default = str(falcon_synthesis_timeout("falcon_dev"))
    elif provider_id == "falcon_prod":
        raw = _clean_env("FALCON_PROD_HEALTH_TIMEOUT")
        default = "45"
    else:
        raw = _clean_env("ARENA_HEALTH_TIMEOUT")
        default = "25"
    return float(raw or default)


def get_falcon_dev_api_key() -> str:
    """API key / JWT for Falcon dev stream."""
    key = _clean_env("FALCON_DEV_API_KEY") or _clean_env("OMNI_API_KEY")
    if key:
        return key
    raise ValueError(
        "Set FALCON_DEV_API_KEY (or OMNI_API_KEY) to a Murf ap2_ key or JWT for Falcon dev."
    )


def get_falcon_prod_api_key() -> str:
    """API key for Falcon production."""
    key = _clean_env("FALCON_PROD_API_KEY") or _clean_env("MURF_API_KEY")
    if key and is_murf_api_key(key):
        return key
    if key:
        return key
    raise ValueError(
        "Set FALCON_PROD_API_KEY (or MURF_API_KEY) to a Murf ap2_ production API key."
    )


def get_omni_falcon_api_key() -> str:
    """Backward-compatible alias for Falcon dev credentials."""
    return get_falcon_dev_api_key()


def _infer_voice_info(voice_id: str) -> VoiceInfo:
    """Build display metadata from a Murf voice id and FALCON_BATTLE_VOICES gender."""
    parts = voice_id.split("-")
    accent_map = {"US": "US", "UK": "UK", "IN": "IN"}
    accent = accent_map.get(parts[1], parts[1].upper()) if len(parts) >= 2 else "US"
    slug = parts[-1].split("[")[0].replace("cc-", "")
    name = slug[:1].upper() + slug[1:] if slug else voice_id
    gender = MURF_VOICE_GENDERS.get(voice_id, "male")
    return VoiceInfo(voice_id, name, gender, accent)


_FALCON_BATTLE_VOICE_IDS: List[str] = flatten_voice_ids(FALCON_BATTLE_VOICES)


def get_falcon_supported_voices(provider_id: str = "falcon_dev") -> List[str]:
    """Voice IDs used for Falcon validation (shared dev/prod battle pool)."""
    return list(_FALCON_BATTLE_VOICE_IDS)


def get_falcon_voice_info() -> Dict[str, VoiceInfo]:
    return {vid: _infer_voice_info(vid) for vid in _FALCON_BATTLE_VOICE_IDS}

TTS_PROVIDERS = {
    "falcon_dev": TTSConfig(
        name="Dev",
        api_key_env="FALCON_DEV_API_KEY",
        base_url=get_falcon_dev_url(),
        supported_voices=get_falcon_supported_voices(),
        max_chars=5000,
        supports_streaming=True,
        model_name="FALCON",
        voice_info=get_falcon_voice_info(),
    ),
    "falcon_prod": TTSConfig(
        name="Prod",
        api_key_env="FALCON_PROD_API_KEY",
        base_url=get_falcon_prod_url(),
        supported_voices=get_falcon_supported_voices(),
        max_chars=5000,
        supports_streaming=True,
        model_name="FALCON",
        voice_info=get_falcon_voice_info(),
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
    "page_title": "Falcon Dev vs Prod — Listening Test",
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

    if provider == "falcon_dev":
        return get_falcon_dev_api_key()
    if provider == "falcon_prod":
        return get_falcon_prod_api_key()

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
def _resolve_anchor_provider(raw: str) -> str:
    """Map legacy provider ids to the current Falcon dev/prod registry."""
    legacy = {
        "omni_tts": "falcon_dev",
        "murf_gen2": "falcon_prod",
    }
    return legacy.get(raw, raw)


# Production Falcon is the default reference (strength pinned to 0).
ANCHOR_PROVIDER: str = _resolve_anchor_provider(
    os.getenv("ARENA_ANCHOR", "falcon_prod")
)

# "anchor_only" (default) | "anchor_plus" | "all_vs_all" (see scheduler.py).
PAIRING_STRATEGY: str = os.getenv("ARENA_PAIRING_STRATEGY", "anchor_only")

# --- Languages ---------------------------------------------------------------
# BCP-47-style locale keys -> human display name. Extensible: add a row here,
# add provider support below, and add corpus lines for the language.
LANGUAGES: Dict[str, str] = dict(ARENA_LANGUAGES)

# Which corpus bucket feeds each language (all share en-shared).
LANGUAGE_TO_CORPUS: Dict[str, str] = {
    "en-US": "en-shared",
    "en-IN": "en-shared",
    "en-UK": "en-shared",
}

# Legacy blind-UI locale key for each language (voice helpers + corpus loader).
LANGUAGE_TO_UI_LOCALE: Dict[str, str] = {
    "en-US": "US",
    "en-IN": "IN",
    "en-UK": "UK",
}

# Per-provider languages with male/female voice id lists per language.
# Scheduler picks one at random per gender (see language_voices).
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
def share_voice_across_providers(provider_a: str, provider_b: str) -> bool:
    """True when both clips in a battle must use the same voice id."""
    return provider_a in FALCON_PROVIDER_IDS and provider_b in FALCON_PROVIDER_IDS


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
    pool = PROVIDER_LANGUAGES.get(provider_id, {}).get(language)
    if not pool:
        return False
    return bool(pool.get("male") or pool.get("female"))


def representative_voice(provider_id: str, language: str, gender: str) -> Optional[str]:
    """First configured voice id for a (provider, language, gender).

    Used for health probes and other pinned checks; battles randomize via
    language_voices().
    """
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