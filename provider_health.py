"""Live provider health checks for the Voice Arena UI.

Each configured provider gets a minimal synthesis probe so the sidebar can
show working vs failing vs not configured (not just whether a key exists).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Set

import config
from tts_providers import TTSProviderFactory, TTSRequest

# Short probe text per language bucket (keeps checks fast and cheap).
_TEST_TEXT = {
    "en-US": "Voice arena connection test.",
    "en-IN": "Voice arena connection test.",
    "en-UK": "Voice arena connection test.",
    "hi-IN": "यह एक संक्षिप्त परीक्षण है।",
    "bn-IN": "এটি একটি সংক্ষিপ্ত পরীক্ষা।",
    "ta-IN": "இது ஒரு சிறிய சோதனை.",
    "fr-FR": "Ceci est un court test.",
    "es-ES": "Esta es una prueba breve.",
    "mr-IN": "हा एक छोटा चाचणी वाक्य आहे.",
    "ml-IN": "ഇത് ഒരു ചെറിയ പരീക്ഷണ വാക്യമാണ്.",
}
_DEFAULT_TIMEOUT = 25.0


def _test_text(language: str) -> str:
    return _TEST_TEXT.get(language, "Voice arena connection test.")


def _test_target(provider_id: str) -> tuple[Optional[str], Optional[str]]:
    """Pick a (language, voice) pair for probing this provider."""
    langs = config.PROVIDER_LANGUAGES.get(provider_id, {})
    if not langs:
        return None, None
    language = "en-US" if "en-US" in langs else next(iter(langs))
    voices = langs[language]
    voice = voices.get("female") or voices.get("male")
    if isinstance(voice, list):
        voice = voice[0] if voice else None
    return language, voice


def _configured_but_blocked(provider_id: str) -> Optional[str]:
    """Pre-flight failures before hitting the network."""
    if not config.is_provider_configured(provider_id):
        return None
    if provider_id == "azure_tts" and not os.getenv("AZURE_SPEECH_REGION", "").strip():
        return "AZURE_SPEECH_REGION not set"
    if provider_id == "amazon_polly":
        if not os.getenv("AWS_ACCESS_KEY_ID", "").strip():
            return "AWS_ACCESS_KEY_ID not set"
        if not os.getenv("AWS_SECRET_ACCESS_KEY", "").strip():
            return "AWS_SECRET_ACCESS_KEY not set"
    return None


async def check_provider(
    provider_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Run one synthesis probe for a provider."""
    blocked = _configured_but_blocked(provider_id)
    if blocked:
        return {
            "state": "fail",
            "configured": True,
            "message": blocked,
            "latency_ms": None,
        }

    if not config.is_provider_configured(provider_id):
        return {
            "state": "unconfigured",
            "configured": False,
            "message": "No credentials",
            "latency_ms": None,
        }

    language, voice = _test_target(provider_id)
    if not voice:
        return {
            "state": "fail",
            "configured": True,
            "message": "No representative voice",
            "latency_ms": None,
        }

    text = _test_text(language or "en-US")
    try:
        provider = TTSProviderFactory.create_provider(provider_id)
        result = await asyncio.wait_for(
            provider.generate_speech(
                TTSRequest(text=text, voice=voice, provider=provider_id)
            ),
            timeout=timeout,
        )
        if result.success and result.audio_data:
            return {
                "state": "ok",
                "configured": True,
                "message": f"{result.latency_ms:.0f} ms",
                "latency_ms": result.latency_ms,
            }
        err = (result.error_message or "Synthesis failed").replace("\n", " ")
        # #region agent log
        try:
            from tts_providers import _agent_log
            _agent_log(
                "B",
                "provider_health.py:check_provider",
                "provider_probe_failed",
                {
                    "provider_id": provider_id,
                    "error_snippet": err[:200],
                    "has_unusual_activity": "detected_unusual_activity" in err,
                    "is_cloud": bool(os.getenv("STREAMLIT_SERVER_PORT")),
                },
            )
        except Exception:
            pass
        # #endregion
        return {
            "state": "fail",
            "configured": True,
            "message": err[:220],
            "latency_ms": result.latency_ms,
        }
    except asyncio.TimeoutError:
        return {
            "state": "fail",
            "configured": True,
            "message": "Timed out",
            "latency_ms": None,
        }
    except Exception as e:
        return {
            "state": "fail",
            "configured": True,
            "message": str(e).replace("\n", " ")[:140],
            "latency_ms": None,
        }


async def check_all(
    provider_ids: Optional[List[str]] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Dict[str, Dict[str, Any]]:
    """Probe every arena provider (parallel)."""
    ids = provider_ids or list(config.TTS_PROVIDERS.keys())
    checked_at = time.time()
    results = await asyncio.gather(
        *[check_provider(pid, timeout=timeout) for pid in ids]
    )
    out: Dict[str, Dict[str, Any]] = {}
    for pid, row in zip(ids, results):
        out[pid] = {**row, "checked_at": checked_at}
    return out


def healthy_ids(health: Dict[str, Dict[str, Any]]) -> Set[str]:
    return {pid for pid, row in health.items() if row.get("state") == "ok"}


def languages_available(health: Dict[str, Dict[str, Any]]) -> List[str]:
    """Languages where anchor + at least one competitor passed the health probe."""
    ok = healthy_ids(health)
    anchor = config.anchor_provider()
    if anchor not in ok:
        return []
    out: List[str] = []
    for language in config.LANGUAGES:
        if not config.provider_supports_language(anchor, language):
            continue
        comps = [
            pid for pid in config.competitors_for_language(language, configured_only=False)
            if pid in ok
        ]
        if comps:
            out.append(language)
    return out


def competitors_for_language(
    language: str,
    health: Dict[str, Dict[str, Any]],
) -> List[str]:
    ok = healthy_ids(health)
    return [
        pid for pid in config.competitors_for_language(language, configured_only=False)
        if pid in ok
    ]


def arena_ready(health: Dict[str, Dict[str, Any]]) -> bool:
    return bool(languages_available(health))
