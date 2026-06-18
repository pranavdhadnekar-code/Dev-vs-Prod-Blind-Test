#!/usr/bin/env python3
"""Probe every voice in the changed ElevenLabs / Deepgram / Cartesia pools."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

# project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from arena_language_registry import build_provider_languages
from provider_health import _TEST_TEXT
from tts_providers import TTSProviderFactory, TTSRequest

PROVIDERS = ("elevenlabs_v3", "deepgram_aura2", "cartesia_sonic3")
CONCURRENCY = 4


@dataclass
class Row:
    provider: str
    language: str
    gender: str
    voice: str
    ok: bool
    detail: str
    ms: Optional[float] = None


async def probe_one(sem: asyncio.Semaphore, provider: str, language: str,
                    gender: str, voice: str) -> Row:
    text = _TEST_TEXT.get(language, "Voice test.")
    async with sem:
        try:
            p = TTSProviderFactory.create_provider(provider)
            req = TTSRequest(text=text, voice=voice, provider=provider)
            result = await asyncio.wait_for(p.generate_speech(req), timeout=30.0)
            if result.success and result.audio_data:
                return Row(provider, language, gender, voice, True,
                           f"{result.file_size_bytes} bytes", result.latency_ms)
            return Row(provider, language, gender, voice, False,
                       (result.error_message or "empty audio")[:120])
        except Exception as e:
            return Row(provider, language, gender, voice, False, str(e)[:120])


async def main() -> int:
    pools = build_provider_languages()
    tasks: List[asyncio.Task] = []
    sem = asyncio.Semaphore(CONCURRENCY)
    skipped: List[str] = []

    for pid in PROVIDERS:
        if not config.is_provider_configured(pid):
            skipped.append(f"{pid}: API key not configured")
            continue
        lang_pools = pools.get(pid, {})
        for language, genders in sorted(lang_pools.items()):
            for gender, voices in genders.items():
                for voice in voices:
                    tasks.append(asyncio.create_task(
                        probe_one(sem, pid, language, gender, voice)))

    if skipped:
        print("SKIPPED (no API key):")
        for s in skipped:
            print(f"  - {s}")
        print()

    if not tasks:
        print("No probes to run — configure at least one provider API key.")
        return 1

    print(f"Probing {len(tasks)} voices ({CONCURRENCY} concurrent)…\n")
    rows = await asyncio.gather(*tasks)

    by_provider: dict[str, list[Row]] = {}
    for r in rows:
        by_provider.setdefault(r.provider, []).append(r)

    exit_code = 0
    for pid in PROVIDERS:
        group = by_provider.get(pid)
        if not group:
            continue
        ok_n = sum(1 for r in group if r.ok)
        print(f"=== {config.TTS_PROVIDERS[pid].name} ({pid}) — {ok_n}/{len(group)} OK ===")
        for r in sorted(group, key=lambda x: (x.language, x.gender, x.voice)):
            mark = "OK" if r.ok else "FAIL"
            if not r.ok:
                exit_code = 1
            extra = f" ({r.ms:.0f} ms)" if r.ms is not None else ""
            print(f"  [{mark}] {r.language} {r.gender}: {r.voice[:36]}… — {r.detail}{extra}"
                  if len(r.voice) > 36 else
                  f"  [{mark}] {r.language} {r.gender}: {r.voice} — {r.detail}{extra}")
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
