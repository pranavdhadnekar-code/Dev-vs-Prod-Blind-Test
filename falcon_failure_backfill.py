"""Backfill falcon_failures from votes + battles when live capture missed."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import config
from database import BenchmarkDatabase
from tts_providers import TTSProviderFactory, TTSRequest

# Deprecated voice IDs on old battles → current catalog entries for re-synthesis.
_BACKFILL_VOICE_ALIASES: dict[str, str] = {
    "en-US-will": "en-US-gordon",
    "en-US-ezekiel": "en-US-matthew",
    "en-US-lillian": "en-US-luna",
    "en-US-olivia": "en-US-alicia",
    "en-US-madison": "en-US-natalie",
    "en-UK-bertie": "en-UK-jake",
    "en-UK-heidi": "en-UK-sharon",
    "en-IN-arjun": "en-IN-abhinav",
    "en-IN-palak": "en-IN-pooja",
}


def resolve_backfill_voice(provider_id: str, voice_id: str) -> str:
    """Map deprecated voice IDs to current catalog entries for re-synthesis."""
    if provider_id != config.anchor_provider():
        return voice_id
    aliased = _BACKFILL_VOICE_ALIASES.get(voice_id, voice_id)
    allowed = set(config.get_falcon_supported_voices())
    if aliased in allowed:
        return aliased
    if voice_id in allowed:
        return voice_id
    # Same locale prefix, first supported voice as last resort.
    prefix = voice_id.rsplit("-", 1)[0] + "-" if "-" in voice_id else ""
    for vid in config.get_falcon_supported_voices():
        if prefix and vid.startswith(prefix):
            return vid
    return aliased


def raw_audio_ext(data: bytes | None) -> str:
    return "wav" if data and data[:4] == b"RIFF" else "mp3"


def failure_parties(
    row: Dict[str, Any], anchor: Optional[str] = None
) -> Tuple[str, str, str, str]:
    """Return falcon_provider, falcon_voice, competitor_provider, competitor_voice."""
    anchor = anchor or config.anchor_provider()
    left_p = row["left_provider"]
    if left_p == anchor:
        return anchor, row["left_voice"], row["right_provider"], row["right_voice"]
    return anchor, row["right_voice"], row["left_provider"], row["left_voice"]


def try_promote_stored_raw_audio(
    db: BenchmarkDatabase, row: Dict[str, Any], anchor: Optional[str] = None
) -> bool:
    """Use battle_raw_audio staging row if it still exists (no re-synthesis)."""
    stored = db.get_battle_raw_audio(row["battle_id"])
    if not stored or not stored.get("falcon_audio_bytes"):
        return False

    _, falcon_voice, comp_provider, comp_voice = failure_parties(row, anchor)
    saved = db.save_falcon_failure(
        battle_id=row["battle_id"],
        language=row["language"],
        item_id=row["item_id"],
        item_text=row["item_text"],
        falcon_voice=falcon_voice,
        competitor_provider=comp_provider,
        competitor_voice=comp_voice,
        outcome=row["outcome"],
        falcon_audio_bytes=stored["falcon_audio_bytes"],
        falcon_audio_format=stored.get("falcon_audio_format") or "wav",
        competitor_audio_bytes=stored.get("competitor_audio_bytes") or b"",
        competitor_audio_format=stored.get("competitor_audio_format") or "wav",
        rater_session=row.get("rater_session") or "default",
        comment=row.get("comment") or "",
        audio_source="live",
        created_at=row.get("voted_at"),
    )
    if saved:
        db.delete_battle_raw_audio(row["battle_id"])
    return saved


async def _synth_one(
    provider_id: str, voice: str, text: str, timeout: float = 90.0
):
    provider = TTSProviderFactory.create_provider(provider_id)
    req = TTSRequest(text=text, voice=voice, provider=provider_id)
    return await asyncio.wait_for(provider.generate_speech(req), timeout=timeout)


@dataclass
class BackfillResult:
    battle_id: str
    ok: bool
    source: str
    detail: str


async def resynthesize_failure(
    db: BenchmarkDatabase,
    row: Dict[str, Any],
    anchor: Optional[str] = None,
    timeout: float = 90.0,
) -> BackfillResult:
    """Re-synthesize Falcon + competitor clips from stored battle metadata."""
    bid = row["battle_id"]
    text = row["item_text"] or ""
    if not text.strip():
        return BackfillResult(bid, False, "resynth", "empty item_text")

    anchor_p, falcon_voice, comp_provider, comp_voice = failure_parties(row, anchor)
    falcon_voice = resolve_backfill_voice(anchor_p, falcon_voice)
    try:
        falcon_res, comp_res = await asyncio.gather(
            _synth_one(anchor_p, falcon_voice, text, timeout),
            _synth_one(comp_provider, comp_voice, text, timeout),
            return_exceptions=True,
        )
    except Exception as e:
        return BackfillResult(bid, False, "resynth", str(e)[:200])

    for label, res in (("falcon", falcon_res), ("competitor", comp_res)):
        if isinstance(res, Exception):
            return BackfillResult(bid, False, "resynth", f"{label}: {res}")
        if not getattr(res, "success", False) or not res.audio_data:
            msg = getattr(res, "error_message", None) or f"{label} synthesis failed"
            return BackfillResult(bid, False, "resynth", msg[:200])

    falcon_audio = falcon_res.audio_data
    comp_audio = comp_res.audio_data
    saved = db.save_falcon_failure(
        battle_id=bid,
        language=row["language"],
        item_id=row["item_id"],
        item_text=text,
        falcon_voice=falcon_voice,
        competitor_provider=comp_provider,
        competitor_voice=comp_voice,
        outcome=row["outcome"],
        falcon_audio_bytes=falcon_audio,
        falcon_audio_format=raw_audio_ext(falcon_audio),
        competitor_audio_bytes=comp_audio,
        competitor_audio_format=raw_audio_ext(comp_audio),
        rater_session=row.get("rater_session") or "default",
        comment=row.get("comment") or "",
        audio_source="resynth",
        created_at=row.get("voted_at"),
    )
    if not saved:
        return BackfillResult(bid, False, "resynth", "insert skipped (duplicate?)")
    return BackfillResult(bid, True, "resynth", "ok")


async def backfill_missing_failures(
    db: BenchmarkDatabase,
    *,
    anchor: Optional[str] = None,
    since: Optional[datetime] = None,
    battle_id: Optional[str] = None,
    limit: Optional[int] = None,
    concurrency: int = 2,
    timeout: float = 90.0,
    dry_run: bool = False,
) -> list[BackfillResult]:
    anchor = anchor or config.anchor_provider()
    rows = db.get_omni_losses_missing_failures(
        anchor=anchor, since=since, battle_id=battle_id)
    if limit is not None:
        rows = rows[:limit]

    results: list[BackfillResult] = []
    pending_resynth: list[Dict[str, Any]] = []

    for row in rows:
        bid = row["battle_id"]
        if dry_run:
            stored = db.get_battle_raw_audio(bid)
            source = "live" if stored and stored.get("falcon_audio_bytes") else "resynth"
            results.append(BackfillResult(bid, True, source, "dry-run"))
            continue
        if try_promote_stored_raw_audio(db, row, anchor):
            results.append(BackfillResult(bid, True, "live", "promoted staging audio"))
        else:
            pending_resynth.append(row)

    if dry_run or not pending_resynth:
        return results

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run(row: Dict[str, Any]) -> BackfillResult:
        async with sem:
            return await resynthesize_failure(db, row, anchor, timeout)

    resynth_out = await asyncio.gather(*[_run(r) for r in pending_resynth])
    results.extend(resynth_out)
    return results
