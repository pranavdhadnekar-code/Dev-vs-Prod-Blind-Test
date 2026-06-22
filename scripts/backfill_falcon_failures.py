#!/usr/bin/env python3
"""Backfill falcon_failures for anchor losses that were never captured.

Uses battle_raw_audio when still present; otherwise re-synthesizes from the
stored script, provider, and voice IDs on each battle row.

Re-synthesized clips are tagged audio_source=resynth — they approximate the
original listen test but are not bit-identical to what raters heard.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from database import BenchmarkDatabase
from falcon_failure_backfill import backfill_missing_failures


def _parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        help="Only votes after this ISO timestamp (e.g. 2026-06-18T23:20:00+00:00)",
    )
    parser.add_argument("--battle-id", help="Backfill a single battle_id")
    parser.add_argument("--limit", type=int, help="Max battles to process")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Parallel re-synthesis jobs (default 2)")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="Per-provider synthesis timeout seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="List missing failures without writing")
    args = parser.parse_args()

    anchor = config.anchor_provider()
    db = BenchmarkDatabase()
    since = _parse_since(args.since)

    pending = db.get_omni_losses_missing_failures(
        anchor=anchor, since=since, battle_id=args.battle_id)
    print(f"Anchor: {anchor}")
    print(f"Missing falcon_failures: {len(pending)}")
    if args.limit:
        print(f"Limit: {args.limit}")

    results = asyncio.run(backfill_missing_failures(
        db,
        anchor=anchor,
        since=since,
        battle_id=args.battle_id,
        limit=args.limit,
        concurrency=args.concurrency,
        timeout=args.timeout,
        dry_run=args.dry_run,
    ))

    ok = sum(1 for r in results if r.ok)
    fail = sum(1 for r in results if not r.ok)
    by_source: dict[str, int] = {}
    for r in results:
        if r.ok:
            by_source[r.source] = by_source.get(r.source, 0) + 1

    print(f"\nProcessed: {len(results)}  ok={ok}  fail={fail}")
    if by_source:
        print("Sources:", ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))

    for r in results:
        mark = "OK" if r.ok else "FAIL"
        print(f"  [{mark}] {r.battle_id} ({r.source}) — {r.detail}")

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
