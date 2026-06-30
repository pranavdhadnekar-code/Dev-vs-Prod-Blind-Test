#!/usr/bin/env python3
"""Insert temporary demo battles + votes so the leaderboard UI has data to show.

Safe to re-run: removes prior rows whose battle_id starts with ``seed-demo-``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from arena_comparison_corpus import ARENA_COMPARISON_TEXTS
from arena_language_registry import FALCON_BATTLE_VOICES
from database import BenchmarkDatabase
from scheduler import BattlePlan, assign_sides

SEED_PREFIX = "seed-demo-"
ANCHOR = config.anchor_provider()
COMPETITOR = "falcon_dev" if ANCHOR == "falcon_prod" else "falcon_prod"

# (language, outcome, days_ago) — outcome from rater: A=left, B=right, tie
VOTE_PLAN = [
    # en-US — prod anchor slightly ahead overall
    ("en-US", "A", 12, "male", "en-US-will", "Prod sounds clearer on sibilants."),
    ("en-US", "B", 11, "female", "en-US-lillian", "Dev felt more natural in pacing."),
    ("en-US", "A", 10, "male", "en-US-caleb", None),
    ("en-US", "B", 9, "female", "en-US-olivia", "Less robotic on longer sentences."),
    ("en-US", "tie", 8, "male", "en-US-tyler", "Honestly could not tell them apart."),
    ("en-US", "A", 7, "female", "en-US-madison", None),
    ("en-US", "B", 6, "male", "en-US-ezekiel", "Dev pronunciation of numbers was better."),
    ("en-US", "A", 5, "female", "en-US-natalie", None),
    # en-UK
    ("en-UK", "B", 12, "male", "en-UK-benedict", "Dev had warmer tone for this voice."),
    ("en-UK", "A", 11, "female", "en-UK-lydia", None),
    ("en-UK", "B", 10, "male", "en-UK-joshua", "Slight metallic edge on prod clip."),
    ("en-UK", "tie", 9, "female", "en-UK-lucy", None),
    ("en-UK", "A", 8, "male", "en-UK-jake", None),
    ("en-UK", "B", 7, "female", "en-UK-sharon", "Dev pause before commas felt right."),
    # en-IN
    ("en-IN", "A", 12, "male", "en-IN-nikhil", None),
    ("en-IN", "B", 11, "female", "en-IN-anisha", "Dev intonation on Indian English was stronger."),
    ("en-IN", "A", 10, "male", "en-IN-samar", None),
    ("en-IN", "B", 9, "female", "en-IN-anusha", None),
    ("en-IN", "tie", 8, "male", "en-IN-arjun", "Both acceptable for this sentence."),
    ("en-IN", "A", 7, "female", "en-IN-pooja", None),
    ("en-IN", "B", 6, "male", "en-IN-abhinav", "Prod clip clipped slightly at the end."),
]


def _clip_hash(battle_id: str, side: str) -> str:
    return hashlib.sha256(f"{battle_id}:{side}:demo".encode()).hexdigest()


def _clear_seed_rows(db: BenchmarkDatabase) -> int:
    conn = db._connect()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM votes WHERE battle_id LIKE ?",
        (f"{SEED_PREFIX}%",),
    )
    votes = cursor.rowcount if not db.use_postgres else None
    cursor.execute(
        "DELETE FROM battles WHERE battle_id LIKE ?",
        (f"{SEED_PREFIX}%",),
    )
    battles = cursor.rowcount if not db.use_postgres else None
    conn.commit()
    conn.close()
    if votes is None:
        return -1
    return int(votes or 0) + int(battles or 0)


def _insert_battle_and_vote(
    db: BenchmarkDatabase,
    *,
    language: str,
    outcome: str,
    days_ago: int,
    gender: str,
    voice: str,
    comment: str | None,
    text: str,
    item_idx: int,
) -> str:
    battle_id = f"{SEED_PREFIX}{uuid.uuid4().hex[:12]}"
    position_seed = abs(hash(battle_id)) % (2**31)
    swap = assign_sides(ANCHOR, COMPETITOR, position_seed)
    if swap:
        left_p, right_p = COMPETITOR, ANCHOR
    else:
        left_p, right_p = ANCHOR, COMPETITOR

    plan = BattlePlan(
        battle_id=battle_id,
        language=language,
        gender=gender,
        item_id=f"demo-{item_idx:04d}",
        item_text=text,
        strategy=config.PAIRING_STRATEGY,
        anchor=ANCHOR,
        competitor=COMPETITOR,
        is_anchor_pair=True,
        provider_a=ANCHOR,
        provider_b=COMPETITOR,
        left_provider=left_p,
        left_voice=voice,
        right_provider=right_p,
        right_voice=voice,
        position_seed=position_seed,
    )
    db.create_battle(
        plan,
        _clip_hash(battle_id, "left"),
        _clip_hash(battle_id, "right"),
        normalization_params=config.NORMALIZATION,
        session_id="seed-demo",
        location={"country": "US", "city": "Demo", "region": "Seed"},
    )

    left_name = config.TTS_PROVIDERS[left_p].name
    right_name = config.TTS_PROVIDERS[right_p].name
    deanonymized = comment
    if comment:
        deanonymized = (
            comment.replace("Prod", left_name if left_p == ANCHOR else right_name)
            .replace("Dev", left_name if left_p == COMPETITOR else right_name)
        )

    battle = db.get_battle(battle_id)
    left_p = battle["left_provider"]
    right_p = battle["right_provider"]
    if outcome == "A":
        winner, loser = left_p, right_p
    elif outcome == "B":
        winner, loser = right_p, left_p
    else:
        winner, loser = None, None

    conn = db._connect()
    cursor = conn.cursor()
    created = datetime.now() - timedelta(days=days_ago)
    cursor.execute(
        '''
        INSERT INTO votes
        (battle_id, outcome, winner_provider, loser_provider, left_provider,
         right_provider, language, comment, comment_deanonymized, rater_session,
         location_country, location_city, location_region, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''',
        (
            battle_id, outcome, winner, loser, left_p, right_p, language,
            comment or "", deanonymized or "", "seed-demo",
            "US", "Demo", "Seed", created,
        ),
    )
    conn.commit()
    conn.close()
    return battle_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Remove seed-demo rows and exit",
    )
    args = parser.parse_args()

    db = BenchmarkDatabase()
    removed = _clear_seed_rows(db)
    if args.clear_only:
        print(f"Cleared seed-demo rows (approx {removed}).")
        return 0

    if ANCHOR not in config.TTS_PROVIDERS or COMPETITOR not in config.TTS_PROVIDERS:
        print(f"Expected anchor {ANCHOR} and competitor {COMPETITOR} in config.")
        return 1

    inserted = 0
    for i, (lang, outcome, days_ago, gender, voice, comment) in enumerate(VOTE_PLAN):
        if voice not in config.TTS_PROVIDERS[ANCHOR].supported_voices:
            pool = FALCON_BATTLE_VOICES.get(lang, {}).get(gender, [])
            voice = pool[0] if pool else voice
        text = ARENA_COMPARISON_TEXTS[i % len(ARENA_COMPARISON_TEXTS)]
        _insert_battle_and_vote(
            db,
            language=lang,
            outcome=outcome,
            days_ago=days_ago,
            gender=gender,
            voice=voice,
            comment=comment,
            text=text,
            item_idx=i,
        )
        inserted += 1

    counts = db.get_vote_counts()
    print(
        f"Inserted {inserted} demo battles + votes "
        f"(anchor={ANCHOR}, competitor={COMPETITOR})."
    )
    print(f"DB totals: {counts['votes']} votes, {counts['battles']} battles.")
    backend = "PostgreSQL" if db.use_postgres else "SQLite"
    print(f"Backend: {backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
