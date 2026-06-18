"""Leaderboard + reporting for the Voice Arena.

Pure computation (no Streamlit) so it can be unit-tested and exported:
  * Omni-vs-each-competitor head-to-head win-rate grid (directly measured, Wilson CIs).
  * Per-language Bradley-Terry/Davidson leaderboards with bootstrap CIs.
  * Overall business-weighted ranking with CIs + statistically-tied marking.
  * A method/config/version snapshot attached to every export for reproducibility.

Under the default `anchor_only` topology only Omni-vs-competitor battles are
DIRECTLY measured; competitor-vs-competitor standings are transitively inferred
through Omni and are flagged as such (with their wider intervals).
"""
from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
from rating_engine import (
    RatingEngine,
    LanguageFit,
    ProviderRating,
    code_version,
    inputs_hash,
    statistically_tied,
)


# --- small stats -------------------------------------------------------------
def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (handles small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pair_tally(outcomes: List[Dict], p1: str, p2: str) -> Dict[str, int]:
    """Wins/ties between two providers from (provider_a=left, provider_b=right)."""
    p1_wins = p2_wins = ties = 0
    for o in outcomes:
        a, b, res = o["provider_a"], o["provider_b"], o["outcome"]
        if {a, b} != {p1, p2}:
            continue
        if res == "tie":
            ties += 1
        else:
            winner = a if res == "A" else b
            if winner == p1:
                p1_wins += 1
            else:
                p2_wins += 1
    return {"p1_wins": p1_wins, "p2_wins": p2_wins, "ties": ties,
            "n": p1_wins + p2_wins + ties}


def _outcome_triplet(row: Dict) -> Dict[str, str]:
    return {
        "provider_a": row["provider_a"],
        "provider_b": row["provider_b"],
        "outcome": row["outcome"],
    }


def _partition_by_language(rows: List[Dict], lang_key: str = "language") -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        lang = row.get(lang_key)
        if lang:
            out[lang].append(row)
    return dict(out)


def winrate_grid_from_outcomes(outcomes: List[Dict], anchor: str) -> List[Dict]:
    """Omni-vs-each-competitor preference rate (ties excluded from rate)."""
    competitors = sorted({
        (o["provider_a"] if o["provider_b"] == anchor else o["provider_b"])
        for o in outcomes
        if anchor in (o["provider_a"], o["provider_b"])
        and o["provider_a"] != o["provider_b"]
    } - {anchor})
    rows = []
    for comp in competitors:
        t = pair_tally(outcomes, anchor, comp)
        decisive = t["p1_wins"] + t["p2_wins"]
        rate = (t["p1_wins"] / decisive) if decisive else float("nan")
        lo, hi = wilson_ci(t["p1_wins"], decisive) if decisive else (float("nan"), float("nan"))
        rows.append({
            "competitor": comp,
            "competitor_name": _name(comp),
            "anchor_preferred": t["p1_wins"],
            "competitor_preferred": t["p2_wins"],
            "ties": t["ties"],
            "n": t["n"],
            "anchor_win_rate": rate,
            "ci_low": lo,
            "ci_high": hi,
            "directly_measured": True,
        })
    return rows


def anchor_voice_winrate_from_rows(voice_rows: List[Dict], anchor: str) -> List[Dict]:
    """Per-anchor-voice preference rate vs. all competitors."""
    agg: Dict[str, Dict] = {}
    for r in voice_rows:
        lp, rp = r["left_provider"], r["right_provider"]
        if anchor not in (lp, rp) or lp == rp:
            continue
        if lp == anchor:
            voice, anchor_side, competitor = r["left_voice"], "A", rp
        else:
            voice, anchor_side, competitor = r["right_voice"], "B", lp
        if not voice:
            continue
        cell = agg.setdefault(
            voice, {"wins": 0, "losses": 0, "ties": 0, "losses_by": {}})
        outcome = r["outcome"]
        if outcome == "tie":
            cell["ties"] += 1
        elif outcome == anchor_side:
            cell["wins"] += 1
        else:
            cell["losses"] += 1
            cell["losses_by"][competitor] = cell["losses_by"].get(competitor, 0) + 1

    out = []
    for voice, c in agg.items():
        decisive = c["wins"] + c["losses"]
        rate = (c["wins"] / decisive) if decisive else float("nan")
        lo, hi = wilson_ci(c["wins"], decisive) if decisive else (float("nan"), float("nan"))
        loses_to = sorted(c["losses_by"].items(), key=lambda kv: (-kv[1], kv[0]))
        out.append({
            "voice": voice,
            "voice_name": _voice_label(anchor, voice),
            "gender": config.get_voice_gender(anchor, voice),
            "anchor_preferred": c["wins"],
            "competitor_preferred": c["losses"],
            "ties": c["ties"],
            "n": decisive + c["ties"],
            "anchor_win_rate": rate,
            "ci_low": lo,
            "ci_high": hi,
            "loses_to": [
                {"competitor": comp, "competitor_name": _name(comp), "losses": cnt}
                for comp, cnt in loses_to
            ],
            "loses_to_label": ", ".join(f"{_name(comp)} ({cnt})" for comp, cnt in loses_to) or "—",
            "directly_measured": True,
        })
    out.sort(key=lambda r: (-r["n"], r["voice"]))
    return out


def _heatmap_cells(
    winrate_by_lang: Dict[str, List[Dict]],
    lang_display,
) -> List[Dict]:
    cells = []
    for lang, grid in sorted(winrate_by_lang.items()):
        label = lang_display(lang)
        for row in grid:
            decisive = row["anchor_preferred"] + row["competitor_preferred"]
            rate = (row["anchor_preferred"] / decisive) if decisive else 0.5
            cells.append({
                "language": label,
                "competitor": row["competitor_name"],
                "win_rate": rate,
                "n": row["n"],
            })
    return cells


@dataclass(frozen=True)
class LeaderboardBundle:
    """Precomputed, cache-friendly leaderboard payload (no DB, no numpy)."""
    anchor: str
    languages: Tuple[str, ...]
    fits_meta: Dict[str, Dict[str, int]]
    head_to_head: Tuple[Dict, ...]
    winrate_by_lang: Dict[str, Tuple[Dict, ...]]
    heatmap_cells: Tuple[Dict, ...]
    overall_rows: Tuple[Dict, ...]
    voice_by_lang: Dict[str, Tuple[Dict, ...]]
    language_weights: Dict[str, float]


def build_leaderboard_bundle(
    outcomes_with_lang: List[Dict],
    voice_with_lang: List[Dict],
    engine: Optional[RatingEngine] = None,
) -> LeaderboardBundle:
    """Fit ratings and aggregate all leaderboard series from one in-memory fetch."""
    engine = engine or RatingEngine()
    anchor = engine.anchor

    outcomes_by_lang = _partition_by_language(outcomes_with_lang)
    voice_by_lang_raw = _partition_by_language(voice_with_lang)

    fits: Dict[str, LanguageFit] = {}
    for lang, lang_outcomes in sorted(outcomes_by_lang.items()):
        triplets = [_outcome_triplet(o) for o in lang_outcomes if o.get("provider_a") and o.get("provider_b")]
        if triplets:
            fits[lang] = engine.fit_language(triplets, lang)

    if not fits:
        return LeaderboardBundle(
            anchor=anchor,
            languages=tuple(),
            fits_meta={},
            head_to_head=tuple(),
            winrate_by_lang={},
            heatmap_cells=tuple(),
            overall_rows=tuple(),
            voice_by_lang={},
            language_weights={},
        )

    all_triplets = [
        _outcome_triplet(o)
        for o in outcomes_with_lang
        if o.get("provider_a") and o.get("provider_b")
    ]
    weights = config.language_weights(list(fits.keys()))
    overall = engine.fit_overall(fits, weights)
    report = ArenaReport(object(), engine=engine)
    overall_rows = report.overall_leaderboard(overall)

    winrate_by_lang = {
        lang: winrate_grid_from_outcomes(
            [_outcome_triplet(o) for o in outcomes_by_lang.get(lang, [])],
            anchor,
        )
        for lang in fits
    }
    voice_by_lang = {
        lang: anchor_voice_winrate_from_rows(voice_by_lang_raw.get(lang, []), anchor)
        for lang in fits
    }

    return LeaderboardBundle(
        anchor=anchor,
        languages=tuple(fits.keys()),
        fits_meta={
            lang: {"n_comparisons": fit.n_comparisons, "n_ties": fit.n_ties}
            for lang, fit in fits.items()
        },
        head_to_head=tuple(winrate_grid_from_outcomes(all_triplets, anchor)),
        winrate_by_lang={k: tuple(v) for k, v in winrate_by_lang.items()},
        heatmap_cells=tuple(_heatmap_cells(
            {k: list(v) for k, v in winrate_by_lang.items()},
            config.get_language_display,
        )),
        overall_rows=tuple(overall_rows),
        voice_by_lang={k: tuple(v) for k, v in voice_by_lang.items()},
        language_weights=weights,
    )


def leaderboard_cache_key(votes: int, battles: int, outcomes_with_lang: List[Dict]) -> str:
    triplets = [_outcome_triplet(o) for o in outcomes_with_lang if o.get("provider_a") and o.get("provider_b")]
    return f"{votes}:{battles}:{inputs_hash(triplets)}"


# --- orchestration -----------------------------------------------------------
class ArenaReport:
    def __init__(self, db, engine: Optional[RatingEngine] = None):
        self.db = db
        self.engine = engine or RatingEngine()
        self.anchor = self.engine.anchor

    def languages(self) -> List[str]:
        return self.db.get_languages_with_votes()

    def fit_all(self, outcomes_by_lang: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, LanguageFit]:
        fits: Dict[str, LanguageFit] = {}
        if outcomes_by_lang is not None:
            for lang, outcomes in sorted(outcomes_by_lang.items()):
                if outcomes:
                    fits[lang] = self.engine.fit_language(outcomes, lang)
            return fits
        for lang in self.languages():
            outcomes = self.db.get_outcomes(lang)
            if outcomes:
                fits[lang] = self.engine.fit_language(outcomes, lang)
        return fits

    def overall(self, fits: Optional[Dict[str, LanguageFit]] = None) -> Dict[str, ProviderRating]:
        fits = fits if fits is not None else self.fit_all()
        weights = config.language_weights(list(fits.keys()))
        return self.engine.fit_overall(fits, weights)

    # -- win-rate grid (directly measured) --
    def winrate_grid(
        self,
        language: Optional[str] = None,
        outcomes: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Omni-vs-each-competitor preference rate (ties excluded from rate)."""
        if outcomes is None:
            outcomes = self.db.get_outcomes(language)
        return winrate_grid_from_outcomes(outcomes, self.anchor)

    def anchor_voice_winrate(
        self,
        language: Optional[str] = None,
        voice_rows: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Per-anchor-voice preference rate vs. all competitors (ties excluded
        from the rate). Directly measured from the served clips."""
        if voice_rows is None:
            voice_rows = self.db.get_voice_outcomes(language)
        return anchor_voice_winrate_from_rows(voice_rows, self.anchor)

    # -- per-language leaderboard --
    def language_leaderboard(self, fit: LanguageFit) -> List[Dict]:
        rows = []
        ranked = sorted(fit.ratings.values(), key=lambda r: r.elo, reverse=True)
        for r in ranked:
            is_anchor = (r.provider_id == self.anchor)
            rows.append({
                "provider": r.provider_id,
                "provider_name": _name(r.provider_id),
                "is_anchor": is_anchor,
                "elo": r.elo,
                "elo_ci_low": r.elo_ci[0],
                "elo_ci_high": r.elo_ci[1],
                "strength": r.strength,
                "n_comparisons": r.n_comparisons,
                # vs the anchor this is directly measured; ordering among
                # competitors is inferred transitively through Omni.
                "measurement": "anchor" if is_anchor else "direct_vs_anchor",
            })
        return rows

    def overall_leaderboard(self, overall: Dict[str, ProviderRating]) -> List[Dict]:
        ranked = sorted(overall.values(), key=lambda r: r.elo, reverse=True)
        # statistically-tied grouping by overlapping CIs (adjacent).
        rows = []
        for i, r in enumerate(ranked):
            tied_with = [
                ranked[j].provider_id for j in range(len(ranked))
                if j != i and statistically_tied(r, ranked[j])
            ]
            rows.append({
                "rank": i + 1,
                "provider": r.provider_id,
                "provider_name": _name(r.provider_id),
                "is_anchor": r.provider_id == self.anchor,
                "elo": r.elo,
                "elo_ci_low": r.elo_ci[0],
                "elo_ci_high": r.elo_ci[1],
                "n_comparisons": r.n_comparisons,
                "statistically_tied_with": tied_with,
            })
        return rows

    # -- inferred competitor-vs-competitor matrix (flagged) --
    def inferred_matrix(self, fit: LanguageFit) -> Dict:
        """Pairwise predicted P(row preferred over col) from fitted strengths.

        Cells NOT involving the anchor are model-inferred (flagged).
        """
        provs = fit.providers
        cells = {}
        for a in provs:
            for b in provs:
                if a == b:
                    continue
                ta = fit.ratings[a].strength
                tb = fit.ratings[b].strength
                nu = fit.tie_param
                pa, pb = math.exp(ta), math.exp(tb)
                s = nu * math.sqrt(pa * pb)
                denom = pa + pb + s
                cells[f"{a}|{b}"] = {
                    "p_row_preferred": pa / denom,
                    "p_tie": s / denom,
                    "inferred": self.anchor not in (a, b),
                }
        return cells

    # -- method/config snapshot --
    def method_snapshot(self, fits: Optional[Dict[str, LanguageFit]] = None) -> Dict:
        fits = fits if fits is not None else self.fit_all()
        langs = list(fits.keys())
        return {
            "code_version": code_version(),
            "generated_at": datetime.now().isoformat(),
            "anchor": self.anchor,
            "pairing_strategy": config.PAIRING_STRATEGY,
            "engine_params": {
                "model": "bradley_terry_davidson",
                "solver": "scipy.optimize L-BFGS-B (MLE)",
                "anchor_strength": config.ENGINE_CONFIG.get("anchor_strength", 0.0),
                "regularization": config.ENGINE_CONFIG.get("regularization"),
                "bootstrap_samples": self.engine.n_boot,
                "bootstrap_seed": self.engine.seed,
                "ci_percentiles": self.engine.ci_pct,
                "elo_display_anchor": self.engine.elo_anchor,
                "elo_display_scale": self.engine.scale,
                "adaptive_sampling": config.ENGINE_CONFIG.get("adaptive_sampling"),
                "target_ci_halfwidth_elo": config.ENGINE_CONFIG.get("target_ci_halfwidth_elo"),
            },
            "language_weights": config.language_weights(langs),
            "normalization": config.NORMALIZATION,
        }

    # -- adaptive sampling weights (closeness + uncertainty) --
    def adaptive_uncertainty(self, language: str) -> Dict[str, float]:
        """Per-competitor sampling weight: larger when the Omni-vs-competitor CI
        is wider than the target and/or the matchup is close to the anchor."""
        outcomes = self.db.get_outcomes(language)
        if not outcomes:
            return {}
        fit = self.engine.fit_language(outcomes, language)
        target = float(config.ENGINE_CONFIG.get("target_ci_width_elo", 20.0))
        weights = {}
        for pid, r in fit.ratings.items():
            if pid == self.anchor:
                continue
            ci_width = r.elo_ci[1] - r.elo_ci[0]
            closeness = 1.0 / (1.0 + abs(r.elo - self.engine.elo_anchor) / 100.0)
            weights[pid] = max(ci_width - target, 0.0) + 1.0 + closeness
        return weights

    # -- full export --
    def build_export(self) -> Dict:
        fits = self.fit_all()
        overall = self.overall(fits)
        export = {
            "snapshot": self.method_snapshot(fits),
            "overall": self.overall_leaderboard(overall),
            "per_language": {},
            "head_to_head": {"overall": self.winrate_grid(None)},
            "anchor_voice_winrate": {"overall": self.anchor_voice_winrate(None)},
            "counts": self.db.get_vote_counts(),
        }
        for lang, fit in fits.items():
            export["per_language"][lang] = {
                "leaderboard": self.language_leaderboard(fit),
                "winrate_grid": self.winrate_grid(lang),
                "tie_param": fit.tie_param,
                "n_comparisons": fit.n_comparisons,
                "n_ties": fit.n_ties,
                "converged": fit.converged,
            }
        return export

    def persist_runs(self) -> str:
        """Snapshot every per-language + overall fit into ratings_runs (audit)."""
        fits = self.fit_all()
        overall = self.overall(fits)
        run_id = str(uuid.uuid4())
        snap = self.method_snapshot(fits)
        for lang, fit in fits.items():
            outcomes = self.db.get_outcomes(lang)
            self.db.save_ratings_run(
                run_id=run_id, scope="language", language=lang,
                inputs_hash=inputs_hash(outcomes), code_version=snap["code_version"],
                engine_params=snap["engine_params"],
                results={p: r.to_dict() for p, r in fit.ratings.items()},
                n_battles=fit.n_comparisons, n_votes=fit.n_comparisons,
            )
        all_outcomes = self.db.get_outcomes(None)
        self.db.save_ratings_run(
            run_id=run_id, scope="overall", language=None,
            inputs_hash=inputs_hash(all_outcomes), code_version=snap["code_version"],
            engine_params={**snap["engine_params"], "language_weights": snap["language_weights"]},
            results={p: r.to_dict() for p, r in overall.items()},
            n_battles=len(all_outcomes), n_votes=len(all_outcomes),
        )
        return run_id


def _name(provider_id: str) -> str:
    cfg = config.TTS_PROVIDERS.get(provider_id)
    return cfg.name if cfg else provider_id


def _voice_label(provider_id: str, voice_id: str) -> str:
    """Friendly display name for a voice id (falls back to the raw id)."""
    cfg = config.TTS_PROVIDERS.get(provider_id)
    if cfg:
        info = cfg.voice_info.get(voice_id)
        if info and info.name:
            return info.name
    return voice_id
