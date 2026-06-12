"""Rating engine: Bradley-Terry with a Davidson tie extension.

Replaces the old online Elo (order-dependent, K-factor-sensitive, no CIs). This
engine batch-fits all votes by maximum likelihood, is deterministic given the
same vote set + config (fixed seeds), and lives decoupled from the UI.

Model (Davidson 1970 tie extension of Bradley-Terry)
----------------------------------------------------
Each provider i has a latent strength theta_i (log-scale "worth" pi_i=exp(theta_i)).
For a comparison between i and j with a single global tie parameter nu >= 0:

    P(i preferred) = pi_i / (pi_i + pi_j + nu*sqrt(pi_i*pi_j))
    P(j preferred) = pi_j / (pi_i + pi_j + nu*sqrt(pi_i*pi_j))
    P(tie)         = nu*sqrt(pi_i*pi_j) / (pi_i + pi_j + nu*sqrt(pi_i*pi_j))

We fit {theta_i} (anchor pinned to 0 for identifiability) and log(nu) by L-BFGS-B
on the negative log-likelihood with a tiny ridge for stability.

Confidence intervals
---------------------
Nonparametric bootstrap: resample comparisons with replacement, refit, take
per-provider percentiles. Seeds are fixed => reproducible. Overall (cross-language)
CIs propagate per-language bootstrap draws through the disclosed business weights.

Display
-------
Strengths are linearly rescaled to an Elo-like scale: elo = anchor + scale*theta,
with scale = 400/ln(10) so a logistic strength gap maps to the familiar Elo gap.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

import config

_OUTCOME_A = 0
_OUTCOME_B = 1
_OUTCOME_TIE = 2

_THETA_BOUND = 20.0          # keep exp(theta) finite under separation
_LOGNU_BOUNDS = (-20.0, 20.0)


# --- Results ----------------------------------------------------------------
@dataclass
class ProviderRating:
    provider_id: str
    strength: float                 # theta (anchor = 0)
    elo: float                      # display scale
    strength_ci: Tuple[float, float]
    elo_ci: Tuple[float, float]
    n_comparisons: int

    def to_dict(self) -> Dict:
        return {
            "provider_id": self.provider_id,
            "strength": self.strength,
            "elo": self.elo,
            "strength_ci_low": self.strength_ci[0],
            "strength_ci_high": self.strength_ci[1],
            "elo_ci_low": self.elo_ci[0],
            "elo_ci_high": self.elo_ci[1],
            "elo_ci_halfwidth": (self.elo_ci[1] - self.elo_ci[0]) / 2.0,
            "n_comparisons": self.n_comparisons,
        }


@dataclass
class LanguageFit:
    language: str
    providers: List[str]
    ratings: Dict[str, ProviderRating]
    tie_param: float
    n_comparisons: int
    n_ties: int
    converged: bool
    # bootstrap strength draws, shape (n_boot, n_providers), aligned to `providers`.
    bootstrap_strengths: np.ndarray = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "providers": self.providers,
            "ratings": {p: r.to_dict() for p, r in self.ratings.items()},
            "tie_param": self.tie_param,
            "n_comparisons": self.n_comparisons,
            "n_ties": self.n_ties,
            "converged": self.converged,
        }


# --- Engine -----------------------------------------------------------------
class RatingEngine:
    def __init__(self, anchor: Optional[str] = None, engine_config: Optional[Dict] = None):
        self.anchor = anchor or config.anchor_provider()
        self.cfg = dict(config.ENGINE_CONFIG)
        if engine_config:
            self.cfg.update(engine_config)
        self.scale = float(self.cfg.get("elo_display_scale", 400.0 / math.log(10.0)))
        self.elo_anchor = float(self.cfg.get("elo_display_anchor", 1000.0))
        self.reg = float(self.cfg.get("regularization", 1e-6))
        self.n_boot = int(self.cfg.get("bootstrap_samples", 400))
        self.seed = int(self.cfg.get("bootstrap_seed", 12345))
        self.ci_pct = list(self.cfg.get("ci_percentiles", [2.5, 97.5]))

    # -- elo display --
    def to_elo(self, theta: float) -> float:
        return self.elo_anchor + self.scale * float(theta)

    # -- data prep --
    def _prepare(self, outcomes: Sequence[Dict]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        """Return (providers, a_idx, b_idx, code) with the anchor at index 0."""
        present = []
        for o in outcomes:
            present.append(o["provider_a"])
            present.append(o["provider_b"])
        others = sorted(set(present) - {self.anchor})
        providers = [self.anchor] + others
        idx = {p: i for i, p in enumerate(providers)}

        a_idx, b_idx, code = [], [], []
        for o in outcomes:
            a_idx.append(idx[o["provider_a"]])
            b_idx.append(idx[o["provider_b"]])
            res = o["outcome"]
            code.append(_OUTCOME_TIE if res == "tie" else (_OUTCOME_A if res == "A" else _OUTCOME_B))
        return providers, np.array(a_idx), np.array(b_idx), np.array(code)

    # -- negative log-likelihood --
    def _nll(self, x: np.ndarray, n_providers: int, a_idx: np.ndarray,
             b_idx: np.ndarray, code: np.ndarray) -> float:
        theta = np.empty(n_providers)
        theta[0] = 0.0                         # anchor pinned
        theta[1:] = x[:-1]
        log_nu = x[-1]

        ta = theta[a_idx]
        tb = theta[b_idx]
        # log-sum-exp style denom: log(e^ta + e^tb + nu*e^((ta+tb)/2))
        tie_term = log_nu + (ta + tb) / 2.0
        m = np.maximum(np.maximum(ta, tb), tie_term)
        denom = m + np.log(np.exp(ta - m) + np.exp(tb - m) + np.exp(tie_term - m))

        ll = np.where(code == _OUTCOME_A, ta - denom,
             np.where(code == _OUTCOME_B, tb - denom, tie_term - denom))
        nll = -np.sum(ll)
        nll += self.reg * np.sum(theta[1:] ** 2)   # ridge on free strengths
        return nll

    def _fit_once(self, n_providers: int, a_idx: np.ndarray, b_idx: np.ndarray,
                  code: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool]:
        n_free = n_providers - 1
        if x0 is None:
            x0 = np.zeros(n_free + 1)
        bounds = [(-_THETA_BOUND, _THETA_BOUND)] * n_free + [_LOGNU_BOUNDS]
        res = minimize(
            self._nll, x0, args=(n_providers, a_idx, b_idx, code),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
        )
        theta = np.empty(n_providers)
        theta[0] = 0.0
        theta[1:] = res.x[:-1]
        nu = float(np.exp(res.x[-1]))
        return theta, nu, bool(res.success)

    # -- per-language fit + bootstrap --
    def fit_language(self, outcomes: Sequence[Dict], language: str = "") -> LanguageFit:
        outcomes = [o for o in outcomes if o.get("provider_a") and o.get("provider_b")]
        if not outcomes:
            providers = [self.anchor]
            ratings = {self.anchor: ProviderRating(
                self.anchor, 0.0, self.to_elo(0.0), (0.0, 0.0),
                (self.to_elo(0.0), self.to_elo(0.0)), 0)}
            return LanguageFit(language, providers, ratings, 0.0, 0, 0, True,
                               np.zeros((0, 1)))

        providers, a_idx, b_idx, code = self._prepare(outcomes)
        n_providers = len(providers)
        n_ties = int(np.sum(code == _OUTCOME_TIE))

        theta, nu, converged = self._fit_once(n_providers, a_idx, b_idx, code)

        n_comp = np.zeros(n_providers, dtype=int)
        for i in range(n_providers):
            n_comp[i] = int(np.sum(a_idx == i) + np.sum(b_idx == i))

        # bootstrap
        draws = self._bootstrap(n_providers, a_idx, b_idx, code, language)

        ratings: Dict[str, ProviderRating] = {}
        for i, p in enumerate(providers):
            if draws.shape[0] > 0:
                col = draws[:, i]
                lo, hi = np.percentile(col, self.ci_pct)
            else:
                lo, hi = theta[i], theta[i]
            ratings[p] = ProviderRating(
                provider_id=p,
                strength=float(theta[i]),
                elo=self.to_elo(theta[i]),
                strength_ci=(float(lo), float(hi)),
                elo_ci=(self.to_elo(lo), self.to_elo(hi)),
                n_comparisons=int(n_comp[i]),
            )

        return LanguageFit(language, providers, ratings, nu, len(outcomes),
                           n_ties, converged, draws)

    def _bootstrap(self, n_providers: int, a_idx: np.ndarray, b_idx: np.ndarray,
                   code: np.ndarray, language: str) -> np.ndarray:
        n = len(code)
        if n == 0 or self.n_boot <= 0:
            return np.zeros((0, n_providers))
        # Deterministic, language-specific seed.
        lang_seed = (self.seed + (int(hashlib.sha256(language.encode()).hexdigest(), 16) % 1_000_000)) % (2**32)
        rng = np.random.default_rng(lang_seed)
        out = np.zeros((self.n_boot, n_providers))
        for b in range(self.n_boot):
            sample = rng.integers(0, n, size=n)
            theta_b, _nu_b, _ok = self._fit_once(
                n_providers, a_idx[sample], b_idx[sample], code[sample])
            out[b] = theta_b
        return out

    # -- overall (business-weighted) --
    def fit_overall(self, fits: Dict[str, LanguageFit],
                    weights: Optional[Dict[str, float]] = None) -> Dict[str, ProviderRating]:
        """Weighted-average per-language strengths into one Omni-anchored rating.

        Weights are normalized per provider over the languages where that
        provider actually competed. CIs propagate the per-language bootstrap
        draws (aligned by draw index) through the weights.
        """
        langs = [l for l, f in fits.items() if f.n_comparisons > 0]
        if not langs:
            return {self.anchor: ProviderRating(
                self.anchor, 0.0, self.to_elo(0.0), (0.0, 0.0),
                (self.to_elo(0.0), self.to_elo(0.0)), 0)}

        w = weights or config.language_weights(langs)
        # number of bootstrap draws shared across languages
        n_boot = min((fits[l].bootstrap_strengths.shape[0] for l in langs
                      if fits[l].bootstrap_strengths is not None and fits[l].bootstrap_strengths.shape[0] > 0),
                     default=0)

        providers = sorted({p for l in langs for p in fits[l].providers})
        ratings: Dict[str, ProviderRating] = {}
        for p in providers:
            part_langs = [l for l in langs if p in fits[l].ratings]
            wl = {l: w.get(l, 0.0) for l in part_langs}
            wsum = sum(wl.values())
            if wsum <= 0:
                wl = {l: 1.0 / len(part_langs) for l in part_langs}
                wsum = 1.0

            point = sum(wl[l] * fits[l].ratings[p].strength for l in part_langs) / wsum

            if n_boot > 0:
                agg = np.zeros(n_boot)
                for l in part_langs:
                    pi = fits[l].providers.index(p)
                    agg += (wl[l] / wsum) * fits[l].bootstrap_strengths[:n_boot, pi]
                lo, hi = np.percentile(agg, self.ci_pct)
            else:
                lo, hi = point, point

            n_total = sum(fits[l].ratings[p].n_comparisons for l in part_langs)
            ratings[p] = ProviderRating(
                provider_id=p,
                strength=float(point),
                elo=self.to_elo(point),
                strength_ci=(float(lo), float(hi)),
                elo_ci=(self.to_elo(lo), self.to_elo(hi)),
                n_comparisons=int(n_total),
            )
        return ratings


# --- helpers for auditability ------------------------------------------------
def code_version() -> str:
    """Best-effort code version (git short SHA) for ratings-run snapshots."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=config.__file__.rsplit("/", 1)[0],
        ).decode().strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def inputs_hash(outcomes: Sequence[Dict]) -> str:
    """Stable hash of a vote set so a published number is reconstructable."""
    payload = json.dumps(
        [[o["provider_a"], o["provider_b"], o["outcome"]] for o in outcomes],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def statistically_tied(r1: ProviderRating, r2: ProviderRating) -> bool:
    """True if two providers' Elo CIs overlap (=> not separable at this level)."""
    return not (r1.elo_ci[1] < r2.elo_ci[0] or r2.elo_ci[1] < r1.elo_ci[0])
