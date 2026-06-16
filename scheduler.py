"""Battle scheduler + pairing strategies for the Voice Arena.

Every battle pits the anchor (Omni) against one competitor under the default
`anchor_only` star topology. The pairing logic lives behind a `PairingStrategy`
interface so `anchor_plus` / `all_vs_all` can be selected from config without a
rewrite.

Responsibilities:
  * Pick a matchup for a language (only providers that support the language and,
    by construction, the anchor).
  * Pick a corpus item (same text used for both clips).
  * Randomize Left/Right and RECORD THE SEED so the assignment is reproducible.
  * Optionally bias matchup selection adaptively toward the closest / most
    uncertain Omni-vs-competitor pairs (flag, default per ENGINE_CONFIG).
"""
from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set

import config
import corpus


class SchedulerError(RuntimeError):
    pass


# --- Pairing strategies ------------------------------------------------------
@dataclass(frozen=True)
class Matchup:
    provider_a: str
    provider_b: str
    is_anchor_pair: bool


class PairingStrategy(ABC):
    """Decides which provider pairs are eligible for a language."""

    name: str = "abstract"

    @abstractmethod
    def matchups(self, language: str, anchor: str, competitors: List[str]) -> List[Matchup]:
        ...


class AnchorOnlyStrategy(PairingStrategy):
    """Star topology: anchor vs. each competitor (the default, PRD-locked)."""

    name = "anchor_only"

    def matchups(self, language, anchor, competitors):
        return [Matchup(anchor, c, True) for c in competitors]


class AnchorPlusStrategy(PairingStrategy):
    """Anchor vs. each competitor PLUS competitor-vs-competitor pairs.

    Selectable stub: the interface admits it without a rewrite. Note the rating
    engine pins the anchor for identifiability; competitor-vs-competitor battles
    become directly-measured (rather than inferred) if this is enabled.
    """

    name = "anchor_plus"

    def matchups(self, language, anchor, competitors):
        ms = [Matchup(anchor, c, True) for c in competitors]
        for i in range(len(competitors)):
            for j in range(i + 1, len(competitors)):
                ms.append(Matchup(competitors[i], competitors[j], False))
        return ms


class AllVsAllStrategy(PairingStrategy):
    """Every unordered pair including the anchor. Selectable stub."""

    name = "all_vs_all"

    def matchups(self, language, anchor, competitors):
        everyone = [anchor] + list(competitors)
        ms: List[Matchup] = []
        for i in range(len(everyone)):
            for j in range(i + 1, len(everyone)):
                a, b = everyone[i], everyone[j]
                ms.append(Matchup(a, b, anchor in (a, b)))
        return ms


_STRATEGIES: Dict[str, PairingStrategy] = {
    s.name: s for s in (AnchorOnlyStrategy(), AnchorPlusStrategy(), AllVsAllStrategy())
}


def get_strategy(name: Optional[str] = None) -> PairingStrategy:
    name = name or config.PAIRING_STRATEGY
    return _STRATEGIES.get(name, AnchorOnlyStrategy())


# --- Battle plan -------------------------------------------------------------
@dataclass
class BattlePlan:
    """A scheduled (but not-yet-synthesized) battle.

    left/right are what the rater sees as Sample A / Sample B. The
    `position_seed` deterministically reproduces the side assignment from the
    (provider_a, provider_b) order, which is also stored.
    """
    battle_id: str
    language: str
    gender: str
    item_id: str
    item_text: str
    strategy: str
    anchor: str
    competitor: str
    is_anchor_pair: bool
    provider_a: str         # canonical matchup order (pre-randomization)
    provider_b: str
    left_provider: str
    left_voice: str
    right_provider: str
    right_voice: str
    position_seed: int

    def to_dict(self) -> Dict:
        return asdict(self)


def assign_sides(provider_a: str, provider_b: str, position_seed: int) -> bool:
    """Deterministic side assignment from a recorded seed.

    Returns True if (provider_a, provider_b) were swapped so provider_b is on
    the Left. Pure function => reproducible audit of any stored battle.
    """
    return random.Random(position_seed).random() < 0.5


# --- Scheduler ---------------------------------------------------------------
class Scheduler:
    def __init__(
        self,
        strategy: Optional[PairingStrategy] = None,
        seed: Optional[int] = None,
        adaptive: Optional[bool] = None,
        target_ci_width: Optional[float] = None,
    ):
        self.strategy = strategy or get_strategy()
        self.master_seed = seed
        self._rng = random.Random(seed)
        self.adaptive = (
            config.ENGINE_CONFIG.get("adaptive_sampling", False)
            if adaptive is None else adaptive
        )
        self.target_ci_width = (
            config.ENGINE_CONFIG.get("target_ci_width_elo")
            if target_ci_width is None else target_ci_width
        )
        self.healthy_providers: Optional[Set[str]] = None
        # Round-robin rotation state (per language): remaining matchup keys for
        # the current round + the eligible set the round was built from.
        self._rotation: Dict[str, List[tuple]] = {}
        self._rotation_pool: Dict[str, Set[tuple]] = {}
        self._rotation_last: Dict[str, tuple] = {}

    def available_languages(self, configured_only: bool = True) -> List[str]:
        if self.healthy_providers is not None:
            anchor = config.anchor_provider()
            if anchor not in self.healthy_providers:
                return []
            out: List[str] = []
            for language in config.LANGUAGES:
                if not config.provider_supports_language(anchor, language):
                    continue
                comps = [
                    c for c in config.competitors_for_language(language, configured_only=configured_only)
                    if c in self.healthy_providers
                ]
                if comps:
                    out.append(language)
            return out
        return config.languages_available(configured_only=configured_only)

    def matchups_for_language(self, language: str, configured_only: bool = True) -> List[Matchup]:
        anchor = config.anchor_provider()
        competitors = config.competitors_for_language(language, configured_only=configured_only)
        if self.healthy_providers is not None:
            competitors = [c for c in competitors if c in self.healthy_providers]
        return self.strategy.matchups(language, anchor, competitors)

    @staticmethod
    def _matchup_key(m: Matchup) -> tuple:
        return (m.provider_a, m.provider_b, m.is_anchor_pair)

    def _select_matchup(
        self,
        language: str,
        matchups: List[Matchup],
        rng: random.Random,
    ) -> Matchup:
        """Round-robin matchup selection.

        Every eligible matchup (anchor vs each competitor) is served exactly once
        per round before any repeats, so providers rotate evenly instead of being
        starved by adaptive weighting. The within-round order is shuffled for
        fairness, and the round rebuilds automatically when the eligible set
        changes (e.g. a provider's health flips).
        """
        if not matchups:
            raise SchedulerError("No eligible matchups for this language.")

        by_key = {self._matchup_key(m): m for m in matchups}
        eligible = set(by_key)

        queue = [k for k in self._rotation.get(language, []) if k in eligible]
        if self._rotation_pool.get(language) != eligible or not queue:
            queue = list(eligible)
            rng.shuffle(queue)
            # Avoid an immediate repeat across the round boundary when possible.
            last = self._rotation_last.get(language)
            if len(queue) > 1 and queue[0] == last:
                queue.append(queue.pop(0))
            self._rotation_pool[language] = eligible

        key = queue.pop(0)
        self._rotation[language] = queue
        self._rotation_last[language] = key
        return by_key[key]

    def _voice_for(self, provider: str, language: str, gender: str) -> str:
        v = config.representative_voice(provider, language, gender)
        if v is None:
            cfg = config.TTS_PROVIDERS.get(provider)
            provider_label = cfg.name if cfg else provider
            lang_label = config.get_language_display(language)
            raise SchedulerError(
                f"{provider_label} does not support {gender} voices in {lang_label}."
            )
        return v

    def next_battle(
        self,
        language: str,
        gender: Optional[str] = None,
        item_id: Optional[str] = None,
        uncertainty: Optional[Dict[str, float]] = None,  # deprecated: ignored (round-robin)
        rng: Optional[random.Random] = None,
    ) -> BattlePlan:
        """Schedule one battle for `language` (round-robin over competitors)."""
        rng = rng or self._rng

        matchups = self.matchups_for_language(language)
        if not matchups:
            raise SchedulerError(
                f"Language '{language}' has no configured competitors for the anchor."
            )
        m = self._select_matchup(language, matchups, rng)

        chosen_gender = gender or rng.choice(["male", "female"])

        items = corpus.get_items(language)
        if not items:
            raise SchedulerError(f"No corpus items for language '{language}'.")
        item = corpus.get_item(language, item_id) if item_id else None
        if item is None:
            item = rng.choice(items)

        va = self._voice_for(m.provider_a, language, chosen_gender)
        vb = self._voice_for(m.provider_b, language, chosen_gender)

        position_seed = rng.getrandbits(31)
        swap = assign_sides(m.provider_a, m.provider_b, position_seed)
        if swap:
            left_p, left_v, right_p, right_v = m.provider_b, vb, m.provider_a, va
        else:
            left_p, left_v, right_p, right_v = m.provider_a, va, m.provider_b, vb

        anchor = config.anchor_provider()
        if m.is_anchor_pair:
            competitor = m.provider_b if m.provider_a == anchor else m.provider_a
        else:
            competitor = m.provider_b

        return BattlePlan(
            battle_id=str(uuid.uuid4()),
            language=language,
            gender=chosen_gender,
            item_id=item.id,
            item_text=item.text,
            strategy=self.strategy.name,
            anchor=anchor,
            competitor=competitor,
            is_anchor_pair=m.is_anchor_pair,
            provider_a=m.provider_a,
            provider_b=m.provider_b,
            left_provider=left_p,
            left_voice=left_v,
            right_provider=right_p,
            right_voice=right_v,
            position_seed=position_seed,
        )

    def round_for_language(
        self,
        language: str,
        gender: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> List[BattlePlan]:
        """One battle per eligible matchup (a full sweep for the language)."""
        rng = rng or self._rng
        plans: List[BattlePlan] = []
        for m in self.matchups_for_language(language):
            # Reuse next_battle's logic per matchup by temporarily constraining.
            chosen_gender = gender or rng.choice(["male", "female"])
            items = corpus.get_items(language)
            if not items:
                raise SchedulerError(f"No corpus items for language '{language}'.")
            item = rng.choice(items)
            va = self._voice_for(m.provider_a, language, chosen_gender)
            vb = self._voice_for(m.provider_b, language, chosen_gender)
            position_seed = rng.getrandbits(31)
            swap = assign_sides(m.provider_a, m.provider_b, position_seed)
            if swap:
                left_p, left_v, right_p, right_v = m.provider_b, vb, m.provider_a, va
            else:
                left_p, left_v, right_p, right_v = m.provider_a, va, m.provider_b, vb
            anchor = config.anchor_provider()
            competitor = (m.provider_b if m.provider_a == anchor else m.provider_a) \
                if m.is_anchor_pair else m.provider_b
            plans.append(BattlePlan(
                battle_id=str(uuid.uuid4()),
                language=language,
                gender=chosen_gender,
                item_id=item.id,
                item_text=item.text,
                strategy=self.strategy.name,
                anchor=anchor,
                competitor=competitor,
                is_anchor_pair=m.is_anchor_pair,
                provider_a=m.provider_a,
                provider_b=m.provider_b,
                left_provider=left_p,
                left_voice=left_v,
                right_provider=right_p,
                right_voice=right_v,
                position_seed=position_seed,
            ))
        return plans
