# Coding-agent task: rebuild this app as a multi-provider blind "Voice Arena"

## Context

This repo (`Listening-tool-main`) is currently a **two-way** blind A/B listening test:
**Murf Gen 2** vs **NewModel** (`omni_tts`, internally "Omni"), built in Streamlit with an
online **Elo** rating engine and a SQLite store. I want you to **rebuild it cleanly** into a
**multi-provider** blind benchmark ("Voice Arena") that follows the attached PRD
(`PRD - Murf Voice Arena`). Treat the existing code as a **reference implementation to mine**,
not something to preserve — but **reuse** the working HTTP provider logic, the sentence corpus,
and the Streamlit UX patterns wherever they're sound.

The full PRD is the source of truth. The decisions below override or pin anything ambiguous in it.

### Locked decisions (do not re-litigate)
1. **Anchor = Omni / NewModel.** Every battle is `Omni vs. one competitor` (`anchor_only` star
   topology). Murf Gen 2 becomes *just another competitor*. Omni's strength is pinned to 0 for
   identifiability. Implement anchor as a single config value behind a pairing-strategy interface
   so it can be swapped later without code changes.
2. **Enable all existing providers.** Wire up every provider class already in `tts_providers.py`
   (Murf Gen2, ElevenLabs Flash, ElevenLabs v3, Deepgram, Deepgram Aura2, Cartesia Sonic2/Turbo/
   Sonic3, OpenAI, Sarvam, Sarvam Bulbul v3) plus Omni. Each must declare which **languages** it
   supports and a **representative voice per (language, gender)**.
3. **Rebuild cleanly.** You may restructure files, the DB schema, and the engine. Keep it a
   runnable Streamlit app launched the same way (`run.py` / `streamlit run app.py`, port 8501),
   env-var driven, Docker-deployable.

---

## What's wrong with the current app (fix these explicitly)

- **Format artifact breaks blindness.** Murf returns **MP3**, Omni returns **WAV**, other
  providers vary. A rater (or their browser) can distinguish providers by codec/container. The
  PRD requires every served clip to be **loudness-normalized to a common LUFS target** and
  **identical container/codec/sample-rate**. This is mandatory, not optional.
- **Elo is the wrong engine.** Replace it entirely (see Rating engine below). Per PRD §5: Elo is
  order-dependent, needs a K-factor, gives no confidence interval, and is path-dependent.
- **Two-provider hardcoding everywhere.** `config.py` conflates the two providers onto one shared
  Murf voice catalog; `TTSProviderFactory` only instantiates `murf_gen2` and `omni_tts`; `app.py`
  has `_2`-suffixed duplicate flows (`generate_next_comparison_2`, `handle_vote_2`, etc.). All of
  this must generalize to N providers.
- **No tie vote.** Current UI is A/B only. The PRD requires **Left better / Right better /
  About the same** (ties feed the Davidson model).
- **No per-language provider support.** Providers support **different** language sets (with
  overlap). The arena must only pit providers that *both* support a given language, and Omni must
  support it too (it's the anchor).

---

## Functional requirements

### 1. Provider & voice registry (replaces the shared catalog in `config.py`)
Build a clean, config-driven registry. For each provider, declare:
- `provider_id`, display name, the existing `generate_speech` implementation to reuse.
- **Supported languages** (BCP-47-style locale keys; reuse the current set as a starting point:
  `en-US, en-IN, en-UK, hi-IN, bn-IN, ta-IN`, extensible).
- For each supported language: a **designated representative voice for male and for female**,
  matched on gender/style, **disclosed** (store the mapping so it can be published).
- Keep the existing Murf↔Omni voice-id mapping concept where a provider needs an internal id
  translation (see `MURF_TO_OMNI_VOICE` in `config.py`).

Generalize `TTSProviderFactory` so `create_provider` / `create_all_providers` cover **all**
registered providers (today it hard-codes only two). Note: some providers (ElevenLabs, Cartesia)
fetch voices from their API at runtime — preserve that, but pin the *representative* voice per
(language, gender) so battles are reproducible.

### 2. Corpus
Keep the bundled per-language corpus pattern (`voice_battle_corpus.tsv` +
`voice_battle_locale_defaults.py`). PRD wants **10–20 test items per language** covering everyday
speech **and hard cases** (numbers, dates, names, acronyms). The **same item text** is used for
both clips in a battle. Make it easy to add languages/items by editing the TSV.

### 3. Pairing / scheduler (behind an interface)
- Implement a `PairingStrategy` interface. Ship `anchor_only` (every battle = Omni vs. competitor)
  as the default mode; leave `anchor_plus` and `all_vs_all` as config-selectable stubs that the
  interface admits without a rewrite.
- For a chosen language, only schedule competitors that **support that language** (and Omni must
  support it).
- **Position randomization**: randomize Left/Right per battle and **record the seed** with the
  battle for reproducibility.
- Support **optional adaptive sampling**: allocate more battles to the closest / most uncertain
  Omni-vs-competitor matchups (can be a flag, default off).

### 4. Audio normalization pipeline (blindness integrity)
Before a clip reaches the rater: decode → **loudness-normalize to a common LUFS target** →
re-encode to a **single common container/codec/sample-rate** for *all* providers. Strip any
provider-identifying metadata. Use a standard lib (e.g. `pyloudnorm` + `ffmpeg`/`pydub`). Add the
dependency to `requirements.txt` and the Dockerfile.

### 5. Voting UI
- Present two **anonymized, brand-free** clips with controls: **Left better / Right better /
  About the same**. No format/metadata/branding leakage to the UI.
- Rater guidance: judge **overall quality** (naturalness, expressiveness, pace) **including
  pronunciation accuracy**.
- Record each vote as a battle row (below). Keep the "must play both clips before voting" guard
  from the current app and the optional free-text comment, with de-anonymization-on-store
  (see `de_anonymize_comment` in `app.py`).

### 6. Rating engine (replace Elo — PRD §5)
- **Bradley-Terry with a Davidson tie extension**, fit by **maximum likelihood** (batch over all
  votes), one global tie parameter. Use a standard solver (e.g. `scipy.optimize` L-BFGS).
- **Per language**: fit over that language's battle table `(provider_a, provider_b, outcome ∈
  {A, B, tie})`. Pin **Omni = 0** for identifiability.
- **Confidence intervals**: **bootstrap** — resample battles with replacement, refit a few hundred
  times, take per-provider percentiles. **Fixed seeds** → deterministic, reproducible.
- **Overall rating**: business-weighted average of per-language, Omni-anchored strengths, with
  **disclosed weights** (config). 
- **Display**: linearly rescale fitted strengths onto an Elo-like scale for readability.
- The PRD says a reference implementation of BT + Davidson + bootstrap + aggregation already
  exists and can seed the engine — ask me for it / look for it before writing from scratch.
- Engine must be **deterministic** given the same vote set + config (fixed seeds), and live in its
  own module decoupled from the UI.

### 7. Leaderboard & reporting
- **Omni-vs-each-competitor head-to-head win-rate grid** (% preferred, with CIs) — the primary,
  directly-measured artifact.
- **Per-language leaderboards** with ratings + confidence bands. Competitor-vs-competitor
  positions are *transitively inferred* through Omni — **visibly flag them as model-inferred**
  with their (wider) intervals.
- **Overall ranking** with disclosed weights + confidence bands; mark providers in overlapping
  intervals as **statistically tied**.
- **Structured export** (per-language, overall, head-to-head) for Marketing, with the **active
  method/config snapshot attached** (engine params, weights, seeds, code version).

### 8. Data model (rebuild the SQLite schema)
Design for **auditability & reproducibility** (PRD §6): a leaderboard cell must trace back to
battles → votes → the exact clips/config that produced them. Suggested core tables:
- `battles`: id, language, provider_a, provider_b, item_id/text, left/right assignment, **seed**,
  clip refs/hashes, normalization params, timestamp, session/geo.
- `votes`: battle_id, outcome ∈ {A, B, tie}, comment, rater/session id.
- `ratings_runs`: a versioned snapshot of a fit (inputs hash + config + code version + per-provider
  strengths + CIs), so every published number is reconstructable.
Drop the online `elo_ratings` table (or keep read-only for migration). Reuse the geolocation and
rate-limiting modules as-is.

---

## Non-functional requirements (PRD §6)
- **Reproducibility**: any published number reconstructable from stored votes + config + code
  version.
- **Auditability**: full lineage from a leaderboard cell back to clips, battles, votes.
- **Blindness integrity**: no provider-identifying metadata, branding, or format artifact reaches
  the rater UI (this is why §4 normalization is mandatory).

---

## Suggested deliverables / structure
- `config.py` (or a `registry/` module): provider+language+representative-voice registry, anchor
  config, business weights, normalization target, pairing-strategy selection.
- `tts_providers.py`: keep/clean the provider classes; generalize the factory to all providers.
- `audio_norm.py`: LUFS + codec/container/sample-rate normalization.
- `scheduler.py`: `PairingStrategy` interface + `anchor_only` impl + position randomization + seed
  recording + optional adaptive sampling.
- `rating_engine.py`: Bradley-Terry + Davidson MLE, bootstrap CIs, per-language + overall, Elo-like
  display, deterministic.
- `database.py`: new schema (battles / votes / ratings_runs).
- `app.py`: N-provider blind battle flow (collapse the `_2` duplication), tie-capable voting UI,
  leaderboard (win-rate grid + per-language + overall, with CIs and model-inferred flags), export.
- Update `requirements.txt`, `Dockerfile`, `README.md`, `.env.example`.

## Constraints & working style
- Keep it runnable at every step; land it incrementally (registry → providers/factory →
  normalization → scheduler → voting → engine → reporting). Don't break launch.
- Env-var driven config; no hardcoded API keys. Validate that the anchor (Omni) and at least one
  competitor are configured before a language is offered.
- Preserve existing strengths: geolocation tagging, rate limiting, "play both before voting",
  comment capture + de-anonymization, the OpenAI per-language comment summary.
- Before writing the rating engine, **find and reuse the existing BT/Davidson reference
  implementation** the PRD mentions.
- Ask me about the three deferred items in PRD §7 before hardcoding them: **sampling allocation
  / target CI widths**, **overall per-language weights**, and the **launch language + competitor
  set**.

## Acceptance criteria
- A blind battle serves two normalized, indistinguishable-by-format clips (Omni vs. a competitor)
  for a chosen language, with randomized sides and a recorded seed.
- Only providers supporting the selected language (incl. Omni) are scheduled for it.
- Votes capture Left/Right/Tie; the engine produces per-language and overall ratings with bootstrap
  CIs, Omni pinned to 0, on an Elo-like scale, deterministically.
- Leaderboard shows the Omni-vs-each win-rate grid with CIs, per-language boards, an overall
  ranking with disclosed weights, and flags inferred (competitor-vs-competitor) standings.
- Results export with a full method/config/version snapshot; any cell is traceable to its votes.
