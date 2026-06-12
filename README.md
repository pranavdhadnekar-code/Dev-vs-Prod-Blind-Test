# Murf Voice Arena

A multi-provider **blind** Text-to-Speech listening test. Every battle pits the
anchor (**Omni / NewModel**) against one competitor for a chosen language. Both
clips are loudness- and format-normalized so nothing but voice quality is
audible, and votes feed a **Bradley-Terry + Davidson** rating engine with
bootstrap confidence intervals.

## Why this design

- **Blindness integrity.** Providers return different codecs/containers (Murf
  MP3, Omni WAV, …). Every served clip is decoded, **loudness-normalized to a
  common LUFS target**, and re-encoded to one **identical container/codec/
  sample-rate** (24 kHz mono PCM WAV) with metadata stripped — so a rater can't
  fingerprint a provider by format.
- **Principled ratings (not Elo).** Elo is order-dependent, needs a K-factor,
  and gives no confidence interval. We batch-fit Bradley-Terry with a Davidson
  tie term by maximum likelihood (`scipy` L-BFGS-B), pin the anchor to 0 for
  identifiability, and get **bootstrap CIs**. Deterministic given the same votes
  + config (fixed seeds).
- **Anchored, auditable comparisons.** Default `anchor_only` topology measures
  Omni vs. each competitor directly. Competitor-vs-competitor standings are
  transitively inferred through Omni and **flagged as model-inferred**.

## Architecture

| Module | Responsibility |
| --- | --- |
| `config.py` | Provider + language registry, representative voice per (provider, language, gender), anchor/pairing config, business weights, normalization target, engine settings |
| `tts_providers.py` | Provider HTTP clients + generalized `TTSProviderFactory` (all providers) |
| `audio_norm.py` | Decode → LUFS-normalize → common WAV codec/rate → strip metadata |
| `corpus.py` + `voice_battle_corpus.tsv` | Per-language test items (stable ids) |
| `scheduler.py` | `PairingStrategy` (`anchor_only`/`anchor_plus`/`all_vs_all`), per-language filtering, seed-recorded Left/Right randomization, adaptive sampling |
| `rating_engine.py` | Bradley-Terry + Davidson MLE, bootstrap CIs, per-language + weighted overall, Elo-like display |
| `reporting.py` | Win-rate grid (Wilson CIs), leaderboards, overall ranking, method snapshot, export |
| `database.py` | `battles` / `votes` / `ratings_runs` (full clip→vote→rating lineage) |
| `app.py` | Streamlit UI: blind battle, leaderboard, comments, export |

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt    # needs ffmpeg on PATH for audio normalization
cp .env.example .env               # then fill in keys (see below)
streamlit run app.py               # or: python run.py  (port 8501)
```

Open http://localhost:8501.

### Requirements
- Python 3.9+
- **ffmpeg** on PATH (audio decoding/normalization). macOS: `brew install ffmpeg`.
- API keys (below).

## Configuration

The arena is "ready" once the **anchor** plus **at least one competitor** that
shares a language are configured. Set keys in `.env`:

```bash
OMNI_API_KEY=...                 # anchor (required)
OMNI_HOST=host:port              # or OMNI_BASE_URL=https://.../tts
MURF_API_KEY=...                 # competitors (any subset)
ELEVENLABS_API_KEY=...
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
SARVAM_API_KEY=...
OPENAI_API_KEY=...               # OpenAI TTS competitor + comment summaries
# Optional:
# ARENA_ANCHOR=omni_tts
# ARENA_PAIRING_STRATEGY=anchor_only   # | anchor_plus | all_vs_all
```

### Languages & providers

Languages: `en-US, en-IN, en-UK, hi-IN, bn-IN, ta-IN` (extensible in `config.py`).
Each provider declares the languages it supports and a **disclosed representative
voice** per (language, gender). Only providers that support the selected language
(and the anchor) are scheduled for it.

### Tuning knobs (`config.py`)
- `LANGUAGE_WEIGHTS` — business weights for the overall rating (equal by default).
- `ENGINE_CONFIG` — bootstrap samples/seed, CI percentiles, Elo display scale,
  `adaptive_sampling` + `target_ci_halfwidth_elo` (default ±10 display points).
- `NORMALIZATION` — LUFS target, sample rate, codec/container.

## Adding languages / items
Edit `voice_battle_corpus.tsv` (`<bcp-locale>\t<text>`, one item per row), add the
language to `LANGUAGES`/`LANGUAGE_TO_CORPUS`, and declare provider support +
representative voices in `PROVIDER_LANGUAGES`.

## Reproducibility & audit
- Each **battle** stores its item, seed, clip hashes, normalization params, and
  geo/session.
- Each **vote** stores the served Left/Right providers + outcome + de-anonymized
  comment.
- **Ratings runs** snapshot the fit (inputs hash + config + code version +
  per-provider strengths/CIs), so any published number is reconstructable.

## Deployment (Docker)
```bash
docker build -t voice-arena .
docker run -p 8501:8501 --env-file .env voice-arena
```

`legacy_app.py` is the previous two-way (Murf vs NewModel) app, kept for reference.
