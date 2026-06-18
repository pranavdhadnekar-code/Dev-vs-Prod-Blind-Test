"""Murf Voice Arena — multi-provider blind listening test (Streamlit).

Every battle pits the anchor (Falcon 2) against one competitor for a
chosen language. Both clips are loudness/format-normalized so nothing but voice
quality distinguishes them. Votes (Left / Right / About the same) feed a
rating engine with bootstrap confidence intervals.
"""
import asyncio
import csv
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime

import streamlit as st
import pandas as pd
from streamlit_advanced_audio import audix, WaveSurferOptions

from dotenv import load_dotenv
load_dotenv()

try:
    import openai
except ImportError:
    openai = None

import config
import audio_norm
import provider_health
import reporting
import leaderboard_charts as lb_charts
import ui_copy
from ui_copy import Nav, Battle, Leaderboard, Comments, Export, Health, Errors
from scheduler import Scheduler, SchedulerError
from tts_providers import TTSProviderFactory, TTSRequest
from geolocation import geo_service
from security import session_manager
from database import BenchmarkDatabase

db = BenchmarkDatabase()

st.set_page_config(
    page_title=ui_copy.APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css():
    try:
        with open("styles.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def _inject_audix_player_chrome():
    """Hide speed and trim controls in streamlit-advanced-audio players (keep download)."""
    st.html(
        """
        <script>
        (function () {
          const HIDE_CSS = `
            .ant-select.w-20 { display: none !important; }
            .flex.justify-between.items-center > .flex.space-x-3 { display: none !important; }
          `;

          function patchAudixIframes(root) {
            root.querySelectorAll("iframe").forEach((iframe) => {
              try {
                const doc = iframe.contentDocument;
                if (!doc || !doc.querySelector(".ant-select.w-20")) return;
                if (doc.getElementById("arena-audix-hide-toolbar")) return;
                const style = doc.createElement("style");
                style.id = "arena-audix-hide-toolbar";
                style.textContent = HIDE_CSS;
                doc.head.appendChild(style);
              } catch (e) {}
            });
          }

          patchAudixIframes(document);
          new MutationObserver(() => patchAudixIframes(document)).observe(
            document.body,
            { childList: true, subtree: true }
          );
          setInterval(() => patchAudixIframes(document), 500);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


_load_css()
_inject_audix_player_chrome()

# Waveform player styling (matches styles.css primary #6642B3).
_BATTLE_WAVEFORM = WaveSurferOptions(
    wave_color="rgba(102, 66, 179, 0.35)",
    progress_color="#6642B3",
    height=80,
    bar_width=2,
    bar_gap=1,
    cursor_color="#6642B3",
)


def _battle_player(audio_bytes: bytes, key: str, filename: str = "sample.wav") -> None:
    """Waveform audio player for blind battle clips (WAV bytes from normalization).

    The download is a quiet, right-aligned tertiary (link-style) control so it
    stays secondary to the vote CTAs.
    """
    audix(
        audio_bytes,
        format="audio/wav",
        wavesurfer_options=_BATTLE_WAVEFORM,
        autoplay=False,
        key=key,
    )
    _, dl = st.columns([3, 1])
    with dl:
        st.download_button(
            Battle.DOWNLOAD,
            data=audio_bytes,
            file_name=filename,
            mime="audio/wav",
            key=f"download_{key}",
            type="tertiary",
            icon=":material/download:",
            use_container_width=True,
        )


# --- session state -----------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("scheduler", Scheduler())
    ss.setdefault("battle", None)          # current BattlePlan
    ss.setdefault("clips", {})             # {'left': bytes, 'right': bytes}
    ss.setdefault("battle_raw", None)      # falcon + competitor raw clips for failure export
    ss.setdefault("pending_battle", None)  # (language, gender) awaiting synthesis
    ss.setdefault("gen_error", None)
    ss.setdefault("arena_language", None)
    ss.setdefault("arena_gender", "Male")
    ss.setdefault("provider_health", None)
    ss.setdefault("provider_health_at", 0.0)


HEALTH_TTL_SEC = 300


def _ensure_provider_health(force: bool = False) -> dict:
    """Run live synthesis probes (cached) and sync scheduler to healthy providers."""
    ss = st.session_state
    cached = ss.get("provider_health")
    age = time.time() - float(ss.get("provider_health_at") or 0)
    if not force and cached and age < HEALTH_TTL_SEC:
        healthy = provider_health.healthy_ids(cached)
        ss.scheduler.healthy_providers = healthy or None
        return cached

    health = asyncio.run(provider_health.check_all())
    ss.provider_health = health
    ss.provider_health_at = time.time()
    healthy = provider_health.healthy_ids(health)
    ss.scheduler.healthy_providers = healthy or None
    return health


def _health_age_label(checked_at: float) -> str:
    if not checked_at:
        return ""
    secs = int(time.time() - checked_at)
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    return f"{secs // 3600}h ago"


_init_state()


# --- helpers -----------------------------------------------------------------
def _name(provider_id: str) -> str:
    cfg = config.TTS_PROVIDERS.get(provider_id)
    return cfg.name if cfg else provider_id


@st.cache_data(show_spinner=False)
def _cached_leaderboard_bundle(
    cache_key: str,
    outcomes_json: str,
    voices_json: str,
) -> reporting.LeaderboardBundle:
    """Heavy rating + aggregation; invalidated when cache_key changes (votes / hash)."""
    return reporting.build_leaderboard_bundle(
        json.loads(outcomes_json),
        json.loads(voices_json),
    )


def de_anonymize_comment(comment: str, left_id: str, right_id: str) -> str:
    """Map Sample A/B, A/B, and Left/Right tokens to real provider names."""
    if not comment:
        return comment
    left, right = _name(left_id), _name(right_id)
    c = comment
    c = re.sub(r"\bSample\s+A\b", left, c, flags=re.IGNORECASE)
    c = re.sub(r"\bSample\s+B\b", right, c, flags=re.IGNORECASE)
    c = re.sub(r"\bLeft\b", left, c, flags=re.IGNORECASE)
    c = re.sub(r"\bRight\b", right, c, flags=re.IGNORECASE)
    c = re.sub(r"\bA\b", left, c, flags=re.IGNORECASE)
    c = re.sub(r"\bB\b", right, c, flags=re.IGNORECASE)
    return c


def _location() -> dict:
    try:
        loc = geo_service.get_location()
        return {"country": loc.get("country", "Unknown"),
                "city": loc.get("city", "Unknown"),
                "region": loc.get("region", "Unknown")}
    except Exception:
        return {"country": "Unknown", "city": "Unknown", "region": "Unknown"}


def _norm_params() -> audio_norm.NormalizationParams:
    n = config.NORMALIZATION
    return audio_norm.NormalizationParams(
        target_lufs=n["target_lufs"], true_peak_dbfs=n["true_peak_dbfs"],
        sample_rate=n["sample_rate"], channels=n["channels"],
    )


async def _synth_one(provider_id: str, voice: str, text: str):
    provider = TTSProviderFactory.create_provider(provider_id)
    return await provider.generate_speech(
        TTSRequest(text=text, voice=voice, provider=provider_id))


async def _synth_both(plan):
    return await asyncio.gather(
        _synth_one(plan.left_provider, plan.left_voice, plan.item_text),
        _synth_one(plan.right_provider, plan.right_voice, plan.item_text),
        return_exceptions=True,
    )


def _raw_audio_ext(data: bytes | None) -> str:
    return "wav" if data and data[:4] == b"RIFF" else "mp3"


def generate_battle(language: str, gender: str, on_step=None):
    """Schedule + synthesize + normalize a battle; persist it; arm playback.

    `on_step(label)` is an optional callback used to surface progress in the
    battle area (e.g. an st.status). It must never be called from a Streamlit
    callback context.
    """
    def _step(label: str) -> None:
        if on_step is not None:
            on_step(label)

    st.session_state.gen_error = None

    allowed, msg = session_manager.check_rate_limit()
    if not allowed:
        st.session_state.gen_error = msg
        return

    _step(Battle.STEP_SCHEDULE)
    g = gender.lower()
    try:
        plan = st.session_state.scheduler.next_battle(language, gender=g)
    except SchedulerError as e:
        st.session_state.gen_error = str(e)
        return

    _step(Battle.STEP_SYNTHESIZE)
    results = asyncio.run(_synth_both(plan))

    res_left, res_right = results[0], results[1]
    errs = []
    for side, r in (("Left", res_left), ("Right", res_right)):
        if isinstance(r, Exception):
            errs.append(f"{side}: {r}")
        elif not getattr(r, "success", False):
            errs.append(f"{side}: {getattr(r, 'error_message', 'synthesis failed')}")
    if errs:
        st.session_state.gen_error = " | ".join(errs)
        return

    _step(Battle.STEP_NORMALIZE)
    params = _norm_params()
    try:
        nl = audio_norm.normalize_audio(res_left.audio_data, source_format="wav", params=params)
        nr = audio_norm.normalize_audio(res_right.audio_data, source_format="wav", params=params)
    except Exception as e:
        st.session_state.gen_error = Errors.NORMALIZATION.format(error=e)
        return

    loc = _location()
    norm_meta = {**params.to_dict(), "left": nl.metadata(), "right": nr.metadata()}
    db.create_battle(plan, nl.sha256, nr.sha256, norm_meta,
                     session_id=session_manager.get_session_id(), location=loc)

    st.session_state.battle = plan
    st.session_state.clips = {"left": nl.audio, "right": nr.audio}

    # Stash raw model outputs (pre-normalization) for both sides so we can
    # export Falcon + winning competitor clips if Falcon loses this battle.
    anchor = config.anchor_provider()
    falcon_side = (
        "left" if plan.left_provider == anchor
        else "right" if plan.right_provider == anchor
        else None
    )
    if falcon_side:
        comp_side = "right" if falcon_side == "left" else "left"
        falcon_res = res_left if falcon_side == "left" else res_right
        comp_res = res_right if falcon_side == "left" else res_left
        st.session_state.battle_raw = {
            "falcon_side": falcon_side,
            "falcon": {
                "audio": falcon_res.audio_data,
                "ext": _raw_audio_ext(falcon_res.audio_data),
            },
            "competitor": {
                "audio": comp_res.audio_data,
                "ext": _raw_audio_ext(comp_res.audio_data),
            },
        }
    else:
        st.session_state.battle_raw = None

    st.session_state["comment_text"] = ""


def record_vote(outcome: str):
    plan = st.session_state.battle
    if not plan:
        return
    comment = st.session_state.get("comment_text", "") or ""
    deanon = de_anonymize_comment(comment, plan.left_provider, plan.right_provider)
    db.record_vote(plan.battle_id, outcome, comment=comment,
                   comment_deanonymized=deanon,
                   rater_session=session_manager.get_session_id(),
                   location=_location())
    _maybe_store_falcon_failure(plan, outcome, comment_deanonymized=deanon)
    lang = st.session_state.arena_language
    gender = st.session_state.arena_gender
    st.session_state.battle = None
    st.session_state.clips = {}
    st.session_state.battle_raw = None
    st.session_state.pending_battle = (lang, gender)
    st.toast(Battle.VOTE_SAVED)


def _maybe_store_falcon_failure(plan, outcome: str, comment_deanonymized: str = ""):
    """Persist raw Falcon + winning competitor clips when Falcon lost."""
    battle_raw = st.session_state.get("battle_raw")
    if not battle_raw or not battle_raw.get("falcon", {}).get("audio"):
        return
    norm = {"a": "A", "left": "A", "b": "B", "right": "B",
            "tie": "tie", "same": "tie"}.get((outcome or "").strip().lower())
    if norm not in ("A", "B"):
        return

    falcon_side = battle_raw.get("falcon_side")
    falcon_lost = (falcon_side == "left" and norm == "B") or \
                  (falcon_side == "right" and norm == "A")
    if not falcon_lost:
        return

    if falcon_side == "left":
        falcon_voice, competitor, competitor_voice = (
            plan.left_voice, plan.right_provider, plan.right_voice)
    else:
        falcon_voice, competitor, competitor_voice = (
            plan.right_voice, plan.left_provider, plan.left_voice)

    falcon_clip = battle_raw["falcon"]
    comp_clip = battle_raw.get("competitor") or {}

    try:
        db.save_falcon_failure(
            battle_id=plan.battle_id,
            language=plan.language,
            item_id=plan.item_id,
            item_text=plan.item_text,
            falcon_voice=falcon_voice,
            competitor_provider=competitor,
            competitor_voice=competitor_voice,
            outcome=norm,
            falcon_audio_bytes=falcon_clip["audio"],
            falcon_audio_format=falcon_clip.get("ext", "wav"),
            competitor_audio_bytes=comp_clip.get("audio"),
            competitor_audio_format=comp_clip.get("ext", "wav"),
            rater_session=session_manager.get_session_id(),
            comment=comment_deanonymized,
        )
    except Exception:
        # Export capture must never block vote recording.
        pass


def _request_retry():
    """Callback: queue a fresh comparison for the current language/gender."""
    st.session_state.gen_error = None
    st.session_state.pending_battle = (
        st.session_state.arena_language,
        st.session_state.arena_gender,
    )


def _eligible_competitors(language: str, health: dict) -> list[str]:
    """Healthy competitors that support the selected language (excludes anchor)."""
    return provider_health.competitors_for_language(language, health)


def _competitor_multiselect(language: str, health: dict) -> list[str]:
    """Multi-select of eligible competitors; all selected by default per language."""
    eligible = _eligible_competitors(language, health)
    widget_key = f"arena_competitors_{language}"
    prev_key = f"arena_competitors_eligible_{language}"
    prev_eligible = set(st.session_state.get(prev_key, []))
    cur_eligible = set(eligible)

    if widget_key not in st.session_state:
        st.session_state[widget_key] = list(eligible)
    else:
        selected = set(st.session_state[widget_key]) & cur_eligible
        # Auto-select newly eligible providers (e.g. health recovered).
        if prev_eligible:
            selected |= cur_eligible - prev_eligible
        if not selected:
            selected = cur_eligible
        st.session_state[widget_key] = [c for c in eligible if c in selected]

    st.session_state[prev_key] = list(eligible)

    return st.multiselect(
        Battle.COMPETITORS,
        options=eligible,
        format_func=_name,
        key=widget_key,
        placeholder="Select competitors",
    )


def _apply_competitor_selection(selected: list[str]) -> None:
    st.session_state.scheduler.selected_competitors = set(selected)


# --- pages -------------------------------------------------------------------
def battle_page():
    st.title(Battle.TITLE)
    health = _ensure_provider_health()

    languages = st.session_state.scheduler.available_languages()
    if not provider_health.arena_ready(health) or not languages:
        st.warning(Battle.NOT_READY)
        return

    c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
    with c1:
        lang = st.selectbox(
            Battle.LANGUAGE, languages,
            format_func=config.get_language_display,
            index=languages.index(st.session_state.arena_language)
            if st.session_state.arena_language in languages else 0,
        )
    with c2:
        _stored_gender = st.session_state.arena_gender
        _gender_index = (
            Battle.VOICE_GENDER_OPTIONS.index(_stored_gender)
            if _stored_gender in Battle.VOICE_GENDER_OPTIONS else 0
        )
        gender = st.selectbox(
            Battle.VOICE_GENDER, Battle.VOICE_GENDER_OPTIONS,
            index=_gender_index,
        )
    with c3:
        selected_competitors = _competitor_multiselect(lang, health)
    with c4:
        st.write("")
        st.write("")
        start = st.button(
            Battle.START,
            use_container_width=True,
            type="primary",
            disabled=not selected_competitors,
        )

    st.session_state.arena_language = lang
    st.session_state.arena_gender = gender
    _apply_competitor_selection(selected_competitors)

    if not selected_competitors:
        st.warning(Battle.NO_COMPETITORS)

    setup_key = f"{lang}:{gender}:{','.join(sorted(selected_competitors))}"
    plan = st.session_state.battle
    stored_setup = st.session_state.get("battle_setup")
    if plan and stored_setup and stored_setup != setup_key:
        st.session_state.battle = None
        st.session_state.clips = {}
        st.session_state.battle_raw = None
        st.session_state.pending_battle = None
        plan = None
    elif plan and not stored_setup:
        st.session_state.battle_setup = setup_key

    n_comp = len(selected_competitors)
    st.caption(Battle.SETUP.format(lang=config.get_language_display(lang), n=n_comp))

    if start and selected_competitors:
        st.session_state.gen_error = None
        st.session_state.pending_battle = (lang, gender)

    if st.session_state.pending_battle and not st.session_state.gen_error:
        p_lang, p_gender = st.session_state.pending_battle
        st.markdown("---")
        with st.status(Battle.LOADING_TITLE, expanded=True) as status:
            st.caption(Battle.LOADING_HINT)
            generate_battle(
                p_lang, p_gender,
                on_step=lambda label: status.update(label=label),
            )
            if st.session_state.gen_error:
                status.update(label=Battle.LOAD_FAIL.format(
                    error=st.session_state.gen_error), state="error")
            else:
                status.update(label=Battle.LOADING_DONE, state="complete")
        st.session_state.pending_battle = None
        if not st.session_state.gen_error:
            st.session_state.battle_setup = setup_key
            st.rerun()

    if st.session_state.gen_error:
        st.error(Battle.LOAD_FAIL.format(error=st.session_state.gen_error))
        st.button(Battle.RETRY, on_click=_request_retry)
        return

    plan = st.session_state.battle
    if not plan:
        st.info(Battle.PROMPT)
        return

    st.markdown("---")
    st.markdown(f"#### {Battle.QUESTION}")
    st.caption(Battle.GUIDANCE)
    with st.expander(Battle.SHOW_TEXT, expanded=True):
        st.write(plan.item_text)

    player_key = plan.battle_id
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f"##### {Battle.SAMPLE_A}")
        _battle_player(
            st.session_state.clips["left"],
            key=f"battle_player_a_{player_key}",
            filename=f"sample_a_{player_key[:8]}.wav",
        )
        st.checkbox(
            Battle.LISTENED.format(sample=Battle.SAMPLE_A),
            key=f"played_left_{player_key}",
        )
    with right_col:
        st.markdown(f"##### {Battle.SAMPLE_B}")
        _battle_player(
            st.session_state.clips["right"],
            key=f"battle_player_b_{player_key}",
            filename=f"sample_b_{player_key[:8]}.wav",
        )
        st.checkbox(
            Battle.LISTENED.format(sample=Battle.SAMPLE_B),
            key=f"played_right_{player_key}",
        )

    st.text_area(
        Battle.COMMENT,
        key="comment_text",
        height=80,
        placeholder=Battle.COMMENT_PLACEHOLDER,
    )

    can_vote = (
        st.session_state.get(f"played_left_{player_key}")
        and st.session_state.get(f"played_right_{player_key}")
    )
    if not can_vote:
        st.info(Battle.VOTE_GATE)

    v1, v2, v3 = st.columns(3)
    with v1:
        st.button(Battle.VOTE_A, use_container_width=True, disabled=not can_vote,
                  on_click=lambda: record_vote("A"))
    with v2:
        st.button(Battle.VOTE_TIE, use_container_width=True, disabled=not can_vote,
                  on_click=lambda: record_vote("tie"))
    with v3:
        st.button(Battle.VOTE_B, use_container_width=True, disabled=not can_vote,
                  on_click=lambda: record_vote("B"))


def leaderboard_page():
    st.title(Leaderboard.TITLE)
    anchor = config.anchor_provider()
    counts = db.get_vote_counts()
    m1, m2, m3 = st.columns(3)
    m1.metric(Leaderboard.METRIC_VOTES, counts["votes"])
    m2.metric(Leaderboard.METRIC_COMPARISONS, counts["battles"])
    m3.metric(Leaderboard.METRIC_REFERENCE, _name(anchor))

    inputs = db.fetch_leaderboard_inputs()
    cache_key = reporting.leaderboard_cache_key(
        counts["votes"], counts["battles"], inputs["outcomes"],
    )
    with st.spinner(Leaderboard.LOADING):
        bundle = _cached_leaderboard_bundle(
            cache_key,
            json.dumps(inputs["outcomes"], sort_keys=True),
            json.dumps(inputs["voices"], sort_keys=True),
        )

    if not bundle.languages:
        st.info(Leaderboard.NO_DATA)
        return

    st.caption(Leaderboard.COLOR_LEGEND)

    st.markdown(f"### {Leaderboard.HEAD_TO_HEAD}")
    if bundle.head_to_head:
        st.plotly_chart(
            lb_charts.outcome_stacked_bar(
                lb_charts.prep_head_to_head(list(bundle.head_to_head)),
            ),
            use_container_width=True,
        )
    else:
        st.info(Leaderboard.NO_DATA)

    st.markdown(f"### {Leaderboard.OVERALL}")
    st.caption(Leaderboard.WEIGHTS.format(weights=", ".join(
        f"{config.get_language_display(l)} {bundle.language_weights[l] * 100:.0f}%"
        for l in bundle.languages
    )))
    if bundle.overall_rows:
        st.plotly_chart(
            lb_charts.forest_plot(list(bundle.overall_rows)),
            use_container_width=True,
        )

    st.markdown(f"### {Leaderboard.PER_LANGUAGE}")
    if bundle.heatmap_cells:
        st.plotly_chart(
            lb_charts.language_winrate_heatmap(list(bundle.heatmap_cells)),
            use_container_width=True,
        )

    lang_pick = st.selectbox(
        Leaderboard.LANG_DETAIL_PICK,
        options=[""] + list(bundle.languages),
        format_func=lambda code: (
            Leaderboard.LANG_DETAIL_NONE if not code
            else config.get_language_display(code)
        ),
    )
    if lang_pick:
        meta = bundle.fits_meta[lang_pick]
        st.caption(Leaderboard.LANG_SUMMARY.format(
            n=meta["n_comparisons"], ties=meta["n_ties"],
        ))
        battle_counts = db.get_competitor_battle_counts(lang_pick)
        if battle_counts:
            breakdown = ", ".join(
                f"{_name(pid)} {n}" for pid, n in sorted(
                    battle_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            st.caption(Leaderboard.BATTLES_BY_COMPETITOR.format(breakdown=breakdown))
        lang_grid = bundle.winrate_by_lang.get(lang_pick, ())
        if lang_grid:
            st.plotly_chart(
                lb_charts.outcome_stacked_bar(
                    lb_charts.prep_head_to_head(list(lang_grid)),
                ),
                use_container_width=True,
            )

    st.markdown(f"### {Leaderboard.VOICE_BREAKDOWN}")
    voice_lang = st.selectbox(
        Leaderboard.VOICE_LANG_PICK,
        options=list(bundle.languages),
        format_func=config.get_language_display,
    )
    vrows = bundle.voice_by_lang.get(voice_lang, ())
    if not vrows:
        st.info(Leaderboard.VOICE_NO_DATA)
    else:
        st.plotly_chart(
            lb_charts.outcome_stacked_bar(
                lb_charts.prep_voice_rows(list(vrows)),
                scale_low_n=True,
            ),
            use_container_width=True,
        )


def comments_page():
    st.title(Comments.TITLE)
    languages = db.get_languages_with_votes()
    if not languages:
        st.info(Comments.NO_COMMENTS)
        return
    lang = st.selectbox(Comments.LANGUAGE, languages, format_func=config.get_language_display)
    rows = db.get_arena_comments(lang)
    comments = [r["comment_deanonymized"] for r in rows if r.get("comment_deanonymized")]
    st.caption(Comments.COUNT.format(n=len(comments), lang=config.get_language_display(lang)))
    for r in rows:
        if r.get("comment_deanonymized"):
            st.markdown(f"- {r['comment_deanonymized']}")

    st.markdown("---")
    st.markdown(f"### {Comments.SUMMARY_HEADER}")
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        st.warning(Comments.NEED_KEY)
        return
    if not comments:
        st.info(Comments.NOTHING_TO_SUMMARIZE)
        return

    stored = db.get_locale_summary(lang)
    if stored and stored["comment_count"] == len(comments) and not st.session_state.get(f"force_{lang}"):
        st.success(Comments.SAVED_SUMMARY.format(n=stored["comment_count"]))
        st.markdown(stored["summary"])
    else:
        if st.button(Comments.GENERATE):
            with st.spinner(Comments.SUMMARIZING):
                try:
                    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    joined = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))
                    prompt = (
                        f"Summarize rater feedback for TTS voices in locale {lang}. "
                        f"Focus only on critical, recurring quality issues (naturalness, "
                        f"pronunciation, pace) and clear preferences. Max 200 words, bullet points.\n\n{joined}"
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": "You filter for critical, impactful issues only. Be concise."},
                                  {"role": "user", "content": prompt}],
                        temperature=0.2, max_tokens=500)
                    summary = resp.choices[0].message.content
                    db.save_locale_summary(lang, summary, len(comments), "gpt-4o")
                    st.session_state.pop(f"force_{lang}", None)
                    st.markdown(summary)
                except Exception as e:
                    st.error(Comments.SUMMARY_FAIL.format(error=e))


def _provider_display(pid: str) -> str:
    cfg = config.TTS_PROVIDERS.get(pid)
    return cfg.name if cfg else pid


def _fname_safe(value: str) -> str:
    """Filesystem-safe token (no path separators / spaces / colons)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-") or "unknown"


def _build_falcon_failures_zip(rows: list) -> bytes:
    """ZIP of paired Falcon + competitor raw clips per lost battle."""
    manifest_cols = [
        "battle_id", "language", "item_id", "text",
        "falcon_voice", "competitor_provider", "competitor_voice",
        "outcome", "comment", "created_at",
        "falcon_audio_file", "falcon_audio_format",
        "competitor_audio_file", "competitor_audio_format",
    ]
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=manifest_cols)
    writer.writeheader()
    jsonl_lines: list[str] = []

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: dict[str, int] = {}
        for rec in rows:
            lang = rec.get("language") or "unknown"
            ts = _fname_safe(rec.get("created_at"))
            bid = _fname_safe(rec.get("battle_id"))
            falcon_voice = _fname_safe(rec.get("falcon_voice"))
            comp_pid = _fname_safe(rec.get("competitor_provider"))
            comp_voice = _fname_safe(rec.get("competitor_voice"))
            stem = f"{ts}__{bid}"
            seen[stem] = seen.get(stem, 0) + 1
            if seen[stem] > 1:
                stem = f"{stem}__{seen[stem]}"
            pair_dir = f"audio/{_fname_safe(lang)}/{stem}"

            falcon_ext = rec.get("falcon_audio_format") or rec.get("audio_format") or "wav"
            comp_ext = rec.get("competitor_audio_format") or "wav"
            falcon_file = f"{pair_dir}/falcon__{falcon_voice}.{falcon_ext}"
            comp_file = f"{pair_dir}/{comp_pid}__{comp_voice}.{comp_ext}"

            falcon_audio = rec.get("falcon_audio_bytes") or b""
            comp_audio = rec.get("competitor_audio_bytes") or b""
            if falcon_audio:
                zf.writestr(f"falcon_failures/{falcon_file}", falcon_audio)
            if comp_audio:
                zf.writestr(f"falcon_failures/{comp_file}", comp_audio)

            row = {
                "battle_id": rec.get("battle_id", ""),
                "language": lang,
                "item_id": rec.get("item_id", ""),
                "text": rec.get("item_text", ""),
                "falcon_voice": rec.get("falcon_voice", ""),
                "competitor_provider": _provider_display(rec.get("competitor_provider", "")),
                "competitor_voice": rec.get("competitor_voice", ""),
                "outcome": rec.get("outcome", ""),
                "comment": rec.get("comment") or "",
                "created_at": str(rec.get("created_at", "")),
                "falcon_audio_file": falcon_file if falcon_audio else "",
                "falcon_audio_format": falcon_ext if falcon_audio else "",
                "competitor_audio_file": comp_file if comp_audio else "",
                "competitor_audio_format": comp_ext if comp_audio else "",
            }
            writer.writerow(row)
            jsonl_lines.append(json.dumps(row, ensure_ascii=False, default=str))

        zf.writestr("falcon_failures/failures.csv", csv_buf.getvalue())
        zf.writestr("falcon_failures/failures.jsonl", "\n".join(jsonl_lines))
        zf.writestr(
            "falcon_failures/README.txt",
            "Falcon failure export\n"
            "=====================\n\n"
            "Each row is a battle where the rater preferred the competitor over\n"
            "Falcon 2. Ties and Falcon wins are not included.\n\n"
            "Folder layout (one pair per battle):\n"
            "  audio/<language>/<timestamp>__<battle_id>/\n"
            "    falcon__<falcon_voice>.<wav|mp3>      — raw Falcon model output\n"
            "    <competitor>__<competitor_voice>.<wav|mp3> — raw winning clip\n\n"
            "failures.csv and failures.jsonl list both audio paths, the spoken\n"
            "text, voices, competitor name, deanonymized comment (if any), and\n"
            "timestamp. Older rows may only have the Falcon clip.\n",
        )
    return buf.getvalue()


def export_page():
    st.title(Export.TITLE)
    report = reporting.ArenaReport(db)
    if not db.get_languages_with_votes():
        st.info(Export.NO_DATA)
        _falcon_failures_section()
        return

    with st.spinner(Export.BUILDING):
        export = report.build_export()

    payload = json.dumps(export, indent=2, default=str)
    odf = pd.DataFrame(export["overall"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            Export.DOWNLOAD_JSON,
            payload,
            file_name=f"voice_arena_report_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            Export.DOWNLOAD_CSV,
            odf.to_csv(index=False),
            file_name="voice_arena_overall.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        if st.button(
            Export.SAVE_SNAPSHOT,
            use_container_width=True,
            help=Export.SAVE_SNAPSHOT_HELP,
        ):
            run_id = report.persist_runs()
            st.success(Export.SNAPSHOT_SAVED.format(run_id=run_id))

    with st.expander(Export.TECH_DETAILS, expanded=False):
        st.json(export["snapshot"])

    runs = db.get_latest_ratings_runs(10)
    if runs:
        st.markdown(f"### {Export.RECENT}")
        st.dataframe(pd.DataFrame([{
            "run_id": r["run_id"][:8], "scope": r["scope"], "language": r["language"],
            "n_votes": r["n_votes"], "code_version": r["code_version"],
            "created_at": r["created_at"],
        } for r in runs]), use_container_width=True, hide_index=True)

    _falcon_failures_section()


def _falcon_failures_section():
    """Download paired Falcon + competitor clips for battles where Falcon lost."""
    st.markdown("---")
    st.markdown(f"### {Export.FAILURES_TITLE}")
    st.caption(Export.FAILURES_HELP)

    total = db.count_falcon_failures()
    if not total:
        st.info(Export.FAILURES_EMPTY)
        return

    langs = sorted({r["language"] for r in db.get_falcon_failures() if r.get("language")})
    options = [Export.FAILURES_ALL_LANGS] + langs
    choice = st.selectbox(
        Export.FAILURES_LANGUAGE,
        options,
        format_func=lambda o: o if o == Export.FAILURES_ALL_LANGS
        else config.get_language_display(o),
        key="falcon_failures_lang",
    )
    lang_filter = None if choice == Export.FAILURES_ALL_LANGS else choice
    count = db.count_falcon_failures(lang_filter)
    st.caption(Export.FAILURES_COUNT.format(n=count))

    if not count:
        return

    suffix = "all" if lang_filter is None else _fname_safe(lang_filter)
    if st.button(Export.FAILURES_PREPARE, use_container_width=True):
        with st.spinner(Export.BUILDING):
            rows = db.get_falcon_failures(lang_filter, with_audio=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state["falcon_failures_zip"] = {
                "bytes": _build_falcon_failures_zip(rows),
                "name": f"falcon_failures_{suffix}_{ts}.zip",
                "filter": choice,
            }

    prepared = st.session_state.get("falcon_failures_zip")
    if prepared and prepared.get("filter") == choice:
        st.download_button(
            Export.FAILURES_DOWNLOAD,
            prepared["bytes"],
            file_name=prepared["name"],
            mime="application/zip",
            use_container_width=True,
        )


def _status_badge(state: str) -> tuple[str, str]:
    """Return (label, color) for st.badge."""
    return {
        "ok": (Health.OK, "green"),
        "fail": (Health.DOWN, "red"),
        "unconfigured": (Health.UNCONFIGURED, "gray"),
    }.get(state, (Health.UNCONFIGURED, "gray"))


def _render_provider_health_row(row: dict, pid: str, *, anchor: bool = False) -> None:
    state = row.get("state", "unconfigured")
    badge_label, badge_color = _status_badge(state)
    name = _name(pid)
    title = Health.REFERENCE_ROW.format(name=name) if anchor else name
    detail = row.get("message", "")

    c1, c2 = st.columns([1.4, 4], gap="small")
    with c1:
        st.badge(badge_label, color=badge_color)
    with c2:
        if state == "ok":
            st.markdown(f"**{title}** — {detail}")
        elif state == "fail":
            st.markdown(f"**{title}**")
            if detail:
                st.caption(detail)
        else:
            st.markdown(title)


def _sidebar_health() -> None:
    """Provider health panel — lives below native st.navigation in the sidebar."""
    cached = st.session_state.get("provider_health")
    age = time.time() - float(st.session_state.get("provider_health_at") or 0)
    need_probe = not cached or age >= HEALTH_TTL_SEC
    if need_probe:
        with st.sidebar.spinner(Health.CHECKING):
            health = _ensure_provider_health(force=True)
    else:
        health = _ensure_provider_health()

    n_total = len(config.TTS_PROVIDERS)
    n_ok = sum(1 for row in health.values() if row.get("state") == "ok")
    checked_at = max((row.get("checked_at") or 0) for row in health.values()) if health else 0

    with st.sidebar.expander(Health.SUMMARY.format(n_ok=n_ok, n_total=n_total), expanded=False):
        if st.button(Health.CHECK, use_container_width=True, key="sidebar_retest_providers"):
            st.session_state.provider_health = None
            st.session_state.provider_health_at = 0.0
            st.rerun()

        st.caption(Health.LAST_CHECKED.format(age=_health_age_label(checked_at)))

        anchor = config.anchor_provider()
        _render_provider_health_row(health.get(anchor, {}), anchor, anchor=True)

        for pid in config.TTS_PROVIDERS:
            if pid == anchor:
                continue
            _render_provider_health_row(health.get(pid, {}), pid)


def main():
    pg = st.navigation(
        [
            st.Page(
                battle_page,
                title=Nav.BATTLE,
                icon=":material/headphones:",
                default=True,
            ),
            st.Page(
                leaderboard_page,
                title=Nav.LEADERBOARD,
                icon=":material/leaderboard:",
                url_path="leaderboard",
            ),
            st.Page(
                comments_page,
                title=Nav.COMMENTS,
                icon=":material/forum:",
                url_path="comments",
            ),
            st.Page(
                export_page,
                title=Nav.EXPORT,
                icon=":material/download:",
                url_path="export",
            ),
        ],
        position="sidebar",
    )
    _sidebar_health()
    pg.run()


if __name__ == "__main__":
    main()
