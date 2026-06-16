"""Murf Voice Arena — multi-provider blind listening test (Streamlit).

Every battle pits the anchor (Falcon 2) against one competitor for a
chosen language. Both clips are loudness/format-normalized so nothing but voice
quality distinguishes them. Votes (Left / Right / About the same) feed a
rating engine with bootstrap confidence intervals.
"""
import asyncio
import json
import os
import re
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from streamlit_advanced_audio import audix, WaveSurferOptions

from dotenv import load_dotenv
load_dotenv()

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import openai
except ImportError:
    openai = None

import config
import audio_norm
import provider_health
import reporting
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
    ss.setdefault("pending_battle", None)  # (language, gender) awaiting synthesis
    ss.setdefault("gen_error", None)
    ss.setdefault("arena_language", None)
    ss.setdefault("arena_gender", "Any")
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
    g = None if gender == "Any" else gender.lower()
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
    st.session_state.battle_setup = f"{language}:{gender}"
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
    lang = st.session_state.arena_language
    gender = st.session_state.arena_gender
    st.session_state.battle = None
    st.session_state.clips = {}
    st.session_state.pending_battle = (lang, gender)
    st.toast(Battle.VOTE_SAVED)


def _request_retry():
    """Callback: queue a fresh comparison for the current language/gender."""
    st.session_state.gen_error = None
    st.session_state.pending_battle = (
        st.session_state.arena_language,
        st.session_state.arena_gender,
    )


# --- pages -------------------------------------------------------------------
def battle_page():
    st.title(Battle.TITLE)
    health = _ensure_provider_health()

    languages = st.session_state.scheduler.available_languages()
    if not provider_health.arena_ready(health) or not languages:
        st.warning(Battle.NOT_READY)
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        lang = st.selectbox(
            Battle.LANGUAGE, languages,
            format_func=config.get_language_display,
            index=languages.index(st.session_state.arena_language)
            if st.session_state.arena_language in languages else 0,
        )
    with c2:
        gender = st.selectbox(
            Battle.VOICE_GENDER, Battle.VOICE_GENDER_OPTIONS,
            index=Battle.VOICE_GENDER_OPTIONS.index(st.session_state.arena_gender),
        )
    with c3:
        st.write("")
        st.write("")
        start = st.button(Battle.START, use_container_width=True, type="primary")

    st.session_state.arena_language = lang
    st.session_state.arena_gender = gender

    setup_key = f"{lang}:{gender}"
    plan = st.session_state.battle
    stored_setup = st.session_state.get("battle_setup")
    if plan and stored_setup and stored_setup != setup_key:
        st.session_state.battle = None
        st.session_state.clips = {}
        st.session_state.pending_battle = None
        plan = None
    elif plan and not stored_setup:
        st.session_state.battle_setup = setup_key

    n_comp = len(provider_health.competitors_for_language(lang, health))
    st.caption(Battle.SETUP.format(lang=config.get_language_display(lang), n=n_comp))

    if start:
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

    st.text_area(Battle.COMMENT, key="comment_text", height=80)

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


def _winrate_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df[Leaderboard.COL_WIN_RATE] = (df["anchor_win_rate"] * 100).round(1)
    df[Leaderboard.COL_CONFIDENCE] = df.apply(
        lambda r: f"{r['ci_low']*100:.0f}–{r['ci_high']*100:.0f}%"
        if pd.notna(r["ci_low"]) else "—", axis=1)
    return df[["competitor_name", "anchor_preferred", "competitor_preferred",
               "ties", "n", Leaderboard.COL_WIN_RATE, Leaderboard.COL_CONFIDENCE]].rename(columns={
        "competitor_name": Leaderboard.COL_COMPETITOR,
        "anchor_preferred": Leaderboard.COL_ANCHOR_WINS,
        "competitor_preferred": Leaderboard.COL_PROVIDER_WINS,
        "ties": Leaderboard.COL_TIES, "n": Leaderboard.COL_COMPARISONS})


def _voice_winrate_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df[Leaderboard.COL_WIN_RATE] = df["anchor_win_rate"].apply(
        lambda x: round(x * 100, 1) if pd.notna(x) else None)
    df[Leaderboard.COL_CONFIDENCE] = df.apply(
        lambda r: f"{r['ci_low']*100:.0f}–{r['ci_high']*100:.0f}%"
        if pd.notna(r["ci_low"]) else "—", axis=1)
    df[Leaderboard.COL_VOICE] = df.apply(
        lambda r: f"{r['voice_name']} ({r['voice']})"
        if r["voice_name"] != r["voice"] else r["voice"], axis=1)
    df[Leaderboard.COL_GENDER] = df["gender"].str.capitalize()
    df[Leaderboard.COL_LOSES_TO] = df["loses_to_label"]
    return df[[Leaderboard.COL_VOICE, Leaderboard.COL_GENDER, "anchor_preferred",
               "competitor_preferred", "ties", "n",
               Leaderboard.COL_WIN_RATE, Leaderboard.COL_CONFIDENCE,
               Leaderboard.COL_LOSES_TO]].rename(columns={
        "anchor_preferred": Leaderboard.COL_ANCHOR_WINS,
        "competitor_preferred": Leaderboard.COL_ANCHOR_LOSSES,
        "ties": Leaderboard.COL_TIES, "n": Leaderboard.COL_COMPARISONS})


def leaderboard_page():
    st.title(Leaderboard.TITLE)
    report = reporting.ArenaReport(db)
    counts = db.get_vote_counts()
    m1, m2, m3 = st.columns(3)
    m1.metric(Leaderboard.METRIC_VOTES, counts["votes"])
    m2.metric(Leaderboard.METRIC_COMPARISONS, counts["battles"])
    m3.metric(Leaderboard.METRIC_REFERENCE, _name(report.anchor))

    fits = report.fit_all()
    if not fits:
        st.info(Leaderboard.NO_DATA)
        return

    st.markdown(f"### {Leaderboard.HEAD_TO_HEAD}")
    grid = report.winrate_grid(None)
    df = _winrate_df(grid)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        if px is not None:
            fig = px.bar(df, x=Leaderboard.COL_COMPETITOR, y=Leaderboard.COL_WIN_RATE,
                         title=Leaderboard.CHART_TITLE, range_y=[0, 100])
            fig.add_hline(y=50, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### {Leaderboard.OVERALL}")
    weights = config.language_weights(list(fits.keys()))
    st.caption(Leaderboard.WEIGHTS.format(weights=", ".join(
        f"{config.get_language_display(l)} {w*100:.0f}%" for l, w in weights.items())))
    overall = report.overall(fits)
    orows = report.overall_leaderboard(overall)
    odf = pd.DataFrame([{
        Leaderboard.COL_RANK: r["rank"],
        Leaderboard.COL_PROVIDER: r["provider_name"] + (" ⚓" if r["is_anchor"] else ""),
        Leaderboard.COL_SCORE: round(r["elo"], 1),
        Leaderboard.COL_CONFIDENCE: f"{r['elo_ci_low']:.0f}–{r['elo_ci_high']:.0f}",
        Leaderboard.COL_COMPARISONS: r["n_comparisons"],
        Leaderboard.COL_TIED_WITH: ", ".join(_name(p) for p in r["statistically_tied_with"]) or "—",
    } for r in orows])
    st.dataframe(odf, use_container_width=True, hide_index=True)

    st.markdown(f"### {Leaderboard.PER_LANGUAGE}")
    tabs = st.tabs([config.get_language_display(l) for l in fits])
    for tab, (lang, fit) in zip(tabs, fits.items()):
        with tab:
            st.caption(Leaderboard.LANG_SUMMARY.format(n=fit.n_comparisons, ties=fit.n_ties))
            battle_counts = db.get_competitor_battle_counts(lang)
            if battle_counts:
                breakdown = ", ".join(
                    f"{_name(pid)} {n}" for pid, n in sorted(
                        battle_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                )
                st.caption(Leaderboard.BATTLES_BY_COMPETITOR.format(breakdown=breakdown))
            st.caption(Leaderboard.SCHEDULER_NOTE)
            lb = report.language_leaderboard(fit)
            ldf = pd.DataFrame([{
                Leaderboard.COL_PROVIDER: r["provider_name"] + (" ⚓" if r["is_anchor"] else ""),
                Leaderboard.COL_SCORE: round(r["elo"], 1),
                Leaderboard.COL_CONFIDENCE: f"{r['elo_ci_low']:.0f}–{r['elo_ci_high']:.0f}",
                Leaderboard.COL_COMPARISONS: r["n_comparisons"],
            } for r in lb])
            st.dataframe(ldf, use_container_width=True, hide_index=True)
            st.markdown(f"**{Leaderboard.LANG_HEAD_TO_HEAD}**")
            gdf = _winrate_df(report.winrate_grid(lang))
            if not gdf.empty:
                st.dataframe(gdf, use_container_width=True, hide_index=True)
            st.info(Leaderboard.INFERRED_NOTE)

    st.markdown(f"### {Leaderboard.VOICE_BREAKDOWN}")
    st.caption(Leaderboard.VOICE_BREAKDOWN_HINT)
    vtabs = st.tabs([config.get_language_display(l) for l in fits])
    for tab, lang in zip(vtabs, fits):
        with tab:
            vdf = _voice_winrate_df(report.anchor_voice_winrate(lang))
            if vdf.empty:
                st.info(Leaderboard.VOICE_NO_DATA)
            else:
                st.dataframe(vdf, use_container_width=True, hide_index=True)


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


def export_page():
    st.title(Export.TITLE)
    report = reporting.ArenaReport(db)
    if not db.get_languages_with_votes():
        st.info(Export.NO_DATA)
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
