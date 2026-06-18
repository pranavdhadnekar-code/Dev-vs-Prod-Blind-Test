"""Centralized user-facing copy for Voice Arena.

Every string a user can read lives here so the tone stays consistent and plain.
Rules:
  * One idea per line; short sentences.
  * No engine/stats jargon in the main UI (Elo, LUFS, Bradley-Terry, synthesis
    probe). Technical detail is relocated to the Export "Technical details" panel.
  * Buttons are verbs ("Start comparison", "Try again", "Check again").
  * Consistent terms: "comparison" (not battle/matchup), "provider" (not
    competitor), "score" (not Elo), "reference voice" / ANCHOR (not anchor).
  * Dynamic bits use named {placeholders} via str.format — never ad-hoc concat.

ANCHOR is the single source of truth for the reference model's display name; a
future rename touches one line here.
"""

APP_TITLE = "Murf Voice Arena"
ANCHOR = "Falcon 2"


class Nav:
    BATTLE = "Compare voices"
    LEADERBOARD = "Leaderboard"
    COMMENTS = "Comments"
    EXPORT = "Export"


class Battle:
    TITLE = "Compare voices"
    NOT_READY = (
        f"{ANCHOR} and at least one other provider need to be working before you "
        "can compare voices. Open **Provider health** in the sidebar, then tap "
        "**Check again** if you just updated your API keys."
    )
    LANGUAGE = "Language"
    VOICE_GENDER = "Voice gender"
    VOICE_GENDER_OPTIONS = ["Male", "Female"]
    COMPETITORS = "Competitors"
    NO_COMPETITORS = "Select at least one competitor to run a comparison."
    START = "Start comparison"
    SETUP = (
        "{lang} · {n} providers compared to " + ANCHOR + " · both clips are "
        "loudness-matched so you judge voice quality only."
    )
    SYNTHESIZING = "Creating both clips…"
    LOADING_TITLE = "Preparing your comparison…"
    LOADING_DONE = "Comparison ready"
    STEP_SCHEDULE = "Picking a matchup…"
    STEP_SYNTHESIZE = "Generating Sample A and Sample B…"
    STEP_NORMALIZE = "Matching loudness so you judge voice only…"
    LOADING_HINT = "Hang tight — this usually takes a few seconds."
    LOAD_FAIL = "Couldn’t load this comparison: {error}"
    RETRY = "Try again"
    PROMPT = "Tap **Start comparison** to hear two clips and pick the better voice."
    QUESTION = "Which voice is better overall?"
    GUIDANCE = (
        "Listen for how natural, clear, and well-paced each voice sounds. "
        "Both clips read the same text."
    )
    SHOW_TEXT = "Spoken text"
    SAMPLE_A = "Sample A"
    SAMPLE_B = "Sample B"
    LISTENED = "I listened to {sample}"
    DOWNLOAD = "Download"
    COMMENT = "Add a note (optional)"
    COMMENT_PLACEHOLDER = (
        "e.g. Sample B sounds more natural and expressive — clearer tone and "
        "better pacing."
    )
    VOTE_GATE = "Listen to both clips and check both boxes to vote."
    VOTE_A = "Sample A is better"
    VOTE_TIE = "About the same"
    VOTE_B = "Sample B is better"
    VOTE_SAVED = "Vote saved — loading the next comparison…"


class Leaderboard:
    TITLE = "Leaderboard"
    METRIC_VOTES = "Votes"
    METRIC_COMPARISONS = "Comparisons"
    METRIC_REFERENCE = "Reference voice"
    NO_DATA = "No votes yet. Run some comparisons first."
    HEAD_TO_HEAD = f"How often {ANCHOR} wins against each provider"
    CHART_TITLE = f"{ANCHOR} win rate (%)"
    OVERALL = "Overall ranking"
    WEIGHTS = "Language weights: {weights}"
    PER_LANGUAGE = "Results by language"
    VOICE_BREAKDOWN = f"{ANCHOR} Win Rate by Voice"
    VOICE_BREAKDOWN_HINT = (
        f"Each {ANCHOR} voice across all comparisons it appeared in. "
        "Win rate excludes ties."
    )
    VOICE_NO_DATA = f"No {ANCHOR} voice comparisons yet."
    COL_VOICE = "Voice"
    COL_GENDER = "Gender"
    COL_ANCHOR_LOSSES = f"{ANCHOR} loses"
    COL_LOSES_TO = "Loses to"
    LANG_SUMMARY = "{n} comparisons · {ties} ties"
    BATTLES_BY_COMPETITOR = "Comparisons run: {breakdown}"
    SCHEDULER_NOTE = (
        f"Each comparison is {ANCHOR} vs one competitor. Only providers that pass "
        "the health check (see sidebar) are scheduled — failed or unconfigured "
        "providers are skipped."
    )
    COL_BATTLES_RUN = "Battles run"
    LANG_HEAD_TO_HEAD = f"{ANCHOR} vs each provider"
    INFERRED_NOTE = (
        f"Rankings against {ANCHOR} come straight from your votes. Rankings "
        "between other providers are estimated and less certain."
    )
    # Column headers
    COL_RANK = "Rank"
    COL_PROVIDER = "Provider"
    COL_SCORE = "Score"
    COL_CONFIDENCE = "Confidence range"
    COL_COMPARISONS = "Comparisons"
    COL_TIED_WITH = "Tied with"
    COL_COMPETITOR = "Provider"
    COL_ANCHOR_WINS = f"{ANCHOR} wins"
    COL_PROVIDER_WINS = "Provider wins"
    COL_TIES = "Ties"
    COL_WIN_RATE = f"{ANCHOR} win rate"


class Comments:
    TITLE = "Comments and summaries"
    NO_COMMENTS = "No comments yet."
    LANGUAGE = "Language"
    COUNT = "{n} comments for {lang} (with provider names)"
    SUMMARY_HEADER = "Summary"
    NEED_KEY = "Add an OpenAI API key to generate a summary."
    NOTHING_TO_SUMMARIZE = "No comments to summarize yet."
    SAVED_SUMMARY = "Saved summary ({n} comments)"
    GENERATE = "Create or update summary"
    SUMMARIZING = "Writing summary…"
    SUMMARY_FAIL = "Couldn’t create summary: {error}"


class Export:
    TITLE = "Export results"
    NO_DATA = "Nothing to export yet. Run some comparisons first."
    BUILDING = "Preparing your report…"
    DOWNLOAD_JSON = "Download full report"
    DOWNLOAD_CSV = "Download rankings (CSV)"
    SAVE_SNAPSHOT = "Save audit snapshot"
    SAVE_SNAPSHOT_HELP = "Save a versioned snapshot for audit and reproducibility."
    SNAPSHOT_SAVED = "Snapshot saved ({run_id})"
    TECH_DETAILS = "Technical details"
    RECENT = "Recent snapshots"
    FAILURES_TITLE = "Falcon failures (lost battles)"
    FAILURES_HELP = (
        "Download the raw Falcon 2 audio and text for battles where the "
        "competitor was preferred. Only failures recorded after this feature "
        "was deployed are included."
    )
    FAILURES_EMPTY = "No Falcon losses recorded yet."
    FAILURES_ALL_LANGS = "All languages"
    FAILURES_LANGUAGE = "Language"
    FAILURES_COUNT = "{n} failing clip(s) ready to export."
    FAILURES_PREPARE = "Prepare failures (.zip)"
    FAILURES_DOWNLOAD = "Download failures (.zip)"


class Health:
    SUMMARY = "{n_ok}/{n_total} providers OK"
    CHECK = "Check again"
    CHECKING = "Checking voice providers…"
    LAST_CHECKED = "Last checked {age}"
    REFERENCE_ROW = f"{{name}} (reference voice)"
    OK = "OK"
    DOWN = "Down"
    UNCONFIGURED = "Not configured"


class Errors:
    NORMALIZATION = "Audio processing failed: {error}"
    NO_PROVIDERS = "No providers available for this language."
    NO_SENTENCES = "No test sentences available for this language."
