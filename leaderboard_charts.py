"""Plotly charts for the Voice Arena leaderboard.

Consistent color language:
  * teal  — Falcon wins / strength
  * coral — competitor wins
  * gray  — ties / uncertainty (CI bands)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

COLOR_FALCON = "#0d9488"
COLOR_COMPETITOR = "#f97066"
COLOR_TIE = "#94a3b8"
COLOR_CI = "#cbd5e1"
COLOR_ANCHOR = "#115e59"
COLOR_SCORE_LABEL = "#e2e8f0"

LOW_N_THRESHOLD = 5
VOICE_REF_N = 20


def _pct(part: int, whole: int) -> float:
    return (100.0 * part / whole) if whole else 0.0


def _hover_outcome(name: str, wins: int, losses: int, ties: int, n: int) -> str:
    return (
        f"<b>{name}</b><br>"
        f"n={n}<br>"
        f"Falcon wins: {wins} ({_pct(wins, n):.1f}%)<br>"
        f"Ties: {ties} ({_pct(ties, n):.1f}%)<br>"
        f"Competitor wins: {losses} ({_pct(losses, n):.1f}%)"
        "<extra></extra>"
    )


def _bar_label(pct: float) -> str:
    """Segment label; omit empty slices."""
    return f"{pct:.0f}%" if pct >= 0.5 else ""


def _chart_layout(
    fig: go.Figure,
    height: int,
    *,
    margin_l: int = 140,
    title: Optional[str] = None,
) -> go.Figure:
    layout: Dict[str, Any] = dict(
        height=height,
        margin=dict(l=margin_l, r=24, t=28 if not title else 48, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(size=13),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        yaxis=dict(showgrid=False, automargin=True),
    )
    if title:
        layout["title"] = dict(text=title, x=0, xanchor="left", font=dict(size=14))
    fig.update_layout(**layout)
    return fig


def outcome_stacked_bar(
    rows: List[Dict[str, Any]],
    *,
    name_key: str = "name",
    wins_key: str = "wins",
    losses_key: str = "losses",
    ties_key: str = "ties",
    n_key: str = "n",
    scale_low_n: bool = False,
    low_n_threshold: int = LOW_N_THRESHOLD,
    title: Optional[str] = None,
) -> go.Figure:
    """Horizontal normalized win / tie / loss bars with n= labels on the left."""
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", x=0.5, y=0.5, showarrow=False)
        return fig

    ordered = sorted(rows, key=lambda r: (-int(r.get(n_key) or 0), str(r.get(name_key, ""))))
    labels: List[str] = []
    win_x, tie_x, loss_x = [], [], []
    win_text, tie_text, loss_text = [], [], []
    hovers = []

    for r in ordered:
        n = int(r.get(n_key) or 0)
        wins = int(r.get(wins_key) or 0)
        losses = int(r.get(losses_key) or 0)
        ties = int(r.get(ties_key) or 0)
        name = str(r.get(name_key, ""))
        low_tag = f" · low n" if scale_low_n and n <= low_n_threshold else ""
        labels.append(f"n={n}  {name}{low_tag}")

        win_pct = _pct(wins, n)
        tie_pct = _pct(ties, n)
        loss_pct = _pct(losses, n)

        scale = 1.0
        if scale_low_n and n > 0:
            scale = max(0.25, min(1.0, n / VOICE_REF_N))

        win_x.append(win_pct * scale)
        tie_x.append(tie_pct * scale)
        loss_x.append(loss_pct * scale)
        win_text.append(_bar_label(win_pct))
        tie_text.append(_bar_label(tie_pct))
        loss_text.append(_bar_label(loss_pct))
        hovers.append(_hover_outcome(name, wins, losses, ties, n))

    text_kw = dict(textposition="inside", insidetextanchor="middle", textfont=dict(size=11, color="white"))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=win_x, orientation="h", name="Falcon wins",
        marker_color=COLOR_FALCON, text=win_text,
        customdata=hovers, hovertemplate="%{customdata}",
        **text_kw,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=tie_x, orientation="h", name="Ties",
        marker_color=COLOR_TIE, text=tie_text,
        customdata=hovers, hovertemplate="%{customdata}",
        textfont=dict(size=11, color="#334155"),
        textposition="inside", insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=loss_x, orientation="h", name="Competitor wins",
        marker_color=COLOR_COMPETITOR, text=loss_text,
        customdata=hovers, hovertemplate="%{customdata}",
        **text_kw,
    ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="Share of comparisons (%)",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(labels))),
    )
    bar_h = max(280, 44 * len(labels))
    return _chart_layout(fig, bar_h, margin_l=200, title=title)


def forest_plot(
    rows: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """One row per provider: dot at score, horizontal CI line, n= in label."""
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", x=0.5, y=0.5, showarrow=False)
        return fig

    ordered = sorted(rows, key=lambda r: float(r.get("elo", 0)), reverse=True)
    labels = [
        f"n={r.get('n_comparisons', 0)}  {r['provider_name']}"
        + (" ⚓" if r.get("is_anchor") else "")
        for r in ordered
    ]
    y_pos = list(range(len(ordered)))

    fig = go.Figure()
    for i, r in enumerate(ordered):
        lo, hi = float(r["elo_ci_low"]), float(r["elo_ci_high"])
        score = float(r["elo"])
        is_anchor = bool(r.get("is_anchor"))
        color = COLOR_ANCHOR if is_anchor else COLOR_FALCON
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[i, i],
            mode="lines",
            line=dict(color=COLOR_CI, width=6),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[score], y=[i],
            mode="markers",
            marker=dict(color=color, size=12, symbol="circle"),
            showlegend=False,
            customdata=[[r["provider_name"], lo, hi, r.get("n_comparisons", 0)]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Score: %{x:.1f}<br>"
                "CI: %{customdata[1]:.0f}–%{customdata[2]:.0f}<br>"
                "n=%{customdata[3]}"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[score], y=[i + 0.32],
            mode="text",
            text=[f"{score:.0f}"],
            textfont=dict(size=11, color=COLOR_SCORE_LABEL),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        xaxis_title="Score",
        yaxis=dict(
            tickmode="array",
            tickvals=y_pos,
            ticktext=labels,
            autorange="reversed",
        ),
    )
    return _chart_layout(fig, max(360, 58 * len(ordered)), margin_l=220, title=title)


def language_winrate_heatmap(
    cells: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """Providers × languages; cell color = Falcon win rate (teal high, coral low)."""
    if not cells:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", x=0.5, y=0.5, showarrow=False)
        return fig

    languages = sorted({c["language"] for c in cells}, key=lambda x: x)
    providers = sorted({c["competitor"] for c in cells})
    lookup = {(c["language"], c["competitor"]): c for c in cells}

    z, text, custom = [], [], []
    for prov in providers:
        z_row, t_row, c_row = [], [], []
        for lang in languages:
            cell = lookup.get((lang, prov))
            if cell and cell.get("n", 0) > 0:
                rate = float(cell.get("win_rate", 0.5))
                n = int(cell["n"])
                z_row.append(rate * 100)
                t_row.append(f"{rate * 100:.0f}%<br>n={n}")
                c_row.append(n)
            else:
                z_row.append(None)
                t_row.append("—")
                c_row.append(0)
        z.append(z_row)
        text.append(t_row)
        custom.append(c_row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=languages,
        y=providers,
        text=text,
        texttemplate="%{text}",
        hovertemplate=(
            "Provider: %{y}<br>Language: %{x}<br>"
            "Falcon win rate: %{z:.1f}%<br>n=%{customdata}<extra></extra>"
        ),
        customdata=custom,
        colorscale=[
            [0.0, COLOR_COMPETITOR],
            [0.5, "#f8fafc"],
            [1.0, COLOR_FALCON],
        ],
        zmin=0,
        zmax=100,
        colorbar=dict(title="Falcon win %", ticksuffix="%"),
    ))
    fig.update_layout(
        xaxis_title="Language",
        yaxis_title="Provider",
        yaxis=dict(autorange="reversed"),
    )
    return _chart_layout(fig, max(360, 40 * len(providers)), margin_l=160, title=title)


def prep_head_to_head(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "name": r["competitor_name"],
        "wins": r["anchor_preferred"],
        "losses": r["competitor_preferred"],
        "ties": r["ties"],
        "n": r["n"],
    } for r in rows]


def prep_voice_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "name": (
            f"{r['voice_name']} ({r['voice']})"
            if r.get("voice_name") != r.get("voice")
            else r["voice"]
        ),
        "wins": r["anchor_preferred"],
        "losses": r["competitor_preferred"],
        "ties": r["ties"],
        "n": r["n"],
    } for r in rows]


def prep_heatmap_cells(
    report,
    fits: Dict[str, Any],
    lang_display,
) -> List[Dict[str, Any]]:
    cells = []
    for lang in fits:
        label = lang_display(lang)
        for row in report.winrate_grid(lang):
            decisive = row["anchor_preferred"] + row["competitor_preferred"]
            rate = (row["anchor_preferred"] / decisive) if decisive else 0.5
            cells.append({
                "language": label,
                "competitor": row["competitor_name"],
                "win_rate": rate,
                "n": row["n"],
            })
    return cells
