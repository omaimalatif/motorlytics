"""Plotly figures shared by the pages, in the motorlytics template (ui/theme.py). Figures only: pages show them
with theme.plot(fig, height)."""
import pandas as pd
import plotly.graph_objects as go

from ui import theme

# eco-score bands, same thresholds as core.scoring.score_label
BANDS = [(0, 40, "aggressive"), (40, 70, "normal"), (70, 100, "eco")]


def eco_gauge(score, style: str = None) -> go.Figure:
    """Radial 0-100 gauge, bar coloured by the score's style band (the page shows the style name as a pill).
    The number sits under the arc (not inside it), so it never overlaps the arc in a narrow column."""
    from core import scoring
    band = scoring.score_label(score)
    color = theme.STYLE_COLORS.get(band, theme.MUTED)
    value = float(score) if score is not None and score == score else 0
    fig = go.Figure(go.Indicator(
        mode="gauge", value=value,
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 40, 70, 100], tickfont=dict(color=theme.CHART_TEXT, size=11),
                             tickcolor="rgba(0,0,0,0)"),
                   bar=dict(color=color, thickness=0.32), bgcolor="#CBD5E1", borderwidth=0,
                   steps=[dict(range=[a, b], color=theme._hex_rgba(theme.STYLE_COLORS[n], .45)) for a, b, n in BANDS]),
        domain=dict(x=[0.08, 0.92], y=[0.3, 1])))         # narrower arc: room for the "100" end label
    fig.add_annotation(x=0.5, y=0.0, xref="paper", yref="paper", xanchor="center", yanchor="bottom", showarrow=False,
                       text=f"<b>{value:.0f}</b>",
                       font=dict(size=32, color=theme.INK, family="Plus Jakarta Sans, Inter, sans-serif"))
    fig.update_layout(margin=dict(l=24, r=24, t=22, b=4))   # side room for the 0 / 100 end labels
    return fig


def speed_area(sp: list, stop_kmh: float = 2.0) -> go.Figure:
    """Speed over time: dark orange line with a filled area, stopped periods shaded grey. sp: [[minute, km/h], ...]."""
    d = pd.DataFrame(sp, columns=["minute", "speed"])
    fig = go.Figure()
    stopped = d["speed"] <= stop_kmh
    runs = (stopped != stopped.shift()).cumsum()
    first = True
    for _, g in d[stopped].groupby(runs[stopped]):
        if len(g) < 2 and len(d) > 2:
            continue
        x0, x1 = float(g["minute"].iloc[0]), float(g["minute"].iloc[-1])
        if x1 - x0 < 0.05:
            continue
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(15,23,42,0.14)", line_width=0, layer="below")
        if first:                                          # one legend entry for the shading
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="stopped",
                                     marker=dict(size=10, symbol="square", color="rgba(15,23,42,0.35)")))
            first = False
    fig.add_trace(go.Scatter(
        x=d["minute"], y=d["speed"], mode="lines", name="speed", line=dict(color=theme.CHART_ORANGE, width=2.5),
        fill="tozeroy", fillcolor="rgba(194,65,12,0.28)",
        hovertemplate="%{x:.1f} min · %{y:.0f} km/h<extra></extra>"))
    fig.update_xaxes(title="minutes")
    fig.update_yaxes(title="km/h", rangemode="tozero")
    fig.update_layout(showlegend=not first, hovermode="x unified")
    return fig
