"""Trends across trips: fuel over time, by road type and style, measured vs predicted, cost by month, eco-score.
Each chart sits in a card with a one-line takeaway; each figure carries an id in layout.meta (used by the tests)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import components as ui
from ui import session
from ui import theme

u = session.require_user()
ui.page_header("Analytics", "Your trips over time and what drives your fuel use.")
car = session.active_car(u)
all_cars = st.toggle("All my cars", value=False)
t = session.trips_df(u, None if all_cars else (car["id"] if car else None))
if t.empty:
    ui.empty_state("No trips yet.", "Upload a trip", session.PAGES["upload"])
    st.stop()
if len(t) < 3:
    st.caption("Charts fill in as you upload more trips.")

k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.metric_tile("Trips", f"{len(t)}")
with k2:
    ui.metric_tile("Distance (km)", session.fmt(t["dist_km"].sum(), digits=0))
with k3:
    ui.metric_tile("Fuel (L)", session.fmt(t["litres"].sum(), digits=1))
with k4:
    ui.metric_tile("Average (L/100 km)", session.fmt(t["litres"].sum() / t["dist_km"].sum() * 100, digits=1),
                   help="Distance-weighted.", accent=True)

ROADS, STYLES = ["city", "mixed", "highway"], ["eco", "normal", "aggressive"]


def empty(text: str) -> None:
    st.info(text, icon=":material/hourglass_empty:")


# ---------------- fuel over time: trips + 7-trip rolling average ----------------
d = t.dropna(subset=["started"]).sort_values("started").assign(day=lambda x: x["started"].dt.normalize())
roll = d["l100"].rolling(7, min_periods=1).mean()
take = None
if len(d) >= 3:
    last_v, avg_v = d["l100"].iloc[-1], roll.iloc[-1]
    take = (f"Latest trip {last_v:.1f} L/100 km, {'above' if last_v > avg_v else 'below'} your 7-trip average of "
            f"{avg_v:.1f}.")
with ui.chart_card("Fuel use over time", take or "Each dot is a trip; the line is the average of the last 7 trips."):
    if len(d) < 2:
        empty("Upload at least 2 trips to see your fuel use over time.")
    else:
        fig = go.Figure()
        for src, color, name in [("measured", theme.MEASURED, "measured (MAF)"), ("predicted", theme.PREDICTED, "predicted")]:
            g = d[d["source"] == src]
            if len(g):
                fig.add_trace(go.Scatter(x=g["day"], y=g["l100"], mode="markers", name=name,
                                         marker=dict(size=10, color=color, line=dict(width=1, color=theme.BG)),
                                         customdata=g[["dist_km"]].to_numpy(),
                                         text=g["started"].dt.strftime("%d %b %Y %H:%M"),
                                         hovertemplate="%{text}<br>%{y:.1f} L/100 km · %{customdata[0]:.1f} km<extra></extra>"))
        if len(d) >= 3:
            fig.add_trace(go.Scatter(x=d["day"], y=roll, mode="lines", name="7-trip average",
                                     line=dict(color=theme.CHART_ORANGE, width=3, dash="dot", shape="spline"),
                                     hovertemplate="7-trip average %{y:.1f} L/100 km<extra></extra>"))
        lo, hi = d["day"].min(), d["day"].max()
        pad = pd.Timedelta(days=3) if (hi - lo) < pd.Timedelta(days=7) else pd.Timedelta(days=1)
        span_days = max(1, ((hi + pad) - (lo - pad)).days)
        step = max(1, -(-span_days // 10))                 # whole days, about 10 ticks: never sub-day ticks
        fig.update_xaxes(type="date", tickformat="%d %b %Y", range=[lo - pad, hi + pad], dtick=step * 86_400_000)
        fig.update_yaxes(rangemode="tozero", title="L/100 km")
        fig.update_layout(meta="fuel_over_time")
        theme.plot(fig, 320)

a1, a2 = st.columns(2, gap="medium")
# ---------------- road type x style: grouped, labelled bars ----------------
with a1:
    s = t.dropna(subset=["style", "road_type"]).groupby(["road_type", "style"], as_index=False).agg(
        l100=("l100", "median"), trips=("id", "size"))
    take = "Median L/100 km per road type and driving style."
    for road in ROADS:
        e = s[(s["road_type"] == road) & (s["style"] == "eco")]
        a = s[(s["road_type"] == road) & (s["style"] == "aggressive")]
        if len(e) and len(a) and a["l100"].iloc[0] > 0:
            take = (f"On {road} roads your eco trips used {(1 - e['l100'].iloc[0] / a['l100'].iloc[0]) * 100:.0f}% "
                    "less fuel than aggressive ones.")
            break
    with ui.chart_card("Fuel use by road type and driving style", take):
        if s.empty:
            empty("No trips with a driving style yet.")
        else:
            fig = go.Figure()
            for style in STYLES:
                g = s[s["style"] == style]
                if len(g):
                    fig.add_trace(go.Bar(x=g["road_type"], y=g["l100"], name=style, marker_color=theme.STYLE_COLORS[style],
                                         text=[session.plural(n, "trip") for n in g["trips"]], textposition="outside",
                                         cliponaxis=False,
                                         hovertemplate=f"{style}, %{{x}}: %{{y:.1f}} L/100 km<extra></extra>"))
            fig.update_xaxes(categoryorder="array", categoryarray=[r for r in ROADS if r in set(s["road_type"])])
            fig.update_yaxes(range=[0, float(s["l100"].max()) * 1.25], title="median L/100 km")
            fig.update_layout(barmode="group", meta="style_road")
            theme.plot(fig, 320)

# ---------------- measured vs predicted ----------------
with a2:
    m = t[t["measured_l100"].notna()].assign(
        calibrated_label=lambda x: x["calibrated"].fillna(0).astype(int).map({1: "yes", 0: "no"}))
    take = None
    if len(m):
        mae = ((m["pred_l100"] - m["measured_l100"]).abs() / m["measured_l100"]).median() * 100
        take = f"The model was {mae:.0f}% off (median) on your {session.plural(len(m), 'trip')} with MAF."
    with ui.chart_card("Measured vs predicted", take):
        if not len(m):
            empty("No trips with MAF yet: the measured fuel needs a log with the MAF sensor.")
        else:
            top = float(max(m["measured_l100"].max(), m["pred_l100"].max()) * 1.15)
            fig = go.Figure()
            for lab, color, name in [("no", theme.PREDICTED, "no"), ("yes", theme.MEASURED, "yes")]:
                g = m[m["calibrated_label"] == lab]
                if len(g):
                    fig.add_trace(go.Scatter(x=g["measured_l100"], y=g["pred_l100"], mode="markers", name=name,
                                             legendgrouptitle_text="calibrated", legendgroup="cal",
                                             marker=dict(size=10, color=color, line=dict(width=1, color=theme.BG)),
                                             hovertemplate="measured %{x:.1f} · predicted %{y:.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect prediction",
                                     line=dict(dash="dash", color=theme.INK, width=1.5), hoverinfo="skip"))
            fig.update_xaxes(range=[0, top], title="measured L/100 km", showgrid=True, constrain="domain")
            fig.update_yaxes(range=[0, top], scaleanchor="x", scaleratio=1, title="predicted L/100 km", constrain="domain")
            fig.update_layout(meta="measured_vs_predicted")
            theme.plot(fig, 320)

b1, b2 = st.columns(2, gap="medium")
# ---------------- cost by month ----------------
with b1:
    has_cost = t["cost_pkr"].notna().any()
    mth = (t.dropna(subset=["started"]).assign(month=lambda x: x["started"].dt.to_period("M"))
           .groupby("month", as_index=False)["cost_pkr"].sum(min_count=1).dropna()) if has_cost else pd.DataFrame()
    take = None
    if len(mth):
        top_m = mth.loc[mth["cost_pkr"].idxmax()]
        take = f"Highest month: {top_m['month'].strftime('%b %Y')}, PKR {top_m['cost_pkr']:,.0f}."
    with ui.chart_card("Cost by month (PKR)", take):
        if not len(mth):
            empty("Set your petrol price on the Account page to see costs for new trips.")
        else:
            labels = [p.strftime("%b %Y") for p in mth["month"]]
            fig = go.Figure(go.Bar(x=labels, y=mth["cost_pkr"], marker_color=theme.SINGLE,
                                   text=[f"{v:,.0f}" for v in mth["cost_pkr"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x}: PKR %{y:,.0f}<extra></extra>"))
            fig.update_xaxes(type="category")
            fig.update_yaxes(range=[0, float(mth["cost_pkr"].max()) * 1.2], title="PKR")
            fig.update_layout(meta="cost_by_month")
            theme.plot(fig, 300)

# ---------------- eco-score distribution, coloured by style band ----------------
with b2:
    sc = t["eco_score"].dropna().clip(upper=99.9)          # a score of 100 goes in the 90-100 bar
    n_eco = int((sc >= 70).sum())
    with ui.chart_card("Eco-score distribution",
                       f"{n_eco} of {session.plural(len(sc), 'trip')} scored eco (70 or more)." if len(sc) else None):
        if len(sc) < 3:
            empty("Upload at least 3 trips to see how your eco-scores spread.")
        else:
            fig = go.Figure()
            for lo_, hi_, name in [(0, 40, "aggressive"), (40, 70, "normal"), (70, 110, "eco")]:
                part = sc[(sc >= lo_) & (sc < hi_)] if name != "eco" else sc[sc >= lo_]
                fig.add_trace(go.Histogram(x=part, name=name, marker_color=theme.STYLE_COLORS[name],
                                           xbins=dict(start=0, end=110, size=10),
                                           hovertemplate=f"{name}: eco-score %{{x}}, trips: %{{y}}<extra></extra>"))
            fig.update_xaxes(range=[0, 105], dtick=10, title="eco-score (0-100)")
            fig.update_yaxes(title="trips", dtick=1 if len(sc) < 10 else None)
            fig.update_layout(barmode="stack", bargap=0.06, meta="eco_distribution")
            theme.plot(fig, 300)
