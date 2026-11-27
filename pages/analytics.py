"""Trends across trips: fuel over time, by road type and style, eco-score, measured vs predicted, cost."""
import pandas as pd
import plotly.express as px
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
                   help="Distance-weighted.")


def tidy(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0))
    return fig


ROADS, STYLES = ["city", "mixed", "highway"], ["eco", "normal", "aggressive"]
t = t.assign(figure=t["source"].map({"measured": "measured (MAF)", "predicted": "predicted"}),
             calibrated_label=t["calibrated"].fillna(0).astype(int).map({1: "yes", 0: "no"}))

a1, a2 = st.columns(2)
with a1:
    with ui.card("Fuel use over time"):
        d = t.dropna(subset=["started"]).sort_values("started").assign(day=lambda x: x["started"].dt.normalize())
        fig = px.scatter(d, x="day", y="l100", color="figure", labels={"day": "", "l100": "L/100 km", "figure": ""},
                         hover_data={"started": "|%d %b %Y %H:%M", "dist_km": ":.1f", "day": False})
        fig.update_traces(marker=dict(size=10))
        lo, hi = d["day"].min(), d["day"].max()
        pad = pd.Timedelta(days=3) if (hi - lo) < pd.Timedelta(days=7) else pd.Timedelta(days=1)
        span_days = max(1, ((hi + pad) - (lo - pad)).days)
        step = max(1, -(-span_days // 10))                 # whole days, about 10 ticks: never sub-day ticks
        fig.update_xaxes(type="date", tickformat="%d %b %Y", range=[lo - pad, hi + pad], dtick=step * 86_400_000)
        fig.update_yaxes(rangemode="tozero")
        theme.plot(fig)
with a2:
    with ui.card("Fuel use by road type"):
        counts = t["road_type"].value_counts()
        if counts.min() < 5 or len(counts) < len(t["road_type"].unique()):
            fig = px.strip(t, x="road_type", y="l100", category_orders={"road_type": ROADS},
                           labels={"road_type": "", "l100": "L/100 km"}, hover_data={"dist_km": ":.1f"})
            fig.update_traces(marker=dict(size=10))
        else:
            fig = px.box(t, x="road_type", y="l100", points="all", category_orders={"road_type": ROADS},
                         labels={"road_type": "", "l100": "L/100 km"})
        fig.update_yaxes(rangemode="tozero")
        theme.plot(fig)

b1, b2 = st.columns(2)
with b1:
    with ui.card("Eco-score per trip"):
        fig = go.Figure(go.Histogram(x=t["eco_score"], xbins=dict(start=0, end=110, size=10),
                                     hovertemplate="eco-score %{x}, trips: %{y}<extra></extra>"))
        fig.update_xaxes(range=[0, 105], dtick=10, title="eco-score (0-100)")
        fig.update_yaxes(title="trips", dtick=1 if len(t) < 10 else None)
        theme.plot(fig)
with b2:
    with ui.card("Fuel use by driving style and road type"):
        s = t.groupby(["road_type", "style"], as_index=False).agg(l100=("l100", "median"), trips=("id", "size"))
        s["label"] = s["trips"].map(lambda n: session.plural(n, "trip"))
        fig = px.bar(s, x="road_type", y="l100", color="style", barmode="group", text="label",
                     category_orders={"road_type": ROADS, "style": STYLES},
                     color_discrete_map={"eco": "#2ca02c", "normal": "#4c78a8", "aggressive": "#e4572e"},
                     labels={"road_type": "", "l100": "median L/100 km", "style": "style"},
                     hover_data={"trips": True, "label": False})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, float(s["l100"].max()) * 1.25])      # room for the tallest bar and its label
        theme.plot(fig)

c1, c2 = st.columns(2)
with c1:
    with ui.card("Measured vs model (trips with MAF)"):
        m = t[t["measured_l100"].notna()]
        if len(m):
            top = float(max(m["measured_l100"].max(), m["pred_l100"].max()) * 1.15)
            fig = px.scatter(m, x="measured_l100", y="pred_l100", color="calibrated_label",
                             category_orders={"calibrated_label": ["no", "yes"]},
                             color_discrete_map={"no": "#e4572e", "yes": "#4c78a8"},
                             labels={"measured_l100": "measured L/100 km", "pred_l100": "model L/100 km",
                                     "calibrated_label": "calibrated"})
            fig.update_traces(marker=dict(size=10))
            fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect prediction",
                                     line=dict(dash="dash", color="gray")))
            fig.update_xaxes(range=[0, top])
            fig.update_yaxes(range=[0, top], scaleanchor="x", scaleratio=1)
            theme.plot(fig, 320)
        else:
            st.caption("No trips with MAF yet.")
with c2:
    with ui.card("Cost by month (PKR)"):
        if t["cost_pkr"].notna().any():
            mth = t.assign(month=t["started"].dt.to_period("M").astype(str)).groupby("month", as_index=False)["cost_pkr"].sum()
            fig = px.bar(mth, x="month", y="cost_pkr", labels={"month": "", "cost_pkr": "PKR"})
            fig.update_xaxes(type="category")
            theme.plot(fig)
        else:
            st.caption("Set your petrol price on the Account page to see costs for new trips.")
