"""Overview: greeting, car hero card, monthly KPIs with sparklines, recent trips, last route, top tip, charts."""
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import cars, texts, weather
from ui import components as ui
from ui import session
from ui import theme

u = session.require_user()
car = session.active_car(u)
t = session.trips_df(u, car["id"] if car else None)
P = session.PAGES

hour = datetime.now().hour
greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
cal_label, cal_kind, cal_text = session.calibration_state(car)
status = [session.plural(len(t), "trip") if len(t) else "no trips yet", cars.car_label(car) if car else "no car added",
          cal_label]
act = ui.page_header(f"{greet}, {u['name'].split()[0]}", " · ".join(status), actions=True)
with act:
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Upload trip", type="primary", icon=":material/upload_file:", key="hdr_upload"):
            st.switch_page(P["upload"])

if not car:
    ui.empty_state("Add your car to start: engine size is needed for the predictions.", "Add your car", P["car"])
    st.stop()

# ---------------- hero: car card [2/3] + quick actions [1/3] ----------------
c1, c2 = st.columns([3, 2], gap="medium")      # quick actions wide enough for full labels
with c1:
    with ui.card(kind="hero"):
        has_pic = cars.car_image(car) is not None
        info, pic = st.columns([3, 2], vertical_alignment="center") if has_pic else (st.container(), None)
        with info:
            st.subheader(cars.car_title(car))
            specs = [cars.fmt_disp(car.get("disp_L")) if car.get("disp_L") else None,
                     f"{car['cylinders']} cylinders" if car.get("cylinders") else None,
                     car.get("transmission"), car.get("fuel_type")]
            st.caption(" · ".join(x for x in specs if x) or "Add transmission and fuel type on the My car page.")
            ui.badge(cal_label, cal_kind)
            st.caption(cal_text)
        if pic is not None:
            with pic:
                ui.car_image(car)
with c2:
    with ui.card("Quick actions"):
        a1, a2 = st.columns(2)
        if a1.button("Log fill-up", icon=":material/local_gas_station:", width="stretch"):
            st.switch_page(P["calibration"])
        if a2.button("Estimate", icon=":material/calculate:", width="stretch"):
            st.switch_page(P["quick"])
        if a1.button("Calibrate", icon=":material/tune:", width="stretch"):
            st.switch_page(P["calibration"])
        with a2.popover("Weather", icon=":material/thermostat:", width="stretch"):
            w = weather.current(u.get("city")) if u.get("city") else None
            if w:
                st.metric(f"{w['city']}, {w['country']}", f"{w['temp_c']:.1f} °C", help="Open-Meteo, updated every 30 min")
            else:
                st.caption("Set your city on the Account page to see the temperature.")

if t.empty:
    ui.empty_state("No trips yet. Upload an OBD-II log (Torque, Car Scanner, enviroCar or VED CSV).",
                   "Upload a trip", P["upload"])
    st.stop()

# ---------------- KPI row: this month vs last month, sparkline per day ----------------
now = pd.Timestamp(datetime.now())
dated = t.dropna(subset=["started"])
this_p = now.to_period("M")
if len(dated) and not (dated["started"].dt.to_period("M") == this_p).any():
    this_p = dated["started"].max().to_period("M")          # no trips this month: the latest month with trips
prev_p = this_p - 1
# the period is said once above the tiles (labels stay short)
if this_p == now.to_period("M"):
    st.caption(f":material/calendar_month: This month ({this_p.strftime('%b %Y')})")
else:
    st.caption(f":material/calendar_month: Latest month with trips: {this_p.strftime('%b %Y')}")
m_now = dated[dated["started"].dt.to_period("M") == this_p]
m_prev = dated[dated["started"].dt.to_period("M") == prev_p]


def month_vals(m):
    km = m["dist_km"].sum()
    return {"pkr": m["cost_pkr"].sum(min_count=1) if len(m) else None, "km": km if len(m) else None,
            "l100": m["litres"].sum() / km * 100 if km else None, "eco": m["eco_score"].mean() if len(m) else None}


cur, prev = month_vals(m_now), month_vals(m_prev)


def change(key, digits=0, pct=True):
    a, b = cur[key], prev[key]
    if a is None or b is None or pd.isna(a) or pd.isna(b) or not b:
        return None
    return f"{(a - b) / b * 100:+.0f}%" if pct else f"{a - b:+.{digits}f}"


def vs_text(delta):
    return f"vs {prev_p.strftime('%b %Y')}" if delta else None


def spark(col, agg="sum"):
    d = m_now.groupby(m_now["started"].dt.date)[col].agg(agg).sort_index()
    return d.round(2).tolist() if len(d) >= 2 else None


l100_daily = m_now.groupby(m_now["started"].dt.date).apply(
    lambda g: g["litres"].sum() / g["dist_km"].sum() * 100 if g["dist_km"].sum() else None,
    include_groups=False).dropna().round(2).tolist() if len(m_now) else []
k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.metric_tile("Cost (PKR)", session.fmt(cur["pkr"], digits=0) if u.get("petrol_pkr") else "set price",
                   delta=change("pkr") if u.get("petrol_pkr") else None, delta_color="inverse",
                   delta_description=vs_text(change("pkr") if u.get("petrol_pkr") else None),
                   chart_data=spark("cost_pkr"), accent=True, help="Needs your petrol price (Account page).")
with k2:
    ui.metric_tile("Distance (km)", session.fmt(cur["km"], digits=0), delta=change("km"),
                   delta_color="off", delta_description=vs_text(change("km")), chart_data=spark("dist_km"))
with k3:
    ui.metric_tile("Average (L/100 km)", session.fmt(cur["l100"], digits=1), delta=change("l100"),
                   delta_color="inverse", delta_description=vs_text(change("l100")),
                   chart_data=l100_daily if len(l100_daily) >= 2 else None,
                   help="Distance-weighted; measured where the log has MAF, otherwise the model's estimate.")
with k4:
    ui.metric_tile("Eco-score", session.fmt(cur["eco"], digits=0), delta=change("eco", pct=False),
                   delta_description=vs_text(change("eco", pct=False)), chart_data=spark("eco_score", "mean"), help="0-100, higher is calmer driving.")

# ---------------- recent trips [1/3] + last route [2/3] ----------------
last = t.iloc[0]
r1, r2 = st.columns([1, 2], gap="medium")
with r1:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("**Recent trips**")
        if st.button("All trips", icon=":material/history:", type="tertiary", key="all_trips"):
            st.switch_page(P["history"])
    for _, r in t.head(4).iterrows():
        with ui.card(kind="click"):
            top, btn = st.columns([2, 1], vertical_alignment="center")
            with top:
                st.markdown(f"**{session.fmt_dt(r['started'], '%d %b %Y, %H:%M')}**")
                st.caption(f"{r['dist_km']:.1f} km · {r['l100']:.1f} L/100 km · {r['road_type'] or '?'} road")
                if r["style"]:
                    ui.style_badge(r["style"])
            with btn:
                if st.button("Open", key=f"recent_{r['id']}", width="stretch"):
                    session.open_trip(r["id"])
with r2:
    with ui.card("Last trip", f"{session.fmt_dt(last['started'])} · {last['dist_km']:.1f} km · {last['l100']:.1f} L/100 km"):
        path = (last["route"] or {}).get("path")
        # phones: the map engine is the heaviest part of this page, so it loads only when asked for
        if session.is_phone() and path and not st.session_state.get("overview_show_map"):
            if st.button("Show route map", icon=":material/map:", key="overview_map_btn", width="stretch"):
                st.session_state["overview_show_map"] = True
                st.rerun()
        else:
            ui.route_map(path, height=300 if session.is_phone() else 330, key="overview_map", width_px=560)

# ---------------- top tip ----------------
best = None
for _, r in t.head(10).iterrows():                         # biggest saving among recent trips
    for tip in (r["tips"] or {}).get("tips", []):
        if best is None or tip["saving_l_per_100km"] > best[0]["saving_l_per_100km"]:
            best = (tip, r["id"])
with ui.card(kind="hero"):
    tip_txt, tip_btn = st.columns([4, 1], vertical_alignment="center")
    with tip_txt:
        st.markdown("**:material/lightbulb: Top eco tip**")
        if best:
            st.markdown(best[0]["tip"])
            st.caption(texts.tip_comparison(best[0]))
            st.markdown(f"Possible saving **{texts.saving_text(best[0]['saving_l_per_100km'])}**")
        else:
            st.caption("No tips: your recent trips already look efficient.")
    with tip_btn:
        if best and st.button("See the trip", key="tip_trip", icon=":material/route:"):
            session.open_trip(best[1])

# ---------------- fuel by day + driving by hour ----------------
g1, g2 = st.columns(2, gap="medium")
with g1:
    today = datetime.now().date()
    days = pd.date_range(today - timedelta(days=13), today).date
    by_day = t.groupby("date")["litres"].sum().reindex(days, fill_value=0)
    note = None
    if by_day.sum() == 0:                                       # no trips in the last 2 weeks
        by_day = t.groupby("date")["litres"].sum().sort_index().tail(14)
        note = "No trips in the last 14 days: showing your last 14 driving days."
    top_day = by_day.idxmax() if len(by_day) and by_day.max() > 0 else None
    take = (f"Most fuel on **{pd.Timestamp(top_day):%d %b %Y}**: {by_day.max():.1f} L." if top_day is not None else None)
    with ui.chart_card("Fuel use by day (litres)", note or None):
        labels = [pd.Timestamp(x).strftime("%d %b %Y") for x in by_day.index]
        fig = go.Figure(go.Bar(x=labels, y=by_day.values, marker_color=[theme.CHART_ORANGE if x == today else theme.SINGLE
                                                                        for x in by_day.index],
                               hovertemplate="%{x}: %{y:.2f} L<extra></extra>"))
        # day labels on a category axis: plotly never turns it into a time axis with sub-day ticks
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels, title=None)
        fig.update_yaxes(title="litres")
        theme.plot(fig, 280)
        if take:
            st.caption(take.replace("**", ""))
with g2:
    by_hour = t.groupby("hour")["dist_km"].sum().reindex(range(24), fill_value=0)   # 24 bars, zero-filled
    peak = int(by_hour.idxmax()) if by_hour.max() > 0 else None
    with ui.chart_card("Driving by hour of day (km)"):
        fig = go.Figure(go.Bar(x=list(range(24)), y=by_hour.values, marker_color=theme.SINGLE,
                               hovertemplate="%{x}:00: %{y:.1f} km<extra></extra>"))
        fig.update_xaxes(type="linear", tickmode="array", tickvals=list(range(0, 24, 2)), range=[-0.5, 23.5],
                         title="hour of day")
        fig.update_yaxes(title="km")
        theme.plot(fig, 280)
        if peak is not None:
            st.caption(f"You drive most around {peak:02d}:00.")
