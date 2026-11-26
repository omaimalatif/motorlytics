"""Overview: greeting, car summary, quick actions, last trip, recent trips, top tip, fuel and driving charts."""
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from core import cars, weather
from ui import components as ui
from ui import session

u = session.require_user()
car = session.active_car(u)
t = session.trips_df(u, car["id"] if car else None)
P = session.PAGES

hour = datetime.now().hour
greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
cal_label, cal_kind, cal_text = session.calibration_state(car)
calibrated = cal_label == "calibrated"
status = [session.plural(len(t), "trip") if len(t) else "no trips yet", cars.car_label(car) if car else "no car added",
          cal_label]
ui.page_header(f"{greet}, {u['name'].split()[0]}", " · ".join(status))

if not car:
    ui.empty_state("Add your car to start: engine size is needed for the predictions.", "Add your car", P["car"])
    st.stop()

# ---------------- row 1: hero car card [2/3] + quick actions and calibration [1/3] ----------------
c1, c2 = st.columns([2, 1])
with c1:
    with ui.card():
        h_pic, h_txt = st.columns([1, 3])
        with h_pic:
            ui.car_photo(car, caption=False)
        with h_txt:
            st.subheader(cars.car_title(car))
            st.caption(" · ".join(x for x in [session.fmt(car.get("disp_L"), "L", 1) if car.get("disp_L") else None,
                                              car.get("transmission"), car.get("fuel_type")] if x)
                       or "Add transmission and fuel type on the My car page.")
        month = t[(t["started"].dt.year == datetime.now().year) & (t["started"].dt.month == datetime.now().month)] \
            if len(t) else t
        s1, s2, s3 = st.columns(3)
        with s1:
            ui.metric_tile("Average L/100 km", session.fmt(t["l100"].mean() if len(t) else None, digits=1),
                           help="Measured where the log has MAF, otherwise the model's estimate.")
        with s2:
            cost = month["cost_pkr"].sum(min_count=1) if len(month) else None
            ui.metric_tile("PKR this month", session.fmt(cost, digits=0) if u.get("petrol_pkr") else "set price",
                           help="Needs your petrol price (Account page).")
        with s3:
            ui.metric_tile("Average eco-score", session.fmt(t["eco_score"].mean() if len(t) else None, digits=0),
                           help="0-100, higher is calmer driving.")
with c2:
    with ui.card("Quick actions"):
        a1, a2 = st.columns(2)
        if a1.button("Upload trip", icon=":material/upload_file:", width="stretch"):
            st.switch_page(P["upload"])
        if a2.button("Log fill-up", icon=":material/local_gas_station:", width="stretch"):
            st.switch_page(P["calibration"])
        if a1.button("Calibrate", icon=":material/tune:", width="stretch"):
            st.switch_page(P["calibration"])
        with a2.popover("Weather", icon=":material/thermostat:", width="stretch"):
            w = weather.current(u.get("city")) if u.get("city") else None
            if w:
                st.metric(f"{w['city']}, {w['country']}", f"{w['temp_c']:.1f} °C", help="Open-Meteo, updated every 30 min")
            else:
                st.caption("Set your city on the Account page to see the temperature.")
    with ui.card("Calibration"):
        ui.badge(cal_label, cal_kind)
        st.caption(cal_text)

if t.empty:
    ui.empty_state("No trips yet. Upload an OBD-II log (Torque, Car Scanner, enviroCar or VED CSV).",
                   "Upload a trip", P["upload"])
    st.stop()

# ---------------- row 2: last route [1/2] + recent trips [1/4] + top tip [1/4] ----------------
last = t.iloc[0]
r1, r2, r3 = st.columns([2, 1, 1])
with r1:
    with ui.card("Last trip", f"{last['started']:%d %b %Y, %H:%M} · {last['dist_km']:.1f} km · {last['l100']:.1f} L/100 km"):
        ui.route_map((last["route"] or {}).get("path"), height=300, key="overview_map")
with r2:
    with ui.card("Recent trips"):
        for _, r in t.head(5).iterrows():
            label = f"{r['started']:%d %b} · {r['dist_km']:.1f} km · {r['l100']:.1f} L/100"
            if st.button(label, key=f"recent_{r['id']}", width="stretch"):
                session.open_trip(r["id"])
        if st.button("All trips", icon=":material/history:", width="stretch"):
            st.switch_page(P["history"])
with r3:
    with ui.card("Top eco tip"):
        best = None
        for _, r in t.head(10).iterrows():                         # biggest saving among recent trips
            for tip in (r["tips"] or {}).get("tips", []):
                if best is None or tip["saving_l_per_100km"] > best[0]["saving_l_per_100km"]:
                    best = (tip, r["id"])
        if best:
            st.markdown(best[0]["tip"])
            st.caption(f"Up to {best[0]['saving_l_per_100km']} L/100 km on that trip")
            if st.button("See the trip", key="tip_trip"):
                session.open_trip(best[1])
        else:
            st.caption("No tips: your recent trips already look efficient.")

# ---------------- row 3: fuel by day + driving by hour ----------------
g1, g2 = st.columns(2)
with g1:
    with ui.card("Fuel use by day (litres)"):
        today = datetime.now().date()
        days = pd.date_range(today - timedelta(days=13), today).date
        by_day = t.groupby("date")["litres"].sum().reindex(days, fill_value=0)
        note = None
        if by_day.sum() == 0:                                       # no trips in the last 2 weeks
            by_day = t.groupby("date")["litres"].sum().sort_index().tail(14)
            note = "No trips in the last 14 days: showing your last 14 driving days."
        d = pd.DataFrame({"day": [str(x) for x in by_day.index], "litres": by_day.values,
                          "today": ["today" if x == today else "other" for x in by_day.index]})
        fig = px.bar(d, x="day", y="litres", color="today", color_discrete_map={"today": "#e4572e", "other": "#4c78a8"})
        fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        if note:
            st.caption(note)
with g2:
    with ui.card("Driving by hour of day (km)"):
        by_hour = t.groupby("hour")["dist_km"].sum().reindex(range(24), fill_value=0)
        fig = px.area(x=by_hour.index, y=by_hour.values, labels={"x": "hour", "y": "km"})
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
