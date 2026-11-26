"""One trip: KPIs, route, eco-score, speed over time, tips, measured / calibrated badges."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import cars, db, scoring
from ui import components as ui
from ui import session

u = session.require_user()
P = session.PAGES

tid = st.session_state.get("trip_id") or st.query_params.get("trip")
trips = db.get_trips(u["id"])
if not trips:
    ui.page_header("Trip detail")
    ui.empty_state("No trips yet.", "Upload a trip", P["upload"])
    st.stop()
trip = db.get_trip(int(tid), u["id"]) if tid else None
trip = trip or trips[0]
st.session_state["trip_id"] = trip["id"]

started = pd.to_datetime(trip["started_at"], errors="coerce")
car = db.get_car(trip["car_id"]) if trip["car_id"] else None
info = trip["tips"] or {}
measured = trip["measured_l100"] is not None
main = trip["measured_l100"] if measured else trip["pred_l100"]

last_upload = st.session_state.get("last_upload") or []
if len(last_upload) > 1 and trip["id"] in last_upload:               # several trips in the last upload
    pick = st.segmented_control("Trips from this upload", last_upload, default=trip["id"],
                                format_func=lambda i: f"Trip {last_upload.index(i) + 1}")
    if pick and pick != trip["id"]:
        session.open_trip(pick)

ui.page_header(f"Trip on {started:%d %b %Y, %H:%M}" if pd.notna(started) else f"Trip {trip['id']}",
               f"{cars.car_label(car)} · {trip['road_type'] or '?'} road · {trip['style'] or '?'} style")
ui.badges([("measured (MAF)" if measured else "predicted", "good" if measured else "info"),
           (("calibrated", "good") if trip["calibrated"] else ("adjusted to PakWheels average", "info")
            if info.get("reference_adjusted") else ("uncalibrated", "warn")),
           (f"model: {trip['model']}", "neutral")])

k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.metric_tile("Fuel use", session.fmt(main, "L/100 km"),
                   help=(f"Measured from MAF; the model predicted {trip['pred_l100']:.1f}" if measured
                         else "Model estimate (no MAF in the log)."))
with k2:
    ui.metric_tile("Distance", session.fmt(trip["dist_km"], "km", 2))
with k3:
    ui.metric_tile("Duration", session.fmt(trip["dur_min"], "min", 1), help=f"Average {trip['avg_speed']:.1f} km/h")
with k4:
    ui.metric_tile("Cost", session.fmt(trip["cost_pkr"], "PKR", 0) if trip["cost_pkr"] is not None else "set price",
                   help="From your petrol price (Account page).")

m1, m2 = st.columns([2, 1])
with m1:
    with ui.card("Route"):
        ui.route_map((trip["route"] or {}).get("path"), height=340, key="detail_map")
with m2:
    with ui.card("Eco-score"):
        score = trip["eco_score"]
        st.metric("Eco-score", session.fmt(score, digits=0) + " / 100", help="0-100 between the aggressive and "
                  "eco driving-style centres (k-Means).")
        st.progress(int(score or 0) / 100)
        ui.badge(scoring.score_label(score), {"eco": "good", "moderate": "info", "aggressive": "bad"}.get(
            scoring.score_label(score), "neutral"))
        st.caption(f"Idle {session.fmt(info.get('idle_pct'), '%', 0)} of the time.")
        if info.get("eco_l100") is not None:
            st.caption(f"Driven like an eco driver: about {info['eco_l100']:.1f} L/100 km "
                       f"(up to {info.get('possible_saving_pct', 0):.0f}% less).")

with ui.card("Speed over time"):
    sp = (trip["route"] or {}).get("speed") or []
    if sp:
        d = pd.DataFrame(sp, columns=["minute", "speed"])
        fig = px.line(d, x="minute", y="speed", labels={"minute": "minutes", "speed": "km/h"})
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No speed trace stored for this trip.")

st.subheader("Eco-driving tips")
tips = info.get("tips") or []
if tips:
    cols = st.columns(3)
    for c, tip in zip(cols, tips[:3]):
        with c:
            ui.tip_card(tip)
else:
    st.success("No tips: this trip already looks efficient for its road type.")

st.caption(info.get("calibration", ""))
with st.expander("Delete this trip"):
    if st.button("Delete trip", type="secondary", icon=":material/delete:"):
        db.delete_trip(trip["id"], u["id"])
        st.session_state.pop("trip_id", None)
        st.switch_page(P["history"])
