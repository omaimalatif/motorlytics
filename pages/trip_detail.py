"""One trip: KPIs, route, eco-score, speed over time, tips, measured / calibrated badges."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import cars, db, scoring
from ui import components as ui
from ui import session

u = session.require_user()
P = session.PAGES

# opened from Trip history, Overview or after an upload (hidden from the sidebar); needs a selected trip
tid = st.session_state.get("trip_id") or st.query_params.get("trip")
trip = db.get_trip(int(tid), u["id"]) if tid and str(tid).isdigit() else None
if not trip:
    ui.page_header("Trip detail")
    ui.empty_state("No trip selected. Open a trip from Trip history, or upload one.", "Go to Trip history",
                   P["history"])
    st.stop()
st.session_state["trip_id"] = trip["id"]

started = pd.to_datetime(trip["started_at"], errors="coerce", format="mixed")
car = db.get_car(trip["car_id"]) if trip["car_id"] else None
info = trip["tips"] or {}
measured = trip["measured_l100"] is not None

last_upload = st.session_state.get("last_upload") or []
if len(last_upload) > 1 and trip["id"] in last_upload:               # several trips in the last upload
    pick = st.segmented_control("Trips from this upload", last_upload, default=trip["id"],
                                format_func=lambda i: f"Trip {last_upload.index(i) + 1}")
    if pick and pick != trip["id"]:
        session.open_trip(pick)

ui.page_header(f"Trip on {started:%d %b %Y, %H:%M}" if pd.notna(started) else f"Trip {trip['id']}",
               f"{session.display_car(car)} · {trip['road_type'] or '?'} road · {trip['style'] or '?'} style")
ui.badges([("measured (MAF)" if measured else "predicted", "good" if measured else "info"),
           (("calibrated", "good") if trip["calibrated"] else ("adjusted to PakWheels average", "info")
            if info.get("reference_adjusted") else ("uncalibrated", "warn")),
           (session.model_label(trip["model"]), "neutral")])
if info.get("quick_estimate"):
    inp = info.get("inputs") or {}
    st.info(f"Quick estimate, not a recorded trip: typical {inp.get('style', '')} driving on a {trip['road_type']} road "
            f"over {trip['dist_km']:.0f} km. There is no route, speed trace or tips.", icon=":material/calculate:")

# ---- measured vs the model's prediction ----
k = info.get("k") or 1.0
pred = trip["pred_l100"]
raw = pred / k if k else pred
if trip["calibrated"]:
    pred_label, pred_help = "Model predicted (after calibration)", (
        f"The model alone said {raw:.1f} L/100 km; your car's correction factor k = {k:.2f} gives {pred:.1f}.")
elif info.get("reference_adjusted"):
    pred_label, pred_help = "Model predicted (PakWheels-adjusted)", (
        f"The model alone said {raw:.1f} L/100 km; adjusted to the PakWheels average for this model: {pred:.1f}.")
else:
    pred_label, pred_help = "Model predicted", "The model's estimate from how the trip was driven."
cost_text = session.fmt(trip["cost_pkr"], "PKR", 0) if trip["cost_pkr"] is not None else "set price"
cost_help = "From the measured fuel and your petrol price (Account page)." if measured else     "From the predicted fuel and your petrol price (Account page)."
if measured:
    err = (pred - trip["measured_l100"]) / trip["measured_l100"] * 100
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.metric_tile("Measured", session.fmt(trip["measured_l100"], "L/100 km"),
                       help="Real fuel use, worked out from the MAF (air flow) sensor in your log.")
    with k2:
        ui.metric_tile(pred_label, session.fmt(pred, "L/100 km"), help=pred_help)
    with k3:
        ui.metric_tile("Model error", f"{err:+.0f}%", help="(predicted - measured) / measured. Positive: the "
                       "model expected more fuel than the car used.")
    with k4:
        ui.metric_tile("Cost", cost_text, help=cost_help)
else:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.metric_tile(pred_label, session.fmt(pred, "L/100 km"), help=pred_help + " This log has no MAF, so "
                       "the real fuel use cannot be measured.")
    with k2:
        ui.metric_tile("Distance", session.fmt(trip["dist_km"], "km", 2))
    with k3:
        ui.metric_tile("Duration", session.fmt(trip["dur_min"], "min", 1))
    with k4:
        ui.metric_tile("Cost", cost_text, help=cost_help)
if measured:
    d1, d2, d3 = st.columns(3)
    with d1:
        ui.metric_tile("Distance", session.fmt(trip["dist_km"], "km", 2))
    with d2:
        ui.metric_tile("Duration", session.fmt(trip["dur_min"], "min", 1))
    with d3:
        ui.metric_tile("Average speed", session.fmt(trip["avg_speed"], "km/h", 0))

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

st.caption(session.trip_calibration_line(trip))
with st.expander("Delete this trip"):
    if st.button("Delete trip", type="secondary", icon=":material/delete:"):
        db.delete_trip(trip["id"], u["id"])
        st.session_state.pop("trip_id", None)
        st.switch_page(P["history"])
