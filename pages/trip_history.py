"""All trips in a filterable table; select a row to open the trip."""
import streamlit as st

from ui import components as ui
from ui import session

u = session.require_user()
ui.page_header("Trip history", "Select a row to open the trip.")
t = session.trips_df(u)
if t.empty:
    ui.empty_state("No trips yet.", "Upload a trip", session.PAGES["upload"])
    st.stop()

f1, f2, f3, f4 = st.columns(4)
car_f = f1.multiselect("Car", sorted(t["car"].dropna().unique()))
road_f = f2.multiselect("Road type", ["city", "mixed", "highway"])
style_f = f3.multiselect("Style", ["eco", "normal", "aggressive"])
src_f = f4.multiselect("Fuel figure", ["measured", "predicted"])
dmin, dmax = t["date"].min(), t["date"].max()
dates = st.date_input("Dates", value=(dmin, dmax), min_value=dmin, max_value=dmax)

v = t
if car_f:
    v = v[v["car"].isin(car_f)]
if road_f:
    v = v[v["road_type"].isin(road_f)]
if style_f:
    v = v[v["style"].isin(style_f)]
if src_f:
    v = v[v["source"].isin(src_f)]
if isinstance(dates, tuple) and len(dates) == 2:
    v = v[(v["date"] >= dates[0]) & (v["date"] <= dates[1])]

show = v[["id", "started", "car", "dist_km", "dur_min", "avg_speed", "road_type", "measured_l100", "pred_l100",
          "eco_score", "style", "cost_pkr", "calibrated"]].reset_index(drop=True)
st.caption(f"{len(v)} of {session.plural(len(t), 'trip')} · {v['dist_km'].sum():.0f} km · {v['litres'].sum():.1f} L")
ev = st.dataframe(
    show, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    column_config={
        "id": None, "started": st.column_config.DatetimeColumn("Start", format="DD MMM YYYY, HH:mm"),
        "dist_km": st.column_config.NumberColumn("km", format="%.1f"),
        "dur_min": st.column_config.NumberColumn("min", format="%.0f"),
        "avg_speed": st.column_config.NumberColumn("km/h", format="%.0f"),
        "road_type": "Road",
        "measured_l100": st.column_config.NumberColumn("Measured L/100 km", format="%.1f",
                                                       help="From the MAF sensor; empty when the log has no MAF."),
        "pred_l100": st.column_config.NumberColumn("Predicted L/100 km", format="%.1f",
                                                   help="The model's estimate (after calibration if applied)."),
        "eco_score": st.column_config.ProgressColumn("Eco-score", min_value=0, max_value=100,
                                                                         format="%.0f"),
        "style": "Style", "cost_pkr": st.column_config.NumberColumn("PKR", format="%.0f"),
        "calibrated": st.column_config.CheckboxColumn("Calibrated")})
rows = ev.selection.rows if ev and ev.selection else []
if rows:
    session.open_trip(show.loc[rows[0], "id"])
