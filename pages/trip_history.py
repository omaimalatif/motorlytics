"""All trips in a filterable table; select a row to open the trip."""
import pandas as pd
import streamlit as st

from ui import components as ui
from ui import session
from ui import theme

u = session.require_user()
ui.page_header("Trip history", "Every trip you uploaded or estimated, newest first.")
t = session.trips_df(u)
if t.empty:
    ui.empty_state("No trips yet.", "Upload a trip", session.PAGES["upload"])
    st.stop()

with ui.card(kind="raised"):
    f1, f2, f3, f4, f5 = st.columns([1.3, 1, 1, 1, 1.3])
    car_f = f1.multiselect("Car", sorted(t["car"].dropna().unique()), placeholder="all cars")
    road_f = f2.multiselect("Road type", ["city", "mixed", "highway"], placeholder="all")
    style_f = f3.multiselect("Style", ["eco", "normal", "aggressive"], placeholder="all")
    src_f = f4.multiselect("Fuel figure", ["measured", "predicted"], placeholder="all")
    dmin, dmax = t["date"].min(), t["date"].max()
    dates = f5.date_input("Dates", value=(dmin, dmax), min_value=dmin, max_value=dmax)

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

show = v[["id", "started", "car", "road_type", "style", "dist_km", "avg_speed", "measured_l100", "pred_l100",
          "eco_score", "cost_pkr"]].reset_index(drop=True)
STYLES = ["eco", "normal", "aggressive"]
for col in ["pred_l100", "cost_pkr", "eco_score", "dist_km", "avg_speed"]:
    show[col] = pd.to_numeric(show[col], errors="coerce")
# measured is often missing (no MAF): show a dash instead of Streamlit's grey "None"
show["measured_l100"] = pd.to_numeric(show["measured_l100"], errors="coerce").map(
    lambda x: f"{x:.1f}" if pd.notna(x) else "–")
show["style"] = show["style"].map(lambda x: [x] if isinstance(x, str) and x else [])     # one coloured pill
st.caption(f"Showing {len(v)} of {session.plural(len(t), 'trip')} · {v['dist_km'].sum():.0f} km · {v['litres'].sum():.1f} L "
           "· fuel figures in L/100 km · select a row to open the trip")
ev = st.dataframe(
    show, hide_index=True, width="stretch", height=min(560, 38 + 35 * len(show)), on_select="rerun", selection_mode="single-row",
    column_config={
        "id": None, "started": st.column_config.DatetimeColumn("Start", format="DD MMM YYYY, HH:mm", width=140),
        "car": st.column_config.TextColumn("Car", width=165),
        "dist_km": st.column_config.NumberColumn("km", format="%.1f", width=60),
        "avg_speed": st.column_config.NumberColumn("km/h", format="%.0f", width=60),
        "road_type": st.column_config.TextColumn("Road", width=80),
        "measured_l100": st.column_config.TextColumn("Measured", width=80,
                                                     help="L/100 km from the MAF sensor; – when the log has no MAF."),
        "pred_l100": st.column_config.NumberColumn("Predicted", format="%.1f", width=80,
                                                   help="L/100 km, the model's estimate (after calibration if applied)."),
        "eco_score": st.column_config.ProgressColumn("Eco-score", min_value=0, max_value=100, width=100,
                                                                         format="%.0f"),
        "style": st.column_config.MultiselectColumn("Style", options=STYLES, width=110,
                                                    color=[theme.STYLE_COLORS[x] for x in STYLES]),
        "cost_pkr": st.column_config.NumberColumn("PKR", format="%.0f", width=70)})
rows = ev.selection.rows if ev and ev.selection else []
if rows:
    session.open_trip(show.loc[rows[0], "id"])
