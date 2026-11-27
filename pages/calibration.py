"""Per-car calibration: status per model tier, fill-up form, fill-up history."""
from datetime import date

import pandas as pd
import streamlit as st

from core import app_core, cars, db, pk_reference
from ui import components as ui
from ui import session
from ui import theme

u = session.require_user()
ui.page_header("Calibration", "Each car gets a correction factor k = real fuel / model estimate, from trips with "
               "MAF or from fill-ups. k is the median of the 10 most recent calibration trips.")
car = session.active_car(u)
if not car:
    ui.empty_state("Add your car first.", "Add your car", session.PAGES["car"])
    st.stop()
st.subheader(cars.car_label(car))

status = app_core.calibration_status(car["car_key"])
tiers = session.MODEL_TYPES
cols = st.columns(3, gap="medium")
for c, (tier, label) in zip(cols, tiers.items()):
    with c:
        s = status.get(tier)
        with ui.card(label, kind="hero" if s else None):
            if s:
                ui.badge("calibrated", "good")
                off = (1 / s["k"] - 1) * 100 if s["k"] else 0
                with st.container(key=theme.next_key("mlkpi-accent")):
                    st.metric("k (real ÷ model)", f"{s['k']:.3f}", help="Predictions are multiplied by k.",
                              delta=f"model alone read {abs(off):.0f}% {'high' if off > 0 else 'low'}",
                              delta_color="gray", delta_arrow="off")
                st.caption(f"Before → after: model × 1.00 → × {s['k']:.2f}")
                st.caption(f"{session.plural(s['n_trips'], 'trip')} · last {s['last'][:10]}")
            else:
                ui.badge("uncalibrated", "warn")
                st.caption("Upload a log with MAF, or log a fill-up below.")

ref = pk_reference.match(car)
if "behaviour_only" not in status and ref:
    st.info(f"Until this car has its own calibration, speed-only predictions are adjusted to the PakWheels average "
            f"for a {ref['make']} {ref['model']} ({ref['city_kmpl']:.0f} km/L city, {ref['highway_kmpl']:.0f} km/L "
            "highway). Your own calibration below replaces that as soon as it exists, because it reflects your car, "
            "your driving and your roads.", icon=":material/info:")

t = session.trips_df(u, car["id"])
fills = db.get_fillups(car["id"])
left, right = st.columns([3, 2], gap="medium")
with left, ui.card(":material/local_gas_station: Log a fill-up", "Fill the tank to full, note the litres, and select the "
                   "trips driven since the previous full tank. k = litres ÷ predicted litres over those trips."):
    if t.empty:
        st.info("Upload the trips you drove since your last fill-up first.")
    else:
        covered = {i for f in fills for i in f["trip_ids"]}
        open_trips = t[~t["id"].isin(covered)]
        with st.form("fillup", border=False):
            c1, c2 = st.columns(2)
            litres = c1.number_input("Litres filled", min_value=0.1, value=None, step=0.1)
            when = c2.date_input("Fill-up date", value=date.today())
            opts = open_trips["id"].tolist()
            # trips with MAF already calibrated the car when they were uploaded: not pre-selected (no double counting)
            default = open_trips.loc[open_trips["measured_l100"].isna(), "id"].tolist()
            labels = {r["id"]: f"{session.fmt_dt(r['started'], '%d %b %Y %H:%M')} · {r['dist_km']:.1f} km"
                               + (" · MAF" if pd.notna(r["measured_l100"]) else "") for _, r in open_trips.iterrows()}
            sel = st.multiselect("Trips since the last fill-up", opts, default=default, format_func=lambda i: labels[i],
                                 help="Trips marked MAF already calibrated your car when uploaded; include them only "
                                      "if the fill-up covers them anyway.")
            if st.form_submit_button("Save fill-up", type="primary"):
                if not litres or not sel:
                    st.error("Enter the litres and select at least one trip.")
                else:
                    chosen = [db.get_trip(i, u["id"]) for i in sel]
                    try:
                        out = app_core.log_fillup(car, chosen, float(litres), str(when))
                        now = out["now"].get("behaviour_only")
                        st.success(f"Fill-up saved for {session.plural(len(chosen), 'trip')}."
                                   + (f" Speed-only predictions now use k = {now['k']:.2f} "
                                      f"(from {session.plural(now['n_trips'], 'trip')})." if now else ""))
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

with right, ui.card(":material/history: Fill-up history"):
    if fills:
        st.dataframe(pd.DataFrame([{"date": f["date"], "litres": f["litres"], "km": f["km"], "trips": len(f["trip_ids"]),
                                    "L/100 km": f["litres"] / f["km"] * 100 if f["km"] else None} for f in fills]),
                     hide_index=True, width="stretch",
                     column_config={"km": st.column_config.NumberColumn(format="%.1f"),
                                    "L/100 km": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.caption("No fill-ups yet.")
