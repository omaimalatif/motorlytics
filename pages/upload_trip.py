"""Upload one or more trip logs -> predict -> save as trips -> open the trip detail."""
import numpy as np
import streamlit as st

from core import app_core, cars, weather
from ui import components as ui
from ui import session

u = session.require_user()
P = session.PAGES
ui.page_header("Upload trip", "CSV logs from Torque, Car Scanner, enviroCar or VED. Several files = several trips.")

my_cars = session.user_cars(u)
if not my_cars:
    ui.empty_state("Add your car first: the engine size is needed for the prediction.", "Add your car", P["car"])
    st.stop()
car = session.active_car(u)

left, right = st.columns([2, 1])
with left:
    with ui.card("Log files"):
        files = st.file_uploader("Trip log (CSV)", type=["csv"], accept_multiple_files=True)
        st.caption("Needs at least time and speed. With MAF the real fuel use is measured and your car's "
                   "calibration improves; with RPM and engine load the engine models can be used.")
with right:
    with ui.card("Trip settings"):
        ids = [c["id"] for c in my_cars]
        cid = st.selectbox("Car", ids, index=ids.index(car["id"]),
                           format_func=lambda i: cars.car_title(next(c for c in my_cars if c["id"] == i)))
        w = weather.current(u.get("city")) if u.get("city") else None
        temp = st.number_input("Air temperature (°C)", value=float(w["temp_c"]) if w else None, step=0.5,
                               help="Filled from Open-Meteo for your city if set; change it for an older trip.")
        if not u.get("petrol_pkr"):
            st.warning("Set your petrol price on the Account page to see trip costs.")

chosen_car = next(c for c in my_cars if c["id"] == cid)
fuel = chosen_car.get("fuel_type") or "petrol"
if fuel != "petrol":
    st.warning(f"**{cars.car_title(chosen_car)} is set as a {fuel} car.** The model was trained on petrol "
               "(ICE) cars, so its predictions will be less reliable for this car."
               + (" For a CNG car, upload trips driven on petrol where possible." if fuel == "CNG" else "")
               + " A measured figure from MAF and your car's calibration still apply.",
               icon=":material/warning:")

if st.button("Analyse trip", type="primary", disabled=not files, icon=":material/analytics:"):
    chosen = chosen_car
    with st.spinner("Reading the log and predicting fuel use..."):
        try:
            res = app_core.ingest_logs(u, chosen, [(f.name, f.getvalue()) for f in files],
                                       temp=temp if temp is not None else np.nan)
        except ValueError as e:
            st.error(f"Could not use this log: {e}")
            st.stop()
        except Exception as e:                                   # unreadable CSV etc.
            st.error(f"Could not read the file(s): {e}")
            st.stop()
    st.session_state["car_id"] = chosen["id"]
    st.session_state["last_upload"] = res["trip_ids"]
    st.toast(f"Saved {len(res['trip_ids'])} trip(s) · model {res['model']} · {res['calibration']}")
    session.open_trip(res["trip_ids"][0])
