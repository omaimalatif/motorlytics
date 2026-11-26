"""Upload one or more trip logs -> predict -> save as trips -> open the trip detail."""
import numpy as np
import streamlit as st

from core import app_core, cars, envirocar_fetch, weather
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



def analyse(file_list):
    """Predict and save the trips in [(name, bytes)], then open the first one."""
    with st.spinner("Reading the log and predicting fuel use..."):
        try:
            res = app_core.ingest_logs(u, chosen_car, file_list, temp=temp if temp is not None else np.nan)
        except ValueError as e:
            st.error(f"Could not use this log: {e}")
            st.stop()
        except Exception as e:                                   # unreadable CSV etc.
            st.error(f"Could not read the file(s): {e}")
            st.stop()
    st.session_state["car_id"] = chosen_car["id"]
    st.session_state["last_upload"] = res["trip_ids"]
    st.toast(f"Saved {session.plural(len(res['trip_ids']), 'trip')} · {session.model_label(res['model'])}")
    session.open_trip(res["trip_ids"][0])


if st.button("Analyse trip", type="primary", disabled=not files, icon=":material/analytics:"):
    analyse([(f.name, f.getvalue()) for f in files])

# ---------------- no log at hand: fetch a real trip from enviroCar ----------------
st.divider()
with ui.card("No log at hand? Try a real trip", "Downloads one track recorded by another driver from enviroCar "
             "(open data) that this app has not used before, with real MAF so the measured fuel is shown too."):
    if st.button("Fetch a real trip", icon=":material/cloud_download:"):
        with st.spinner("Looking for an unseen enviroCar track with a real MAF sensor..."):
            try:
                st.session_state["fetched_log"] = envirocar_fetch.fetch_unseen_track()
            except envirocar_fetch.FetchError as e:
                used = st.session_state.get("fallback_used", [])
                fb = envirocar_fetch.fallback_sample(exclude=used)
                st.session_state["fallback_used"] = used + [fb["filename"]]
                st.session_state["fetched_log"] = fb
                st.warning(f"Could not fetch a new track: {e}. Using a bundled real trip from samples/ instead.",
                           icon=":material/wifi_off:")
    fl = st.session_state.get("fetched_log")
    if fl:
        st.markdown(f"**{fl['car']}**" + (f" ({fl['disp_L']:.1f} L)" if fl.get("disp_L") else "")
                    + f" · {fl['dist_km']:.1f} km · {fl['source']}")
        if fl.get("track_id"):
            st.caption(f"enviroCar track {fl['track_id']}" + (f", recorded {fl['begin'][:10]}" if fl.get("begin") else ""))
        st.caption("It is analysed as a separate demo car with this vehicle's own engine size, labelled "
                   "\"enviroCar demo, not your car\". Your cars and their calibration are not changed.")
        a1, a2 = st.columns([1, 1])
        if a1.button("Analyse this trip", type="primary", icon=":material/analytics:", key="analyse_fetched"):
            with st.spinner("Predicting fuel use for the demo trip..."):
                try:
                    res = app_core.ingest_demo(u, fl, temp=temp if temp is not None else np.nan)
                except Exception as e:                           # unreadable log etc.
                    st.error(f"Could not use this trip: {e}")
                    st.stop()
            st.session_state["last_upload"] = res["trip_ids"]
            st.session_state.pop("fetched_log", None)
            st.toast(f"Demo trip saved under \"{res['demo_car']['nickname']}\" (not your car)")
            session.open_trip(res["trip_ids"][0])
        if a2.button("Discard", key="discard_fetched"):
            st.session_state.pop("fetched_log", None)
            st.rerun()
