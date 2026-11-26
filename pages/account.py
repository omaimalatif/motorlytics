"""Account: name, city (for weather), petrol price in PKR per litre (no default), logout."""
import streamlit as st

from core import db, weather
from ui import components as ui
from ui import session

u = session.require_user()
ui.page_header("Account", u["email"])

with ui.card("Profile"):
    with st.form("profile"):
        name = st.text_input("Name", value=u["name"])
        city = st.text_input("City", value=u.get("city") or "", help="Used for the current temperature (Open-Meteo).")
        petrol = st.number_input("Petrol price (PKR per litre)", min_value=0.0, value=u.get("petrol_pkr"), step=1.0,
                                 placeholder="today's price", help="Enter today's price; it is not built in.")
        if st.form_submit_button("Save", type="primary"):
            db.update_user(u["id"], name=name.strip() or u["name"], city=city.strip() or None,
                           petrol_pkr=float(petrol) if petrol else None)
            st.success("Saved. New trips use this price.")
            st.rerun()

if u.get("city"):
    w = weather.current(u["city"])
    if w:
        st.caption(f"Now in {w['city']}, {w['country']}: {w['temp_c']:.1f} °C (Open-Meteo)")
    else:
        st.caption(f"Could not find the weather for '{u['city']}'.")

st.divider()
if st.button("Log out", icon=":material/logout:"):
    st.session_state.clear()
    st.rerun()
