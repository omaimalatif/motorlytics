"""Account: profile, city (for weather), petrol price in PKR per litre (no default), sign out."""
import streamlit as st

from core import db, weather
from ui import components as ui
from ui import session
from ui import theme

u = session.require_user()
ui.page_header("Account", "Your profile and the settings used for costs and weather.")
st.info("Online demo: data may reset when the app restarts.", icon=":material/cloud:")

initials = "".join(w[0] for w in u["name"].split()[:2]).upper() or "?"
left, right = st.columns([1, 2], gap="medium")
with left:
    with ui.card(kind="hero"):
        st.html(f'<div style="display:flex;gap:14px;align-items:center">'
                f'<div style="width:56px;height:56px;border-radius:999px;background:{theme.GRADIENT};display:flex;'
                f'align-items:center;justify-content:center;font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:800;'
                f'font-size:1.25rem;color:#fff">{theme.esc(initials)}</div>'
                f'<div><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:700;font-size:1.1rem">'
                f'{theme.esc(u["name"])}</div><div style="color:{theme.MUTED};font-size:.9rem">{theme.esc(u["email"])}'
                f'</div></div></div>')
        n_cars, n_trips = len(session.user_cars(u)), len(db.get_trips(u["id"]))
        st.caption(f"{session.plural(n_cars, 'car')} · {session.plural(n_trips, 'trip')}")
        if u.get("city"):
            w = weather.current(u["city"])
            if w:
                st.caption(f":material/thermostat: Now in {w['city']}, {w['country']}: {w['temp_c']:.1f} °C (Open-Meteo)")
            else:
                st.caption(f"Could not find the weather for '{u['city']}'.")
        if st.button("Sign out", icon=":material/logout:", width="stretch", key="mldanger-signout"):
            st.session_state.clear()
            st.switch_page("app_pages/welcome.py")

with right:
    with ui.card("Settings", "Your petrol price turns litres into rupees; your city gives the current temperature."):
        with st.form("profile", border=False):
            name = st.text_input("Name", value=u["name"])
            c1, c2 = st.columns(2)
            city = c1.text_input("City", value=u.get("city") or "", help="Used for the current temperature (Open-Meteo).")
            petrol = c2.number_input("Petrol price (PKR per litre)", min_value=0.0, value=u.get("petrol_pkr"), step=1.0,
                                     placeholder="today's price", help="Enter today's price; it is not built in.")
            if st.form_submit_button("Save", type="primary"):
                db.update_user(u["id"], name=name.strip() or u["name"], city=city.strip() or None,
                               petrol_pkr=float(petrol) if petrol else None)
                st.success("Saved. New trips use this price.", icon=":material/check_circle:")
                st.rerun()
