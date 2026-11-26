"""Motorlytics web app (Streamlit).

    streamlit run app.py

Phase 1: structure and plain Streamlit widgets. Logic lives in core/ (no Streamlit code there) and
motorlytics_features.py; reusable widgets in ui/components.py.
Env vars for testing: MOTORLYTICS_DB (SQLite file), MOTORLYTICS_CALIB (calibration store).
"""
import streamlit as st

from core import auth, cars, db
from ui import session

st.set_page_config(page_title="Motorlytics", page_icon=":material/local_gas_station:", layout="wide")
db.init_db()


def login_page():
    st.title("Motorlytics")
    st.caption("Fuel use, cost and eco-driving tips from your car's OBD-II trip logs.")
    left, _ = st.columns([1, 1])
    with left:
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("login"):
                email = st.text_input("Email")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary"):
                    u = auth.login(email, pw)
                    if u:
                        st.session_state["user_id"] = u["id"]
                        st.rerun()
                    st.error("Email or password is not correct.")
        with tab_up:
            with st.form("register"):
                name = st.text_input("Name")
                email = st.text_input("Email", key="reg_email")
                city = st.text_input("City", help="Used for the current temperature (Open-Meteo).")
                pw = st.text_input("Password (8+ characters)", type="password", key="reg_pw")
                pw2 = st.text_input("Repeat password", type="password")
                if st.form_submit_button("Create account", type="primary"):
                    if pw != pw2:
                        st.error("The passwords do not match.")
                    else:
                        u, err = auth.register(name, email, pw, city)
                        if err:
                            st.error(err)
                        else:
                            st.session_state["user_id"] = u["id"]
                            st.rerun()


user = session.user()
if not user:
    st.navigation([st.Page(login_page, title="Sign in", icon=":material/login:")], position="hidden").run()
    st.stop()

P = session.PAGES
pages = {
    "Motorlytics": [
        st.Page(P["overview"], title="Overview", icon=":material/dashboard:", default=True),
        st.Page(P["upload"], title="Upload trip", icon=":material/upload_file:"),
        st.Page(P["detail"], title="Trip detail", icon=":material/route:"),
        st.Page(P["history"], title="Trip history", icon=":material/history:"),
        st.Page(P["analytics"], title="Analytics", icon=":material/insights:"),
    ],
    "My car": [
        st.Page(P["car"], title="My car", icon=":material/directions_car:"),
        st.Page(P["calibration"], title="Calibration", icon=":material/tune:"),
    ],
    "More": [
        st.Page(P["model"], title="About the model", icon=":material/model_training:"),
        st.Page(P["account"], title="Account", icon=":material/person:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown(f"**{user['name']}**")
    my_cars = session.user_cars(user)
    car = session.active_car(user)
    if not my_cars:
        st.caption("No car yet")
        if st.button("Add your car", icon=":material/add:"):
            st.switch_page(P["car"])
    else:
        if len(my_cars) > 1:
            ids = [c["id"] for c in my_cars]
            pick = st.selectbox("Car", ids, index=ids.index(car["id"]),
                                format_func=lambda i: cars.car_title(next(c for c in my_cars if c["id"] == i)))
            if pick != car["id"]:
                st.session_state["car_id"] = pick
                st.rerun()
        else:
            st.markdown(f"**{cars.car_title(car)}**")
        text, ok = session.calibration_text(car)
        st.badge("calibrated" if ok else "uncalibrated", color="green" if ok else "orange")
        st.caption(text)

nav.run()
