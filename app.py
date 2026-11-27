"""Motorlytics web app (Streamlit).

    streamlit run app.py

Look and feel: ui/theme.py (colours, CSS, chart template) and .streamlit/config.toml. Logic lives in core/ (no Streamlit code there) and
motorlytics_features.py; reusable widgets in ui/components.py.
Env vars for testing: MOTORLYTICS_DB (SQLite file), MOTORLYTICS_CALIB (calibration store), MOTORLYTICS_NO_DEMO=1
(no demo account on the first start; core/demo.py creates it when the database has no users).
"""
import os

import streamlit as st

from core import db          # light: sqlite only. pandas, Plotly and the models load only after sign-in
from ui import theme

st.set_page_config(page_title="Motorlytics", page_icon=theme.ICON, layout="wide")


@st.cache_resource(show_spinner="Setting up the demo account (first start only)...")
def first_start() -> bool:
    """Once per server process: database + calibration store, and the demo account if there are no users (online the
    disk starts empty and resets on restart). MOTORLYTICS_NO_DEMO=1 turns the demo off (tests)."""
    db.init_db()
    if os.environ.get("MOTORLYTICS_NO_DEMO") == "1":
        return False
    calib = os.environ.get("MOTORLYTICS_CALIB") or db.ROOT / "models" / "calibration.json"
    if db.user_count() and os.path.exists(calib):         # normal start: nothing to set up, nothing heavy to import
        return False
    from core import demo
    return demo.ensure_first_start()


first_start()
db.init_db()
theme.setup()


uid = st.session_state.get("user_id")
user = db.get_user(uid) if uid else None
if not user:
    # signed out: only the public pages, in a top bar (no app pages, no sidebar)
    st.logo(image=theme.LOGO, size="large")          # dark-text wordmark on the light top bar
    public = st.navigation([st.Page("app_pages/welcome.py", title="Welcome", icon=":material/home:", default=True),
                            st.Page("app_pages/sign_in.py", title="Sign in", icon=":material/login:")], position="top")
    public.run()
    st.stop()

from core import cars  # noqa: E402  (signed in: the app pages need these)
from ui import components as ui  # noqa: E402
from ui import session  # noqa: E402
# no logo at the top of the sidebar after sign-in (the navigation starts right away)

P = session.PAGES
pages = {
    "Motorlytics": [
        st.Page(P["overview"], title="Overview", icon=":material/dashboard:", default=True),
        st.Page(P["upload"], title="Upload trip", icon=":material/upload_file:"),
        st.Page(P["quick"], title="Quick estimate", icon=":material/calculate:"),
        st.Page(P["detail"], title="Trip detail", icon=":material/route:", visibility="hidden"),  # opened from trips
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
        # public pages stay reachable (hidden) so logout can switch to Welcome and Sign in can hand over
        st.Page("app_pages/welcome.py", title="Welcome", icon=":material/home:", visibility="hidden"),
        st.Page("app_pages/sign_in.py", title="Sign in", icon=":material/login:", visibility="hidden"),
    ],
}
nav = st.navigation(pages)
if st.session_state.pop("goto", None):                     # just signed in: open Overview
    st.switch_page(P["overview"])

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
        label, kind, text = session.calibration_state(car)
        ui.badge(label, kind)
        st.caption(text)

nav.run()
