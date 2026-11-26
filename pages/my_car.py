"""Cars: list, choose the active car, add a car (catalogue lookup or manual entry), delete."""
import streamlit as st

from core import cars, db
from ui import components as ui
from ui import session

u = session.require_user()
ui.page_header("My car", "Engine size and cylinders feed the prediction; everything else comes from your trip logs.")

my_cars = session.user_cars(u)
active = session.active_car(u)
if my_cars:
    cols = st.columns(min(3, len(my_cars)))
    for i, c in enumerate(my_cars):
        with cols[i % len(cols)]:
            with ui.card(cars.car_label(c)):
                st.caption(f"{c['cylinders']:.0f} cylinders" if c["cylinders"] else "cylinders unknown")
                text, ok = session.calibration_text(c)
                ui.badge("calibrated" if ok else "uncalibrated", "good" if ok else "warn")
                st.caption(text)
                b1, b2 = st.columns(2)
                if c["id"] == active["id"]:
                    b1.button("Active", disabled=True, key=f"act_{c['id']}", width="stretch")
                elif b1.button("Use this car", key=f"use_{c['id']}", width="stretch"):
                    st.session_state["car_id"] = c["id"]
                    st.rerun()
                with b2.popover("Delete", width="stretch"):
                    st.caption("Its trips stay in your history without a car.")
                    if st.button("Delete car", key=f"del_{c['id']}", type="primary"):
                        db.delete_car(c["id"], u["id"])
                        st.session_state.pop("car_id", None)
                        st.rerun()

st.subheader("Add a car")
manual = st.toggle("Enter engine details manually", value=not cars.makes(),
                   help="Use this if your car is not in the catalogue (Pakistani used-car listings).")
with ui.card():
    if not manual:
        c1, c2, c3 = st.columns(3)
        make = c1.selectbox("Make", cars.makes(), index=None, placeholder="e.g. Toyota")
        model = c2.selectbox("Model", cars.models(make) if make else [], index=None, placeholder="e.g. Corolla")
        year = c3.selectbox("Year", cars.years(make, model) if make and model else [], index=None)
        info = cars.lookup(make, model, year) if make and model else {}
        if info:
            opts = [d for d, _ in info["options"]]
            e1, e2 = st.columns(2)
            disp = e1.selectbox("Engine size (L)", opts, format_func=lambda d: f"{d:.1f} L "
                                f"({dict(info['options'])[d]} listings)")
            cyl = e2.number_input("Cylinders (estimated)", value=float(cars.estimate_cylinders(disp)), step=1.0,
                                  min_value=2.0, max_value=12.0)
        else:
            disp, cyl = None, None
    else:
        c1, c2, c3 = st.columns(3)
        make = c1.text_input("Make")
        model = c2.text_input("Model")
        year = c3.number_input("Year", min_value=1980, max_value=2030, value=None, step=1)
        e1, e2 = st.columns(2)
        disp = e1.number_input("Engine size (L)", min_value=0.5, max_value=8.0, value=None, step=0.1)
        cyl = e2.number_input("Cylinders", min_value=2.0, max_value=12.0,
                              value=float(cars.estimate_cylinders(disp)) if disp else None, step=1.0,
                              help="Estimated from engine size if you leave it.")
    if st.button("Add car", type="primary", disabled=not (make and model and disp), icon=":material/add:"):
        new = db.add_car(u["id"], make.strip(), model.strip(), int(year) if year else None, float(disp),
                         float(cyl) if cyl else cars.estimate_cylinders(disp))
        st.session_state["car_id"] = new["id"]
        st.success(f"Added {cars.car_label(new)}")
        st.rerun()
