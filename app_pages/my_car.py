"""My car: profile card (no pictures), statistics, best / worst trip, edit, switch, add (catalogue suggestions or manual), delete."""
import pandas as pd
import streamlit as st

from core import cars, db, pk_reference
from ui import components as ui
from ui import session

u = session.require_user()
ui.page_header("My car", "Engine size and cylinders feed the prediction; the other details describe your car.")


def opt_index(options, value):
    return options.index(value) if value in options else None


def detail_fields(prefix: str, d: dict) -> dict:
    """Editable car details (shared by the add and edit forms). d: current or suggested values."""
    c1, c2, c3 = st.columns(3)
    nickname = c1.text_input("Nickname (optional)", value=d.get("nickname") or "", key=f"{prefix}_nick")
    colour = c2.text_input("Colour (optional)", value=d.get("colour") or "", key=f"{prefix}_colour")
    reg = c3.text_input("Registration (optional)", value=d.get("registration") or "", key=f"{prefix}_reg",
                        help="Shown masked, e.g. LEA-••-123.")
    c1, c2, c3, c4 = st.columns(4)
    trans = c1.selectbox("Transmission", cars.TRANSMISSIONS, index=opt_index(cars.TRANSMISSIONS, d.get("transmission")),
                         key=f"{prefix}_trans", placeholder="choose")
    fuel = c2.selectbox("Fuel type", cars.FUEL_TYPES, index=opt_index(cars.FUEL_TYPES, d.get("fuel_type") or "petrol"),
                        key=f"{prefix}_fuel", help="The model was trained on petrol cars.")
    bodies = cars.BODY_TYPES + ([d["body_type"]] if d.get("body_type") and d["body_type"] not in cars.BODY_TYPES else [])
    body = c3.selectbox("Body type", bodies, index=opt_index(bodies, d.get("body_type")), key=f"{prefix}_body",
                        placeholder="choose")
    asm = c4.selectbox("Assembled", cars.ASSEMBLY, index=opt_index(cars.ASSEMBLY, d.get("assembled")),
                       key=f"{prefix}_asm", placeholder="choose")
    c1, c2 = st.columns(2)
    tank = c1.number_input("Fuel tank (litres, optional)", min_value=5.0, max_value=200.0, step=1.0,
                           value=float(d["tank_litres"]) if d.get("tank_litres") else None, key=f"{prefix}_tank",
                           help="Used for the range on a full tank.")
    odo = c2.number_input("Odometer (km, optional)", min_value=0.0, step=100.0,
                          value=float(d["odometer_km"]) if d.get("odometer_km") is not None else None,
                          key=f"{prefix}_odo")
    return {"nickname": nickname.strip() or None, "colour": colour.strip() or None,
            "registration": reg.strip().upper() or None, "transmission": trans, "fuel_type": fuel,
            "body_type": body, "assembled": asm, "tank_litres": tank, "odometer_km": odo}


my_cars = session.user_cars(u)
car = session.active_car(u)

if my_cars:
    # ---------------- switch car ----------------
    if len(my_cars) > 1:
        ids = [c["id"] for c in my_cars]
        pick = st.selectbox("Car", ids, index=ids.index(car["id"]),
                            format_func=lambda i: cars.car_title(next(c for c in my_cars if c["id"] == i)))
        if pick != car["id"]:
            st.session_state["car_id"] = pick
            st.rerun()

    prof = cars.car_profile(car["id"])

    # ---------------- profile card ----------------
    with ui.card(kind="hero"):
        st.subheader(cars.car_title(car))
        if car.get("nickname"):
            st.caption(" ".join(str(car[k]) for k in ("make", "model", "year") if car.get(k)))
        details = [("Engine", cars.fmt_disp(car.get("disp_L")) or "–"),
                   ("Cylinders", str(car["cylinders"]) if car.get("cylinders") else "–"),
                   ("Transmission", car.get("transmission") or "–"), ("Fuel type", car.get("fuel_type") or "–"),
                   ("Body type", car.get("body_type") or "–"), ("Assembled", car.get("assembled") or "–"),
                   ("Colour", car.get("colour") or "–"), ("Fuel tank", session.fmt(car.get("tank_litres"), "L", 0)),
                   ("Odometer", session.fmt(car.get("odometer_km"), "km", 0)),
                   ("Registration", cars.mask_registration(car.get("registration")) or "–")]
        cols = st.columns(5, gap="small")
        for i, (label, value) in enumerate(details):
            with cols[i % 5]:
                st.caption(label)
                st.markdown(f"**{value}**")
        label, kind, text = session.calibration_state(car)
        ui.badge(label, kind)
        st.caption(text)
        if (car.get("fuel_type") or "petrol") != "petrol":
            st.warning(f"{car['fuel_type']} car: the model was trained on petrol cars, so its estimates are less "
                       "reliable for this car.")

    # ---------------- statistics ----------------
    st.markdown("**Statistics**")
    s = st.columns(3) + st.columns(3)
    with s[0]:
        ui.metric_tile("Trips", f"{prof['trips']}")
    with s[1]:
        ui.metric_tile("Distance (km)", session.fmt(prof["km"], digits=0))
    with s[2]:
        ui.metric_tile("Fuel (L)", session.fmt(prof["litres"], digits=1))
    with s[3]:
        ui.metric_tile("Spent (PKR)", session.fmt(prof["pkr"], digits=0), help="Trips with a petrol price set.")
    with s[4]:
        ui.metric_tile("Average (L/100 km)", session.fmt(prof["avg_l100"], digits=1), accent=True,
                       help=f"Distance-weighted, measured where available. Mean measured "
                            f"{session.fmt(prof['avg_measured_l100'], digits=1)}, mean predicted "
                            f"{session.fmt(prof['avg_predicted_l100'], digits=1)} L/100 km.")
    with s[5]:
        ui.metric_tile("Full-tank range (km)", session.fmt(prof["range_km"], digits=0),
                       help="Tank size ÷ your average L/100 km × 100." if car.get("tank_litres")
                       else "Add the tank size below to see the range.")

    # ---------------- Pakistani reference (PakWheels) ----------------
    ref = pk_reference.match(car)
    with ui.card(":material/location_on: Pakistani reference (PakWheels)"):
        if ref:
            r1, r2, r3 = st.columns(3)
            r1.metric("PakWheels city (km/L)", f"{ref['city_kmpl']:.0f}", help=f"{100 / ref['city_kmpl']:.1f} L/100 km")
            r2.metric("PakWheels highway (km/L)", f"{ref['highway_kmpl']:.0f}",
                      help=f"{100 / ref['highway_kmpl']:.1f} L/100 km")
            mine = 100 / prof["avg_l100"] if prof.get("avg_l100") else None
            r3.metric("Your average (km/L)", f"{mine:.1f}" if mine else "–", help="Distance-weighted over your trips.")
            st.caption(f"{ref['make']} {ref['model']} around {ref['engine_cc']} cc, median of: "
                       f"{', '.join(ref['variants'][:6])}{' ...' if len(ref['variants']) > 6 else ''}. "
                       f"[PakWheels fuel average]({ref['source_url']}), retrieved {ref['retrieved']}. PakWheels does "
                       "not say whether these are owner reports or manufacturer figures.")
            if label != "calibrated":
                st.caption("Until your car is calibrated, speed-only predictions are scaled so that typical city "
                           "and highway driving matches these figures.")
        else:
            st.caption(f"{car.get('make')} {car.get('model')} is not in the PakWheels reference table yet, so "
                       "predictions are not adjusted. Calibrate with a MAF trip or a fill-up for your own car.")

    def trip_summary(tr, key):
        when = pd.to_datetime(tr["started_at"], errors="coerce", format="mixed")
        st.markdown(f"**{tr['l100']:.1f} L/100 km** · {tr['dist_km']:.1f} km · eco-score {tr['eco_score']:.0f}")
        st.caption(session.fmt_dt(when))
        if st.button("Open trip", key=f"open_{key}"):
            session.open_trip(tr["id"])

    n_long = prof.get("long_trips", 0)
    if n_long >= 2:                                        # best and least efficient are different trips
        b1, b2 = st.columns(2, gap="medium")
        for col, key, title in [(b1, "best_trip", "Best trip"), (b2, "worst_trip", "Least efficient trip")]:
            with col:
                with ui.card(title, "Trips of at least 3 km", kind="click"):
                    trip_summary(prof[key], key)
    elif n_long == 1:                                      # one trip: do not show it as both best and worst
        with ui.card("Your trip", "Your only trip of at least 3 km so far"):
            trip_summary(prof["best_trip"], "only_trip")
            st.caption("Upload more trips to compare your best and least efficient.")
    else:
        st.caption("No trips of 3 km or more yet. Upload more trips to compare your best and least efficient.")

    # ---------------- edit ----------------
    with st.expander("Edit car details", icon=":material/edit:"):
        with st.form(f"edit_{car['id']}"):
            c1, c2, c3, c4, c5 = st.columns(5)
            make = c1.text_input("Make", value=car.get("make") or "")
            model = c2.text_input("Model", value=car.get("model") or "")
            year = c3.number_input("Year", min_value=1980, max_value=2030, step=1,
                                   value=int(car["year"]) if car.get("year") else None)
            disp = c4.number_input("Engine (L)", min_value=0.5, max_value=8.0, step=0.1, format="%.2f",
                                   value=float(car["disp_L"]) if car.get("disp_L") else None)
            cyl = c5.number_input("Cylinders", min_value=2, max_value=12, step=1,
                                  value=int(car["cylinders"]) if car.get("cylinders") else None)
            extra = detail_fields(f"edit{car['id']}", car)
            if st.form_submit_button("Save changes", type="primary"):
                if not (make.strip() and model.strip() and disp):
                    st.error("Make, model and engine size are needed.")
                else:
                    db.update_car(car["id"], u["id"], make=make.strip(), model=model.strip(),
                                  year=int(year) if year else None, disp_L=float(disp),
                                  cylinders=int(cyl) if cyl else cars.estimate_cylinders(disp), **extra)
                    st.success("Saved.")
                    st.rerun()

    # ---------------- delete ----------------
    with st.expander("Delete this car", icon=":material/delete:"):
        st.caption("The car is removed; its trips stay in your history without a car. This cannot be undone.")
        sure = st.checkbox(f"Yes, delete {cars.car_title(car)}", key=f"sure_{car['id']}")
        if st.button("Delete car", disabled=not sure, key=f"mldanger-del-{car['id']}", icon=":material/delete:"):
            db.delete_car(car["id"], u["id"])
            st.session_state.pop("car_id", None)
            st.rerun()

# ---------------- add a car ----------------
st.subheader("Add a car" if my_cars else "Add your car")
st.caption("Pick it from the Pakistani used-car catalogue, or enter the details yourself.")
manual = st.toggle("Enter details manually", value=not cars.makes(),
                   help="Use this if your car is not in the catalogue (Pakistani used-car listings).")
with ui.card():
    if not manual:
        c1, c2, c3 = st.columns(3)
        make = c1.selectbox("Make", cars.makes(), index=None, placeholder="e.g. Suzuki")
        model = c2.selectbox("Model", cars.models(make) if make else [], index=None, placeholder="e.g. Cultus")
        year = c3.selectbox("Year", cars.years(make, model) if make and model else [], index=None)
        info = cars.lookup(make, model, year) if make and model else {}
        disp = cyl = None
        if info:
            st.caption(f"Suggested from {info['listings']} listings of this car; change anything that differs.")
            opts = [o for o, _ in info["options"]]
            e1, e2 = st.columns(2)
            sfx = f"{make}_{model}_{year}"                      # new car -> widgets reset to the new suggestion
            disp = e1.selectbox("Engine size", opts, key=f"add_disp_{sfx}",
                                format_func=lambda o: f"{cars.fmt_disp(o)} ({dict(info['options'])[o]} listings)")
            cyl = e2.number_input("Cylinders (estimated)", value=cars.estimate_cylinders(disp), step=1,
                                  min_value=2, max_value=12, key=f"add_cyl_{sfx}")
            extra = detail_fields(f"add_{sfx}", info)
        else:
            extra = {}
    else:
        c1, c2, c3 = st.columns(3)
        make = c1.text_input("Make")
        model = c2.text_input("Model")
        year = c3.number_input("Year", min_value=1980, max_value=2030, value=None, step=1)
        e1, e2 = st.columns(2)
        disp = e1.number_input("Engine size (L)", min_value=0.5, max_value=8.0, value=None, step=0.1, format="%.2f")
        cyl = e2.number_input("Cylinders", min_value=2, max_value=12,
                              value=cars.estimate_cylinders(disp) if disp else None, step=1,
                              help="Estimated from engine size if you leave it.")
        extra = detail_fields("add_manual", {})
    if st.button("Add car", type="primary", disabled=not (make and model and disp), icon=":material/add:"):
        new = db.add_car(u["id"], make.strip(), model.strip(), int(year) if year else None, float(disp),
                         int(cyl) if cyl else cars.estimate_cylinders(disp), **extra)
        st.session_state["car_id"] = new["id"]
        st.success(f"Added {cars.car_title(new)}")
        st.rerun()
