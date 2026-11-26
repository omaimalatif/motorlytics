"""Quick estimate: fuel and cost for a trip without a log (typical driving for the chosen road type and style)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import cars, quick_estimate, weather
from ui import components as ui
from ui import session

u = session.require_user()
ui.page_header("Quick estimate", "No log needed: fuel and cost for a planned or past trip, from typical driving "
               "on that kind of road. Nothing is saved unless you click Save.")
my_cars = session.user_cars(u)
if not my_cars:
    ui.empty_state("Add your car first: the estimate needs its engine size.", "Add your car", session.PAGES["car"])
    st.stop()
car = session.active_car(u)

with st.form("quick"):
    c1, c2, c3 = st.columns(3)
    ids = [c["id"] for c in my_cars]
    cid = c1.selectbox("Car", ids, index=ids.index(car["id"]),
                       format_func=lambda i: cars.car_title(next(c for c in my_cars if c["id"] == i)))
    dist = c2.number_input("Distance (km)", min_value=0.5, max_value=2000.0, value=None, step=1.0,
                           placeholder="e.g. 25")
    road = c3.selectbox("Road type", quick_estimate.ROAD_TYPES, index=0,
                        help="city: stop-and-go below 30 km/h average; mixed: 30-60 km/h; highway: above 60 km/h")
    c4, c5, c6 = st.columns(3)
    style = c4.selectbox("Driving style", quick_estimate.STYLES, index=1,
                         help="eco: smooth, few hard accelerations; aggressive: frequent hard accelerations and braking")
    w = weather.current(u.get("city")) if u.get("city") else None
    temp = c5.number_input("Air temperature (°C)", value=float(w["temp_c"]) if w else None, step=0.5,
                           help="Filled from Open-Meteo for your city if set.")
    petrol = c6.number_input("Petrol price (PKR per litre)", min_value=0.0, value=u.get("petrol_pkr"), step=1.0,
                             placeholder="today's price", help="From your Account page; it is not built in.")
    claimed = st.number_input("Your own km/L figure (optional)", min_value=1.0, max_value=60.0, value=None, step=0.5,
                              help="If you know what your car does, it is shown next to the estimate.")
    go = st.form_submit_button("Estimate", type="primary", icon=":material/calculate:")

if go:
    if not dist:
        st.error("Enter the distance.")
    else:
        chosen = next(c for c in my_cars if c["id"] == cid)
        try:
            st.session_state["quick_est"] = quick_estimate.estimate(chosen, dist, road, style, temp_c=temp,
                                                                    petrol_pkr=petrol or None, claimed_kmpl=claimed)
        except ValueError as e:
            st.error(str(e))

est = st.session_state.get("quick_est")
if est:
    chosen = next((c for c in my_cars if c["id"] == est["car_id"]), car)
    st.subheader(f"{est['dist_km']:.0f} km, {est['road_type']}, {est['style']} driving · {cars.car_title(chosen)}")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.metric_tile("Fuel economy", f"{est['kmpl']:.1f} km/L", help=f"{est['l100']:.2f} L/100 km")
    with k2:
        ui.metric_tile("Fuel", f"{est['litres']:.2f} L")
    with k3:
        ui.metric_tile("Cost", session.fmt(est["cost_pkr"], "PKR", 0) if est["cost_pkr"] is not None else "set price")
    with k4:
        ui.metric_tile("Time (typical)", f"{est['dur_min']:.0f} min", help=f"At {est['avg_speed']:.0f} km/h average, "
                       "the typical speed for this road type and style.")
    ui.badge({"calibration": "calibrated", "pakwheels": "PakWheels-adjusted", "none": "uncorrected"}[est["factor_source"]],
             {"calibration": "good", "pakwheels": "info", "none": "warn"}[est["factor_source"]])
    st.caption(f"Correction: {est['factor_text']}. Typical driving from {session.plural(est['profile_trips'], 'training trip')} of this "
               "road type and style" + (" (few trips: less reliable)" if est["profile_trips"] < 50 else "") + ".")

    left, right = st.columns(2)
    with left:
        with ui.card("Same trip, other driving styles"):
            d = pd.DataFrame([{"style": s, "litres": v["litres"], "km/L": v["kmpl"]} for s, v in est["by_style"].items()])
            fig = px.bar(d, x="style", y="litres", text=d["km/L"].map(lambda x: f"{x:.1f} km/L"),
                         category_orders={"style": quick_estimate.STYLES})
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="litres")
            st.plotly_chart(fig, width="stretch")
            if est["eco_saving_litres"] > 0.005:
                st.caption(f"Driving eco would save about {est['eco_saving_litres']:.2f} L"
                           + (f" (PKR {est['eco_saving_pkr']:.0f})" if est["eco_saving_pkr"] else "") + " on this trip.")
    with right:
        with ui.card("Your own figure"):
            cl = est.get("claimed")
            if cl:
                st.metric("You said", f"{cl['kmpl']:.1f} km/L", delta=f"estimate {cl['difference_pct']:+.0f}%",
                          delta_color="off")
                st.caption(f"At your figure: {cl['litres']:.2f} L" + (f", PKR {cl['cost_pkr']:.0f}" if cl["cost_pkr"]
                                                                      else "") + ". A fill-up on the Calibration page "
                           "measures your real figure and corrects future estimates.")
            else:
                st.caption("Enter your own km/L above to compare.")

    if st.button("Save as trip", icon=":material/save:"):
        tid = quick_estimate.save_estimate(u, chosen, est)
        st.session_state.pop("quick_est", None)
        st.toast("Saved as a quick-estimate trip")
        session.open_trip(tid)
