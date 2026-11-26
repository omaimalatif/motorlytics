"""Trends across trips: fuel over time, by road type and style, eco-score, measured vs predicted, cost."""
import plotly.express as px
import streamlit as st

from ui import components as ui
from ui import session

u = session.require_user()
ui.page_header("Analytics", "Your trips over time and what drives your fuel use.")
car = session.active_car(u)
all_cars = st.toggle("All my cars", value=False)
t = session.trips_df(u, None if all_cars else (car["id"] if car else None))
if t.empty:
    ui.empty_state("No trips yet.", "Upload a trip", session.PAGES["upload"])
    st.stop()

k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.metric_tile("Trips", f"{len(t)}")
with k2:
    ui.metric_tile("Distance", session.fmt(t["dist_km"].sum(), "km", 0))
with k3:
    ui.metric_tile("Fuel", session.fmt(t["litres"].sum(), "L", 1))
with k4:
    ui.metric_tile("Average L/100 km", session.fmt(t["litres"].sum() / t["dist_km"].sum() * 100, digits=1),
                   help="Distance-weighted.")


def tidy(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0))
    return fig


a1, a2 = st.columns(2)
with a1:
    with ui.card("Fuel use over time"):
        st.plotly_chart(tidy(px.scatter(t.sort_values("started"), x="started", y="l100", color="source",
                                        labels={"started": "", "l100": "L/100 km"})), width="stretch")
with a2:
    with ui.card("Fuel use by road type"):
        st.plotly_chart(tidy(px.box(t, x="road_type", y="l100", points="all",
                                    category_orders={"road_type": ["city", "mixed", "highway"]},
                                    labels={"road_type": "", "l100": "L/100 km"})), width="stretch")
b1, b2 = st.columns(2)
with b1:
    with ui.card("Eco-score per trip"):
        st.plotly_chart(tidy(px.histogram(t, x="eco_score", nbins=20, range_x=[0, 100],
                                          labels={"eco_score": "eco-score"})), width="stretch")
with b2:
    with ui.card("Fuel use by driving style"):
        s = t.groupby("style", as_index=False).agg(l100=("l100", "median"), trips=("id", "size"))
        st.plotly_chart(tidy(px.bar(s, x="style", y="l100", text="trips",
                                    category_orders={"style": ["eco", "normal", "aggressive"]},
                                    labels={"style": "", "l100": "median L/100 km"})), width="stretch")
c1, c2 = st.columns(2)
with c1:
    with ui.card("Measured vs model (trips with MAF)"):
        m = t[t["measured_l100"].notna()]
        if len(m):
            fig = px.scatter(m, x="measured_l100", y="pred_l100", color="calibrated",
                             labels={"measured_l100": "measured L/100 km", "pred_l100": "model L/100 km"})
            lim = [0, float(max(m["measured_l100"].max(), m["pred_l100"].max()) * 1.1)]
            fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(dash="dash"))
            st.plotly_chart(tidy(fig), width="stretch")
        else:
            st.caption("No trips with MAF yet.")
with c2:
    with ui.card("Cost by month (PKR)"):
        if t["cost_pkr"].notna().any():
            mth = t.assign(month=t["started"].dt.to_period("M").astype(str)).groupby("month", as_index=False)["cost_pkr"].sum()
            st.plotly_chart(tidy(px.bar(mth, x="month", y="cost_pkr", labels={"month": "", "cost_pkr": "PKR"})),
                            width="stretch")
        else:
            st.caption("Set your petrol price on the Account page to see costs for new trips.")
