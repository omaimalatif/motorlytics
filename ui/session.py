"""Session helpers shared by the pages: logged-in user, active car, trips table, navigation."""
import pandas as pd
import streamlit as st

from core import app_core, cars, db

PAGES = {
    "overview": "pages/overview.py", "upload": "pages/upload_trip.py", "detail": "pages/trip_detail.py",
    "history": "pages/trip_history.py", "analytics": "pages/analytics.py", "car": "pages/my_car.py",
    "calibration": "pages/calibration.py", "model": "pages/about_model.py", "account": "pages/account.py",
}


def user() -> dict | None:
    uid = st.session_state.get("user_id")
    return db.get_user(uid) if uid else None


def require_user() -> dict:
    u = user()
    if not u:
        st.stop()
    return u


def user_cars(u: dict) -> list:
    return db.get_cars(u["id"])


def active_car(u: dict) -> dict | None:
    cs = user_cars(u)
    if not cs:
        return None
    cid = st.session_state.get("car_id")
    car = next((c for c in cs if c["id"] == cid), None) or cs[0]
    st.session_state["car_id"] = car["id"]
    return car


def calibration_state(car: dict) -> tuple:
    """(badge label, badge kind, text): own calibration > PakWheels starting adjustment > uncalibrated."""
    from core import pk_reference
    if not car:
        return "no car", "neutral", "no car"
    status = app_core.calibration_status(car["car_key"])
    if status:
        tier = "behaviour_only" if "behaviour_only" in status else next(iter(status))
        c = status[tier]
        return "calibrated", "good", f"Calibrated: k = {c['k']:.2f} from {c['n_trips']} trips"
    ref = pk_reference.match(car)
    if ref:
        return ("PakWheels-adjusted", "info",
                f"Adjusted to the PakWheels average for {ref['make']} {ref['model']} ({ref['city_kmpl']:.0f} km/L "
                "city, {:.0f} highway) until you calibrate with a MAF trip or a fill-up".format(ref["highway_kmpl"]))
    return "uncalibrated", "warn", "Uncalibrated: upload a trip with MAF or log a fill-up"


def calibration_text(car: dict) -> tuple:
    """(short text, is_calibrated) - kept for older callers; see calibration_state."""
    label, _, text = calibration_state(car)
    return text, label == "calibrated"


def trips_df(u: dict, car_id=None) -> pd.DataFrame:
    """The user's trips as a DataFrame with handy derived columns."""
    t = pd.DataFrame(db.get_trips(u["id"], car_id))
    if t.empty:
        return t
    t["started"] = pd.to_datetime(t["started_at"], errors="coerce")
    t["date"] = t["started"].dt.date
    t["hour"] = t["started"].dt.hour
    t["l100"] = t["measured_l100"].fillna(t["pred_l100"])            # measured where available
    t["litres"] = t["l100"] * t["dist_km"] / 100
    t["source"] = t["measured_l100"].notna().map({True: "measured", False: "predicted"})
    car_names = {c["id"]: cars.car_label(c) for c in db.get_cars(u["id"])}
    t["car"] = t["car_id"].map(car_names)
    return t


def open_trip(trip_id: int) -> None:
    st.session_state["trip_id"] = int(trip_id)
    st.switch_page(PAGES["detail"])


def fmt(v, unit: str = "", digits: int = 1, none: str = "–") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return none
    return f"{v:,.{digits}f}{(' ' + unit) if unit else ''}"
