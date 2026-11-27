"""Session helpers shared by the pages: logged-in user, active car, trips table, navigation."""
import pandas as pd
import streamlit as st

from core import app_core, cars, db

PAGES = {
    "overview": "app_pages/overview.py", "upload": "app_pages/upload_trip.py", "quick": "app_pages/quick_estimate.py", "detail": "app_pages/trip_detail.py",
    "history": "app_pages/trip_history.py", "analytics": "app_pages/analytics.py", "car": "app_pages/my_car.py",
    "calibration": "app_pages/calibration.py", "model": "app_pages/about_model.py", "account": "app_pages/account.py",
}


# names shown to users for the three model types (same as the Calibration page); never show internal names
MODEL_TYPES = {"behaviour_only": "Speed only", "full_no_load": "Speed + RPM", "full": "Speed + RPM + load"}
_MODEL_NAMES = {"behaviour_only": "Speed-only model", "full_no_load": "Speed + RPM model",
                "full": "Speed + RPM + load model"}


def plural(n, word: str, plural_word: str = None) -> str:
    """'1 trip', '3 trips'."""
    n_int = int(n) if float(n).is_integer() else n
    return f"{n_int} {word if n_int == 1 else (plural_word or word + 's')}"


def model_label(model_key: str) -> str:
    """'behaviour_only/LWR' -> 'Speed-only model (LWR)'."""
    tier, _, algo = (model_key or "").partition("/")
    name = _MODEL_NAMES.get(tier, "Model")
    return f"{name} ({algo})" if algo else name


def friendly(text: str) -> str:
    """Replace internal model-type names in report text with the names users see."""
    for k, v in MODEL_TYPES.items():
        text = text.replace(k, v)
    return text


def trip_calibration_line(trip: dict) -> str:
    """One plain sentence on how the trip's prediction was corrected (no internal names or car keys)."""
    info = trip.get("tips") or {}
    measured = trip.get("measured_l100") is not None
    if info.get("quick_estimate"):
        return ""
    if info.get("demo"):
        return "enviroCar demo trip, not your car: it does not change your car's calibration."
    if trip.get("calibrated"):
        k = info.get("k")
        return ("Calibrated for your car: the model's estimate was multiplied by your car's correction factor"
                + (f" k = {k:.2f}." if k else "."))
    if info.get("reference_adjusted"):
        return ("Not calibrated yet: the estimate is adjusted to the PakWheels average for this model. "
                + ("This trip's MAF data will calibrate your car for the next trip." if measured
                   else "Upload a trip with MAF or log a fill-up to calibrate your car."))
    if measured:
        return "Not calibrated yet: this trip's MAF data will calibrate your car for the next trip."
    return "Not calibrated yet: upload a trip with MAF or log a fill-up to calibrate your car."


def is_phone() -> bool:
    """True on a phone browser (from the User-Agent), to keep heavy widgets such as maps off the first load."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    return any(k in ua for k in ("iPhone", "Android", "Mobile"))


def user() -> dict | None:
    uid = st.session_state.get("user_id")
    return db.get_user(uid) if uid else None


def require_user() -> dict:
    u = user()
    if not u:
        st.stop()
    return u


def user_cars(u: dict) -> list:
    """The user's own cars (demo cars from fetched trips are not offered in car pickers)."""
    return db.get_cars(u["id"], include_demo=False)


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
        return "calibrated", "good", f"Calibrated: k = {c['k']:.2f} from {plural(c['n_trips'], 'trip')}"
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


def display_car(car: dict) -> str:
    """Car name for lists and headers; demo cars say they are not the user's car."""
    if not car:
        return "no car"
    if car.get("is_demo"):
        return f"{car.get('nickname') or 'Demo, not your car'} ({cars.car_label(car)})"
    return cars.car_label(car)


def trips_df(u: dict, car_id=None) -> pd.DataFrame:
    """The user's trips as a DataFrame with handy derived columns."""
    t = pd.DataFrame(db.get_trips(u["id"], car_id))
    if t.empty:
        return t
    # format="mixed": start times come with and without fractions of a second; a single inferred format
    # would turn the others into NaT
    t["started"] = pd.to_datetime(t["started_at"], errors="coerce", format="mixed")
    t["date"] = t["started"].dt.date
    t["hour"] = t["started"].dt.hour
    t["l100"] = t["measured_l100"].fillna(t["pred_l100"])            # measured where available
    t["litres"] = t["l100"] * t["dist_km"] / 100
    t["source"] = t["measured_l100"].notna().map({True: "measured", False: "predicted"})
    car_names = {c["id"]: display_car(c) for c in db.get_cars(u["id"])}
    t["car"] = t["car_id"].map(car_names)
    return t


def fmt_dt(ts, fmt: str = "%d %b %Y, %H:%M", none: str = "date unknown") -> str:
    """Format a timestamp; never fails on a missing date."""
    return ts.strftime(fmt) if ts is not None and pd.notna(ts) else none


def open_trip(trip_id: int) -> None:
    st.session_state["trip_id"] = int(trip_id)
    st.switch_page(PAGES["detail"])


def fmt(v, unit: str = "", digits: int = 1, none: str = "–") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return none
    return f"{v:,.{digits}f}{(' ' + unit) if unit else ''}"
