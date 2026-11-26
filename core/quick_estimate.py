"""Quick estimate without a trip log (no Streamlit code).

The user gives car, distance, road type (city / mixed / highway), driving style (eco / normal / aggressive),
temperature, petrol price and optionally the km/L they believe the car does. The behaviour features of the
trip (stops, idling, accelerations, speed profile ...) are the MEDIANS of training trips with that road type and
style (assets/quick_estimate_profiles.csv, built by build_profiles() from the VED trips and k-Means styles).
Distance, duration, temperature, date and the car's engine are set from the inputs and the speed-only model
predicts L/100 km. Then:
  - car with its own calibration: prediction x k (as for logged trips);
  - car in the PakWheels table: PakWheels sets the LEVEL and the model the DIFFERENCES. A normal-style trip under
    standard conditions (city 10 km / mixed 15 km / highway 40 km, 25 C, weekday morning) gets the PakWheels
    figure for that road type (mixed: city and highway blended by its average speed, 30-60 km/h), and the model
    scales it by style, distance and temperature: estimate = PakWheels L/100 km x model(trip) / model(standard).
    (A single factor fitted on the standard drive cycles does not transfer to these typical-driving profiles.)
  - otherwise: the model's prediction, uncorrected.
Nothing is stored until save_estimate().
"""
from functools import lru_cache

import numpy as np
import pandas as pd

import motorlytics_features as mf
from core import app_core, db, pk_reference, scoring

PROFILES = db.ROOT / "assets" / "quick_estimate_profiles.csv"
ROAD_TYPES = ["city", "mixed", "highway"]
STANDARD = {"city": 10.0, "mixed": 15.0, "highway": 40.0}     # km; with 25 C on a weekday morning
STANDARD_WHEN = pd.Timestamp("2026-03-10 10:00")
STYLES = ["eco", "normal", "aggressive"]


def build_profiles(path=PROFILES) -> pd.DataFrame:
    """Median behaviour features per (road type, style) from the training trips (run after 04_driving_styles.py)."""
    trips = pd.read_parquet(mf.PROC / "trips.parquet")
    styles = pd.read_parquet(mf.PROC / "trip_styles.parquet")[["trip_key", "style", "road_type"]]
    t = trips.merge(styles, on="trip_key")
    cols = [c for c in mf.TRIP_BEHAVIOUR + ["rpm_p90", "mean_rpm", "frac_rpm_gt2500", "mean_rpm_per_kmh",
                                            "mean_load", "l_per_100km"] if c in t]
    prof = t.groupby(["road_type", "style"])[cols].median()
    prof["n_trips"] = t.groupby(["road_type", "style"]).size()
    prof = prof.reset_index()
    prof.to_csv(path, index=False)
    return prof


@lru_cache(maxsize=1)
def profiles() -> pd.DataFrame:
    return pd.read_csv(PROFILES)


def _trip_row(car: dict, dist_km: float, road_type: str, style: str, temp_c, when: pd.Timestamp) -> pd.DataFrame:
    p = profiles()
    row = p[(p["road_type"] == road_type) & (p["style"] == style)].iloc[0].to_dict()
    speed = float(row["avg_speed"])
    row.update(dist_km=float(dist_km), dur_min=float(dist_km) / speed * 60,
               temp_c=np.nan if temp_c is None else float(temp_c), oat_mean=np.nan,
               hour=when.hour, weekday=when.dayofweek, month=when.month, is_weekend=float(when.dayofweek >= 5))
    row["short_trip"] = float(row["dur_min"] < 10)
    row.update(app_core.car_spec(car))
    return pd.DataFrame([row])


def estimate(car: dict, dist_km: float, road_type: str, style: str, temp_c=None, petrol_pkr=None,
             claimed_kmpl=None, when=None) -> dict:
    """Estimate fuel for a trip that was not logged. Returns a dict (nothing is saved)."""
    if road_type not in ROAD_TYPES or style not in STYLES:
        raise ValueError("road type must be city / mixed / highway and style eco / normal / aggressive")
    if not dist_km or dist_km <= 0:
        raise ValueError("distance must be more than 0 km")
    when = pd.Timestamp(when) if when is not None else pd.Timestamp.now()
    meta = app_core._meta()
    model_key = meta["best_behaviour_only"]
    bundle = app_core._bundle(model_key)
    status = app_core.calibration_status(car["car_key"]) if car.get("car_key") else {}
    own = status.get("behaviour_only")
    prior = None if own else pk_reference.prior(car)

    def raw_pred(row):
        return float(bundle["model"].predict(mf.to_matrix(row, bundle["features"]))[0])

    ref_l100 = anchor = None
    if prior:
        std = _trip_row(car, STANDARD[road_type], road_type, "normal", 25.0, STANDARD_WHEN)
        anchor = raw_pred(std)
        w = float(np.clip((float(std["avg_speed"].iloc[0]) - 30) / 30, 0, 1)) if road_type == "mixed" else             (1.0 if road_type == "highway" else 0.0)
        ref_l100 = (1 - w) * 100 / prior["city_kmpl"] + w * 100 / prior["highway_kmpl"]

    def predict(st):
        row = _trip_row(car, dist_km, road_type, st, temp_c, when)
        raw = raw_pred(row)
        if own:
            k, pred = own["k"], raw * own["k"]
        elif prior:
            pred = ref_l100 * raw / anchor
            k = pred / raw
        else:
            k, pred = 1.0, raw
        return row, raw, pred, k

    rows = {st: predict(st) for st in STYLES}
    row, raw, pred, k = rows[style]
    litres = pred * dist_km / 100
    if own:
        source, text = "calibration", f"your car's calibration (k = {own['k']:.2f} from {own['n_trips']} trips)"
    elif prior:
        source, text = "pakwheels", (f"PakWheels average for {prior['make']} {prior['model']} "
                                     f"({prior['city_kmpl']:.0f} km/L city, {prior['highway_kmpl']:.0f} highway) "
                                     "for normal driving; the model adjusts it for style, distance and temperature")
    else:
        source, text = "none", "no correction (model not in the PakWheels table and no calibration yet)"
    out = {"car_id": car.get("id"), "dist_km": float(dist_km), "road_type": road_type, "style": style,
           "temp_c": temp_c, "petrol_pkr": petrol_pkr, "when": when.isoformat(timespec="seconds"),
           "avg_speed": float(row["avg_speed"].iloc[0]), "dur_min": float(row["dur_min"].iloc[0]),
           "l100": pred, "kmpl": 100 / pred, "litres": litres, "cost_pkr": litres * petrol_pkr if petrol_pkr else None,
           "raw_l100": raw, "k": k, "factor_source": source, "factor_text": text, "model": model_key,
           "eco_score": float(scoring.eco_scores(row)[0]), "profile_trips": int(row["n_trips"].iloc[0]),
           "by_style": {st: {"l100": v[2], "kmpl": 100 / v[2], "litres": v[2] * dist_km / 100,
                             "cost_pkr": v[2] * dist_km / 100 * petrol_pkr if petrol_pkr else None}
                        for st, v in rows.items()}}
    eco = out["by_style"]["eco"]
    out["eco_saving_litres"] = max(0.0, litres - eco["litres"])
    out["eco_saving_pkr"] = out["eco_saving_litres"] * petrol_pkr if petrol_pkr else None
    if claimed_kmpl:
        c_l = dist_km / float(claimed_kmpl)
        out["claimed"] = {"kmpl": float(claimed_kmpl), "litres": c_l,
                          "cost_pkr": c_l * petrol_pkr if petrol_pkr else None,
                          "difference_pct": (out["kmpl"] / float(claimed_kmpl) - 1) * 100}
    return out


def save_estimate(user: dict, car: dict, est: dict) -> int:
    """Store an estimate as a trip (flagged quick estimate: no log, no route, no measured fuel)."""
    return db.add_trip(
        user_id=user["id"], car_id=car["id"], started_at=est["when"], dist_km=round(est["dist_km"], 2),
        dur_min=round(est["dur_min"], 1), avg_speed=round(est["avg_speed"], 1), road_type=est["road_type"],
        measured_l100=None, pred_l100=round(est["l100"], 2), tier="behaviour_only", model=est["model"],
        calibrated=int(est["factor_source"] == "calibration"), eco_score=est["eco_score"], style=est["style"],
        cost_pkr=round(est["cost_pkr"], 0) if est["cost_pkr"] is not None else None,
        tips_json={"quick_estimate": True, "tips": [], "k": round(est["k"], 4),
                   "calibration": "quick estimate: " + est["factor_text"],
                   "reference_adjusted": est["factor_source"] == "pakwheels", "main_l100": est["l100"],
                   "eco_l100": est["by_style"]["eco"]["l100"],
                   "possible_saving_pct": max(0.0, (1 - est["by_style"]["eco"]["l100"] / est["l100"]) * 100),
                   "inputs": {k: est[k] for k in ("dist_km", "road_type", "style", "temp_c", "petrol_pkr")},
                   "claimed": est.get("claimed")},
        route_json=None, raw_file=None)
