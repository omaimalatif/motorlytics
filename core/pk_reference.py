"""Pakistani reference fuel economy (PakWheels) and the per-car starting factor it gives (no Streamlit code).

Why: the model was trained on US cars (VED, Ann Arbor 2017-18, engines 1.4 L and up). It only accepts engine
sizes from about 1.5 L (inputs are clipped to the training range), so small Pakistani cars such as a 660 cc
Alto are predicted like a 1.5 L car, and US driving differs from Pakistani traffic.

What: assets/pk_fuel_reference.csv holds PakWheels' city / highway km/L per variant (source link and date per
row; PakWheels does not say whether these are owner reports or manufacturer claims). For a matching car, the
app predicts fuel on two fixed standard cycles (assets/pk_cycles/city.csv, highway.csv) and sets
    k_city    = (100 / PakWheels city km/L)    / predicted L/100 km on the city cycle
    k_highway = (100 / PakWheels highway km/L) / predicted L/100 km on the highway cycle
Trips then use k_city below 30 km/h average speed, k_highway above 60 km/h, and a linear blend in between.
This is a starting correction for the speed-only (behaviour_only) model until the car has its own
calibration from MAF trips or a fill-up, which always replaces it.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from core import db

REF_CSV = db.ROOT / "assets" / "pk_fuel_reference.csv"
CYCLES = db.ROOT / "assets" / "pk_cycles"
ENGINE_TOLERANCE_CC = 150
K_RANGE = (0.3, 2.0)                   # sanity limits for the factor


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    if not REF_CSV.exists():
        return pd.DataFrame()
    t = pd.read_csv(REF_CSV)
    t["_make"], t["_model"] = t["make"].map(_norm), t["model"].map(_norm)
    return t


def match(car: dict) -> dict | None:
    """PakWheels figures for this car (make + model, engine within 150 cc, same transmission and fuel type
    where the table has them; median over matching variants), or None if the model is not in the table."""
    t = table()
    if t.empty or not car:
        return None
    m = t[(t["_make"] == _norm(car.get("make"))) & (t["_model"] == _norm(car.get("model")))]
    if m.empty:
        return None
    disp = car.get("disp_L")
    if disp:
        near = m[(m["engine_cc"] - disp * 1000).abs() <= ENGINE_TOLERANCE_CC]
        if near.empty:                                # engine not listed: closest listed engine
            near = m[(m["engine_cc"] - disp * 1000).abs() == (m["engine_cc"] - disp * 1000).abs().min()]
        m = near
    fuel = (car.get("fuel_type") or "petrol").lower()
    if (m["fuel"].str.lower() == fuel).any():
        m = m[m["fuel"].str.lower() == fuel]
    if car.get("transmission") and (m["transmission"] == car["transmission"]).any():
        m = m[m["transmission"] == car["transmission"]]
    return {"make": m["make"].iloc[0], "model": m["model"].iloc[0], "engine_cc": int(m["engine_cc"].median()),
            "city_kmpl": float(m["city_kmpl"].median()), "highway_kmpl": float(m["highway_kmpl"].median()),
            "variants": m["variant"].tolist(), "source_url": m["source_url"].iloc[0],
            "retrieved": m["retrieved"].iloc[0]}


def _cycle_predictions(disp_L, cylinders) -> dict:
    """Uncalibrated speed-only predictions (L/100 km) of this engine on the two standard cycles."""
    from core import app_core
    car = {"disp_L": disp_L, "cylinders": cylinders, "boost": 0, "weight": np.nan, "cls": "Car"}
    out = {}
    for name in ("city", "highway"):
        res = app_core.predict_logs([pd.read_csv(CYCLES / f"{name}.csv")], car, car_key=None,
                                    feature_set="behaviour_only", update_calibration=False)
        trips = res["trips"]
        km = sum(r["dist_km"] for r in trips)
        out[name] = sum(r["pred_l_per_100km_uncalibrated"] * r["dist_km"] for r in trips) / km
    return out


@lru_cache(maxsize=64)
def _prior_cached(make, model, disp_L, cylinders, transmission, fuel_type) -> dict | None:
    car = {"make": make, "model": model, "disp_L": disp_L, "cylinders": cylinders,
           "transmission": transmission, "fuel_type": fuel_type}
    ref = match(car)
    if not ref:
        return None
    pred = _cycle_predictions(disp_L if disp_L else np.nan, cylinders if cylinders else np.nan)
    k_city = float(np.clip((100 / ref["city_kmpl"]) / pred["city"], *K_RANGE))
    k_hwy = float(np.clip((100 / ref["highway_kmpl"]) / pred["highway"], *K_RANGE))
    return {**ref, "k_city": k_city, "k_highway": k_hwy, "pred_city_l100": pred["city"],
            "pred_highway_l100": pred["highway"],
            "text": (f"adjusted to the PakWheels average for {ref['make']} {ref['model']} "
                     f"({ref['city_kmpl']:.0f} km/L city, {ref['highway_kmpl']:.0f} km/L highway) until your "
                     "car is calibrated")}


def prior(car: dict) -> dict | None:
    """Starting factors for this car, or None if its model is not in the PakWheels table."""
    if not car:
        return None
    return _prior_cached(car.get("make"), car.get("model"), car.get("disp_L"), car.get("cylinders"),
                         car.get("transmission"), car.get("fuel_type"))


def k_for_speed(p: dict, avg_speed: float) -> float:
    """City factor below 30 km/h average speed, highway factor above 60, linear blend in between."""
    w = float(np.clip((avg_speed - 30) / 30, 0, 1)) if avg_speed == avg_speed else 0.0
    return (1 - w) * p["k_city"] + w * p["k_highway"]
