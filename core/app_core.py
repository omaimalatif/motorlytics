"""Prediction core shared by the app and 05_predict_trip.py (no Streamlit code).

predict_logs(dfs, car, ...) turns one or more raw trip logs (VED, enviroCar, Torque / Car Scanner CSV)
into trips with: measured L/100 km (if the log has MAF), the model prediction for the chosen tier,
per-car calibration, eco-driving tips, cost, driving style, eco-score, GPS route and speed trace.

Model tier: VED logs pick full / full_no_load / behaviour_only from their columns. Other logs use
behaviour_only (engine-sensor models did not transfer between loggers on enviroCar / exp1), unless
the car is calibrated for the full tier and the log has RPM and load: the per-car k corrects the
logger offset, so full is used. feature_set always overrides.

Per-car calibration (models/calibration.json, motorlytics_features section 7): k = median(measured /
predicted) per tier over the car's 10 most recent calibration trips; a fill-up adds one entry
k = litres / sum(predicted litres). A log is predicted with the k stored BEFORE it is added.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import motorlytics_features as mf
from core import scoring

TIERS = ["behaviour_only", "full_no_load", "full"]
MAX_ROUTE_POINTS = 400
MAX_SPEED_POINTS = 600


@lru_cache(maxsize=1)
def _meta() -> dict:
    return json.loads((mf.MODELS / "meta_trip.json").read_text())


@lru_cache(maxsize=8)
def _bundle(key: str) -> dict:
    return joblib.load(_meta()["models"][key]["path"])


@lru_cache(maxsize=1)
def _style() -> dict:
    return joblib.load(mf.MODELS / "driving_style.joblib")


def reload_models() -> None:
    """Clear cached models (after retraining)."""
    for f in (_meta, _bundle, _style):
        f.cache_clear()


def model_info() -> dict:
    """The app's model per tier and the reference model, from meta_trip.json."""
    m = _meta()
    return {fs: {"app": m.get(f"best_{fs}"), "reference": m.get(f"best_reference_{fs}")} for fs in TIERS
            if f"best_{fs}" in m}


def _downsample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) <= n:
        return arr
    return arr[np.linspace(0, len(arr) - 1, n).astype(int)]


def calibration_status(car_key, calibration_path=None) -> dict:
    """{tier: {'k', 'n_trips', 'n_entries', 'last'}} for the tiers this car is calibrated for."""
    store = mf.load_calibration(calibration_path or mf.CALIB_PATH)
    return {t: c for t in TIERS if (c := mf.car_k(store, car_key, t))}


def predict_logs(dfs, car: dict, car_key=None, petrol=None, temp=np.nan, feature_set=None,
                 calibration_path=None, update_calibration=True, fillup_litres=None, verbose=False) -> dict:
    """Predict every trip of >= 1 km in the given raw logs.

    dfs: list of raw DataFrames (one per log file).
    car: {'disp_L', 'cylinders', 'boost' (0), 'weight' (NaN), 'cls' ('Car')}.
    car_key: id in the calibration store (None = uncalibrated, nothing stored).
    Returns {'tier', 'why', 'model', 'tips_model', 'calibration', 'calibrated', 'k', 'trips': [...],
             'store_update': {'added': [...], 'now': {...}} or None}.
    Raises ValueError for unusable logs.
    """
    if fillup_litres is not None and not car_key:
        raise ValueError("a fill-up needs a car id (the calibration is stored per car)")
    calibration_path = calibration_path or mf.CALIB_PATH
    meta = _meta()

    # ---------------- read logs -> trips ----------------
    frames, is_ved = [], True
    for i, raw in enumerate(dfs):
        d = mf.standardize(raw, verbose=(verbose and i == 0))
        if "speed" not in d:
            raise ValueError(f"log {i + 1}: the log needs a speed column")
        log_is_ved = "t_ms" in d and "trip" in d
        is_ved &= log_is_ved
        if not log_is_ved:
            d["vehid"] = f"log{i + 1}"                 # keep trips from different files apart
        d = mf.add_time_and_trips(d)
        frames.append(mf.add_row_features(d))
    df = pd.concat(frames, ignore_index=True)
    trips = pd.concat([mf.aggregate_trips(f) for f in frames], ignore_index=True)
    trips = trips[(trips["dist_km"] >= 1) & (trips["dur_min"] >= 1)].reset_index(drop=True)
    if trips.empty:
        raise ValueError("no trip of at least 1 km found in the log(s)")
    trips["disp_L"], trips["cylinders"] = car.get("disp_L", np.nan), car.get("cylinders", np.nan)
    trips["boost"], trips["weight"] = car.get("boost", 0), car.get("weight", np.nan)
    trips["temp_c"] = temp
    cls = car.get("cls", "Car")
    trips["veh_class"] = cls
    trips[f"cls_{cls}"] = 1.0

    store = mf.load_calibration(calibration_path)

    def has(col):
        return col in df and df[col].notna().mean() > 0.5

    # tiers this log can feed (for calibration entries): behaviour always, engine tiers if the sensors exist
    avail = ["behaviour_only"] + (["full_no_load"] if has("rpm") else []) + \
            (["full"] if has("rpm") and has("load") else [])
    avail = [t for t in avail if f"best_{t}" in meta]

    # ---------------- model tier ----------------
    if feature_set:
        fs, why = feature_set, "chosen explicitly with --feature-set"
    elif is_ved and has("rpm") and has("load"):
        fs, why = "full", "VED log with RPM and engine load"
    elif is_ved and has("rpm"):
        fs, why = "full_no_load", "VED log with RPM but no engine load"
    elif is_ved:
        fs, why = "behaviour_only", "VED log without RPM"
    elif "full" in avail and mf.car_k(store, car_key, "full"):
        fs, why = "full", "car is calibrated for the full tier, which corrects the logger offset"
    else:
        fs, why = "behaviour_only", ("non-VED log: engine-sensor models do not transfer between loggers "
                                     "without calibration (use --feature-set to override)")
    if f"best_{fs}" not in meta and fs == "full_no_load":    # older meta_trip.json without this tier
        fs, why = "full", why + "; no full_no_load model saved, using full (load is imputed)"
    bundles = {t: _bundle(meta[f"best_{t}"]) for t in set(avail) | {fs, "behaviour_only"}}
    bundle, tip_bundle = bundles[fs], bundles["behaviour_only"]   # tips always from the behaviour model

    cal = mf.car_k(store, car_key, fs)                     # k stored BEFORE this log
    k = cal.get("k", 1.0)
    if cal:
        cal_txt = f"calibrated (k={k:.3f}, {cal['n_trips']} trips, last {cal['last'][:10]})"
    elif car_key:
        cal_txt = f"uncalibrated estimate (no calibration for car '{car_key}', tier {fs} yet)"
    else:
        cal_txt = "uncalibrated estimate (give --car-id to use per-car calibration)"

    style = _style()
    trips["style"] = mf.predict_style(style, trips)
    trips["eco_score"] = scoring.eco_scores(trips, style)
    eco_profiles = style["eco_profile"]                    # per road type; recommend() picks the match
    preds = {t: bundles[t]["model"].predict(mf.to_matrix(trips, bundles[t]["features"])) for t in avail}
    preds.setdefault(fs, bundle["model"].predict(mf.to_matrix(trips, bundle["features"])))

    # ---------------- per-trip results ----------------
    results = []
    for i, t in trips.iterrows():
        r = mf.recommend(tip_bundle, t, eco_profiles, petrol, eco_p75=style.get("eco_p75"))
        tip_pred, tip_eco = r["pred_l_per_100km"], r["eco_l_per_100km"]   # both from the behaviour model
        raw_pred = float(preds[fs][i])
        pred = raw_pred * k
        eco = pred * (tip_eco / tip_pred) if tip_pred > 0 else pred       # eco scaled to the shown prediction
        r.update({"trip": t["trip_key"], "dist_km": round(t["dist_km"], 2),
                  "duration_min": round(t["dur_min"], 1), "avg_speed": round(t["avg_speed"], 1),
                  "idle_pct": round(100 * t["idle_frac"], 1), "style": t["style"], "model": meta[f"best_{fs}"],
                  "tier": fs, "tips_model": meta["best_behaviour_only"], "calibration": cal_txt, "k": round(k, 4),
                  "pred_l_per_100km_uncalibrated": round(raw_pred, 2), "pred_l_per_100km": round(pred, 2),
                  "eco_l_per_100km_model": tip_eco, "eco_l_per_100km": round(eco, 2)})
        for tip in r["tips"]:
            tip["saving_l_per_100km"] = round(tip["saving_l_per_100km"] * k, 2)
        measured = float(t["l_per_100km"]) if "l_per_100km" in t and pd.notna(t["l_per_100km"]) else None
        main = pred
        if measured is not None:                            # measured is the main figure; eco scaled to it
            r["measured_l_per_100km_from_maf"] = round(measured, 2)
            main = measured
            r["eco_l_per_100km"] = round(measured * (tip_eco / tip_pred) if tip_pred > 0 else measured, 2)
        if petrol:
            r["trip_cost_pkr"] = round(main / 100 * t["dist_km"] * petrol, 0)
            r["cost_per_100km_pkr"] = round(main * petrol, 0)
            r["eco_saving_pkr_per_100km"] = round((main - r["eco_l_per_100km"]) * petrol, 0)
        # extras for the app (not printed by the CLI)
        rows = df[df["trip_key"] == t["trip_key"]].sort_values("t")
        r["start_dt"] = str(t["start_local"]) if pd.notna(t.get("start_local")) else \
            (str(t["start_dt"]) if pd.notna(t.get("start_dt")) else None)
        r["eco_score"] = float(t["eco_score"])
        r["measured"] = measured is not None
        r["calibrated"] = bool(cal)
        if {"lat", "lon"} <= set(rows.columns):
            ll = rows[["lat", "lon"]].dropna()
            ll = ll[(ll["lat"].abs() > 1e-6) | (ll["lon"].abs() > 1e-6)]
            r["route"] = _downsample(ll.to_numpy(), MAX_ROUTE_POINTS).round(6).tolist() if len(ll) >= 2 else None
        else:
            r["route"] = None
        sp = rows[["t", "speed"]].dropna()
        r["speed_trace"] = _downsample(np.column_stack([sp["t"] / 60, sp["speed"]]), MAX_SPEED_POINTS
                                       ).round(3).tolist() if len(sp) else []
        results.append(r)

    # ---------------- update the calibration store (after predicting) ----------------
    store_update = None
    if car_key and update_calibration:
        added = []
        if fillup_litres is not None:
            litres = {tr: float(np.sum(preds[tr] * trips["dist_km"].to_numpy() / 100)) for tr in avail}
            ratios = {tr: fillup_litres / v for tr, v in litres.items() if v > 0}
            when = str(pd.to_datetime(trips["start_dt"]).max()) if trips["start_dt"].notna().any() \
                else pd.Timestamp.now().isoformat(timespec="seconds")
            mf.add_calibration(store, car_key, ratios, len(trips), "fillup", when,
                               {"litres": fillup_litres, "km": round(float(trips["dist_km"].sum()), 2),
                                "predicted_litres": {tr: round(v, 3) for tr, v in litres.items()}})
            added.append(f"fill-up of {fillup_litres} L over {len(trips)} trips "
                         f"(k {', '.join(f'{tr}={v:.3f}' for tr, v in ratios.items())})")
        if "l_per_100km" in trips and fillup_litres is None:   # a fill-up already covers these trips
            for i, t in trips.iterrows():
                if pd.isna(t["l_per_100km"]):
                    continue
                ratios = {tr: t["l_per_100km"] / preds[tr][i] for tr in avail if preds[tr][i] > 0}
                when = str(t["start_dt"]) if pd.notna(t["start_dt"]) else pd.Timestamp.now().isoformat(timespec="seconds")
                mf.add_calibration(store, car_key, ratios, 1, "maf", when,
                                   {"trip": t["trip_key"], "measured": round(float(t["l_per_100km"]), 3)})
                added.append(f"MAF trip {t['trip_key']}")
        if added:
            mf.save_calibration(store, calibration_path)
            store_update = {"added": added, "now": {tr: c for tr in avail if (c := mf.car_k(store, car_key, tr))}}

    return {"tier": fs, "why": why, "model": meta[f"best_{fs}"], "tips_model": meta["best_behaviour_only"],
            "calibration": cal_txt, "calibrated": bool(cal), "k": k, "trips": results, "store_update": store_update}
