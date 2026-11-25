"""
Predict fuel use for an UNSEEN trip log and give eco-driving tips (the viva demo).

Works with a VED file, an enviroCar export, or a phone OBD log
(Torque / Car Scanner CSV with time, speed, RPM, MAF columns).

    python 05_predict_trip.py my_trip.csv --disp 1.0 --cyl 3 --cls Car --petrol 265
    python 05_predict_trip.py my_trip.csv --disp 1.3 --cyl 4 --temp 31 --car-id my_civic
    python 05_predict_trip.py trip1.csv trip2.csv --disp 1.3 --car-id my_civic --fillup-litres 9.4

--petrol is the current PKR price per litre; enter today's price, it is not built in.

Per-car calibration (models/calibration.json, see motorlytics_features section 7):
  On unseen cars most of the error is a per-car offset, so with --car-id each car keeps a factor
  k = median(measured / predicted) per model tier over its 10 most recent calibration trips.
  - A log with MAF: the measured L/100 km is shown as the main figure, and each trip adds a
    calibration entry for that car (skip with --no-update).
  - --fillup-litres L with one or more logs (the trips since the last fill-up): adds one entry
    k = L / sum(predicted litres over those trips).
  Predictions and costs are multiplied by k ("calibrated (k=..., n trips)"); without an entry
  the output says "uncalibrated estimate". A log is always predicted with the k stored BEFORE it
  is added, so a trip never calibrates itself.

Model tier: VED logs pick full / full_no_load / behaviour_only from their columns. Other logs use
behaviour_only (engine-sensor models did not transfer between loggers on enviroCar / exp1), unless
the car is calibrated for the full tier and the log has RPM and load: the per-car k corrects the
logger offset, so full is used. --feature-set always overrides.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import motorlytics_features as mf

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+", help="one or more trip logs")
ap.add_argument("--disp", type=float, required=True, help="engine size in litres, e.g. 1.0")
ap.add_argument("--cyl", type=float, default=np.nan)
ap.add_argument("--boost", type=float, default=0)
ap.add_argument("--weight", type=float, default=np.nan)
ap.add_argument("--cls", default="Car", help="vehicle class as in VED static data, e.g. Car, SUV")
ap.add_argument("--temp", type=float, default=np.nan, help="air temperature, deg C")
ap.add_argument("--petrol", type=float, default=None, help="petrol price, PKR per litre")
ap.add_argument("--feature-set", choices=["full", "full_no_load", "behaviour_only"], default=None,
                help="override the model tier (see the docstring for the defaults)")
ap.add_argument("--car-id", default=None, help="car id for per-car calibration (models/calibration.json)")
ap.add_argument("--fillup-litres", type=float, default=None,
                help="litres filled after driving these logs: calibrates k = L / sum(predicted litres)")
ap.add_argument("--no-update", action="store_true", help="do not add these trips to the calibration store")
ap.add_argument("--calibration", default=str(mf.CALIB_PATH), help="calibration store path")
ap.add_argument("--out", default="prediction.json")
a = ap.parse_args()
if a.fillup_litres is not None and not a.car_id:
    raise SystemExit("--fillup-litres needs --car-id (the calibration is stored per car)")

# ---------------- read logs -> trips ----------------
frames, is_ved = [], True
for i, path in enumerate(a.csv):
    d = mf.standardize(pd.read_csv(path, low_memory=False), verbose=(i == 0))
    if "speed" not in d:
        raise SystemExit(f"{path}: the log needs a speed column")
    is_ved &= "t_ms" in d and "trip" in d
    if not ("t_ms" in d and "trip" in d):
        d["vehid"] = f"log{i + 1}"                 # keep trips from different files apart
    d = mf.add_time_and_trips(d)
    frames.append(mf.add_row_features(d))
df = pd.concat(frames, ignore_index=True)
trips = pd.concat([mf.aggregate_trips(f) for f in frames], ignore_index=True)
trips = trips[(trips["dist_km"] >= 1) & (trips["dur_min"] >= 1)].reset_index(drop=True)
if trips.empty:
    raise SystemExit("no trip of at least 1 km found in the log(s)")
trips["disp_L"], trips["cylinders"], trips["boost"], trips["weight"] = a.disp, a.cyl, a.boost, a.weight
trips["temp_c"] = a.temp
trips["veh_class"] = a.cls
trips[f"cls_{a.cls}"] = 1.0

meta = json.loads((mf.MODELS / "meta_trip.json").read_text())
store = mf.load_calibration(a.calibration)


def has(col):
    return col in df and df[col].notna().mean() > 0.5


def load_tier(fs):
    return joblib.load(meta["models"][meta[f"best_{fs}"]]["path"])


# tiers this log can feed (for calibration entries): behaviour always, engine tiers if the sensors exist
avail = ["behaviour_only"] + (["full_no_load"] if has("rpm") else []) + \
        (["full"] if has("rpm") and has("load") else [])
avail = [t for t in avail if f"best_{t}" in meta]

# ---------------- model tier ----------------
if a.feature_set:
    fs, why = a.feature_set, "chosen explicitly with --feature-set"
elif is_ved and has("rpm") and has("load"):
    fs, why = "full", "VED log with RPM and engine load"
elif is_ved and has("rpm"):
    fs, why = "full_no_load", "VED log with RPM but no engine load"
elif is_ved:
    fs, why = "behaviour_only", "VED log without RPM"
elif "full" in avail and mf.car_k(store, a.car_id, "full"):
    fs, why = "full", "car is calibrated for the full tier, which corrects the logger offset"
else:
    fs, why = "behaviour_only", ("non-VED log: engine-sensor models do not transfer between loggers "
                                 "without calibration (use --feature-set to override)")
if f"best_{fs}" not in meta and fs == "full_no_load":    # older meta_trip.json without this tier
    fs, why = "full", why + "; no full_no_load model saved, using full (load is imputed)"
bundles = {t: load_tier(t) for t in set(avail) | {fs, "behaviour_only"}}
bundle, tip_bundle = bundles[fs], bundles["behaviour_only"]
# tips always come from the behaviour model: they are about driving behaviour, and it transfers best
print(f"\nmodel tier: {fs} -> {meta[f'best_{fs}']} ({why}); tips from {meta['best_behaviour_only']}")

cal = mf.car_k(store, a.car_id, fs)                    # k stored BEFORE this log
k = cal.get("k", 1.0)
if cal:
    cal_txt = f"calibrated (k={k:.3f}, {cal['n_trips']} trips, last {cal['last'][:10]})"
elif a.car_id:
    cal_txt = f"uncalibrated estimate (no calibration for car '{a.car_id}', tier {fs} yet)"
else:
    cal_txt = "uncalibrated estimate (give --car-id to use per-car calibration)"
print(f"calibration: {cal_txt}")

style = joblib.load(mf.MODELS / "driving_style.joblib")
trips["style"] = mf.predict_style(style, trips)
eco_profiles = style["eco_profile"]                    # per road type; recommend() picks the match
preds = {t: bundles[t]["model"].predict(mf.to_matrix(trips, bundles[t]["features"])) for t in avail}
preds.setdefault(fs, bundle["model"].predict(mf.to_matrix(trips, bundle["features"])))

# ---------------- per-trip output ----------------
results = []
for i, t in trips.iterrows():
    r = mf.recommend(tip_bundle, t, eco_profiles, a.petrol, eco_p75=style.get("eco_p75"))
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
    if a.petrol:
        r["trip_cost_pkr"] = round(main / 100 * t["dist_km"] * a.petrol, 0)
        r["cost_per_100km_pkr"] = round(main * a.petrol, 0)
        r["eco_saving_pkr_per_100km"] = round((main - r["eco_l_per_100km"]) * a.petrol, 0)
    results.append(r)
    print(f"\nTrip {r['trip']}: {r['dist_km']} km, {r['duration_min']} min, avg {r['avg_speed']} km/h, "
          f"idle {r['idle_pct']}%  road={r['road_type']}  style={r['style']}")
    if measured is not None:
        print(f"  measured {r['measured_l_per_100km_from_maf']} L/100 km (from MAF)   "
              f"model ({fs}) predicted {r['pred_l_per_100km']} [{'calibrated' if cal else 'uncalibrated'}]")
    else:
        print(f"  predicted {r['pred_l_per_100km']} L/100 km (model: {fs}, {cal_txt})")
    print(f"  eco-driving estimate {r['eco_l_per_100km']} L/100 km -> up to {r['possible_saving_pct']}% saving")
    if "trip_cost_pkr" in r:
        print(f"  trip cost about PKR {r['trip_cost_pkr']:.0f}; eco driving saves about "
              f"PKR {r['eco_saving_pkr_per_100km']:.0f} per 100 km")
    for tip in r["tips"][:3]:
        print(f"  tip (-{tip['saving_l_per_100km']} L/100 km): {tip['tip']}")

# ---------------- update the calibration store (after predicting) ----------------
if a.car_id and not a.no_update:
    added = []
    if a.fillup_litres is not None:
        litres = {tr: float(np.sum(preds[tr] * trips["dist_km"].to_numpy() / 100)) for tr in avail}
        ratios = {tr: a.fillup_litres / v for tr, v in litres.items() if v > 0}
        when = str(pd.to_datetime(trips["start_dt"]).max()) if trips["start_dt"].notna().any() \
            else pd.Timestamp.now().isoformat(timespec="seconds")
        mf.add_calibration(store, a.car_id, ratios, len(trips), "fillup", when,
                           {"litres": a.fillup_litres, "km": round(float(trips["dist_km"].sum()), 2),
                            "predicted_litres": {tr: round(v, 3) for tr, v in litres.items()}})
        added.append(f"fill-up of {a.fillup_litres} L over {len(trips)} trips "
                     f"(k {', '.join(f'{tr}={v:.3f}' for tr, v in ratios.items())})")
    if "l_per_100km" in trips and a.fillup_litres is None:   # a fill-up already covers these trips
        for i, t in trips.iterrows():
            if pd.isna(t["l_per_100km"]):
                continue
            ratios = {tr: t["l_per_100km"] / preds[tr][i] for tr in avail if preds[tr][i] > 0}
            when = str(t["start_dt"]) if pd.notna(t["start_dt"]) else pd.Timestamp.now().isoformat(timespec="seconds")
            mf.add_calibration(store, a.car_id, ratios, 1, "maf", when,
                               {"trip": t["trip_key"], "measured": round(float(t["l_per_100km"]), 3)})
            added.append(f"MAF trip {t['trip_key']}")
    if added:
        mf.save_calibration(store, a.calibration)
        now = {tr: mf.car_k(store, a.car_id, tr) for tr in avail}
        print(f"\ncalibration store updated for car '{a.car_id}': {len(added)} entr{'y' if len(added) == 1 else 'ies'} "
              f"added; now " + ", ".join(f"{tr} k={c['k']:.3f} ({c['n_trips']} trips)" for tr, c in now.items() if c))

with open(a.out, "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print(f"\nsaved {a.out}")
