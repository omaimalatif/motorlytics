"""
Predict fuel use for an UNSEEN trip log and give eco-driving tips (the viva demo).

Works with a VED file, an enviroCar export, or a phone OBD log
(Torque / Car Scanner CSV with time, speed, RPM, MAF columns).

    python 05_predict_trip.py my_trip.csv --disp 1.0 --cyl 3 --cls Car --petrol 265
    python 05_predict_trip.py my_trip.csv --disp 1.3 --cyl 4 --temp 31

--petrol is the current PKR price per litre; enter today's price, it is not built in.
If the log has MAF, the "measured" L/100 km (from MAF) is printed next to the
prediction, which is your real-world check.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import motorlytics_features as mf

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--disp", type=float, required=True, help="engine size in litres, e.g. 1.0")
ap.add_argument("--cyl", type=float, default=np.nan)
ap.add_argument("--boost", type=float, default=0)
ap.add_argument("--weight", type=float, default=np.nan)
ap.add_argument("--cls", default="Car", help="vehicle class as in VED static data, e.g. Car, SUV")
ap.add_argument("--temp", type=float, default=np.nan, help="air temperature, deg C")
ap.add_argument("--petrol", type=float, default=None, help="petrol price, PKR per litre")
ap.add_argument("--feature-set", choices=["full", "full_no_load", "behaviour_only"], default=None,
                help="default: full if the log has engine load, full_no_load if it has RPM but no load, "
                     "else behaviour_only")
ap.add_argument("--out", default="prediction.json")
a = ap.parse_args()

raw = pd.read_csv(a.csv, low_memory=False)
df = mf.standardize(raw, verbose=True)
if "speed" not in df:
    raise SystemExit("the log needs a speed column")
df = mf.add_time_and_trips(df)
df = mf.add_row_features(df)
trips = mf.aggregate_trips(df)
trips = trips[(trips["dist_km"] >= 1) & (trips["dur_min"] >= 1)]
if trips.empty:
    raise SystemExit("no trip of at least 1 km found in this log")

trips["disp_L"], trips["cylinders"], trips["boost"], trips["weight"] = a.disp, a.cyl, a.boost, a.weight
trips["temp_c"] = a.temp
trips["veh_class"] = a.cls
trips[f"cls_{a.cls}"] = 1.0

meta = json.loads((mf.MODELS / "meta_trip.json").read_text())


def has(col):
    return col in df and df[col].notna().mean() > 0.5


# model tier. Engine-sensor models (full / full_no_load) only transfer to logs recorded like VED:
# on enviroCar they over-predicted by 2-4 L/100 km (different "engine load" definition, gearing),
# while behaviour_only held up. So non-VED logs default to behaviour_only.
is_ved = "t_ms" in df and "trip" in df
if a.feature_set:
    fs, why = a.feature_set, "chosen explicitly with --feature-set"
elif not is_ved:
    fs, why = "behaviour_only", ("non-VED log: engine-sensor models do not transfer between loggers, "
                                 "use --feature-set full or full_no_load to override")
elif has("rpm") and has("load"):
    fs, why = "full", "VED log with RPM and engine load"
elif has("rpm"):
    fs, why = "full_no_load", "VED log with RPM but no engine load"
else:
    fs, why = "behaviour_only", "VED log without RPM"
if f"best_{fs}" not in meta and fs == "full_no_load":    # older meta_trip.json without this tier
    fs, why = "full", why + "; no full_no_load model saved, using full (load is imputed)"
bundle = joblib.load(meta["models"][meta[f"best_{fs}"]]["path"])
# tips always come from the behaviour model: they are about driving behaviour, and it transfers best
tip_bundle = bundle if fs == "behaviour_only" else joblib.load(
    meta["models"][meta["best_behaviour_only"]]["path"])
print(f"\nmodel tier: {fs} -> {meta[f'best_{fs}']} ({why}); tips from {meta['best_behaviour_only']}")
style = joblib.load(mf.MODELS / "driving_style.joblib")
trips["style"] = mf.predict_style(style, trips)
# eco profile per road type (city / mixed / highway); recommend() picks the one matching each trip
eco_profiles = style["eco_profile"]

results = []
for _, t in trips.iterrows():
    r = mf.recommend(tip_bundle, t, eco_profiles, a.petrol, eco_p75=style.get("eco_p75"))
    tip_pred, tip_eco = r["pred_l_per_100km"], r["eco_l_per_100km"]   # both from the behaviour model
    if tip_bundle is not bundle:                  # prediction from the chosen tier, tips from behaviour
        r["pred_l_per_100km"] = round(float(bundle["model"].predict(
            mf.to_matrix(t.to_frame().T, bundle["features"]))[0]), 2)
    r.update({"trip": t["trip_key"], "dist_km": round(t["dist_km"], 2),
              "duration_min": round(t["dur_min"], 1), "avg_speed": round(t["avg_speed"], 1),
              "idle_pct": round(100 * t["idle_frac"], 1), "style": t["style"], "model": meta[f"best_{fs}"],
              "tier": fs, "tips_model": meta["best_behaviour_only"]})
    measured = float(t["l_per_100km"]) if "l_per_100km" in t and pd.notna(t["l_per_100km"]) else None
    if measured is not None:
        r["measured_l_per_100km_from_maf"] = round(measured, 2)
        # scale the eco estimate to the measured figure: eco_measured = measured x (eco_model / pred_model)
        eco_measured = measured * (tip_eco / tip_pred) if tip_pred > 0 else measured
        r["eco_l_per_100km_model"] = tip_eco
        r["eco_l_per_100km"] = round(eco_measured, 2)
        if a.petrol:                              # the measured figure is the main one, so cost uses it
            r["trip_cost_pkr"] = round(measured / 100 * t["dist_km"] * a.petrol, 0)
            r["cost_per_100km_pkr"] = round(measured * a.petrol, 0)
            r["eco_saving_pkr_per_100km"] = round((measured - eco_measured) * a.petrol, 0)
    results.append(r)
    print(f"\nTrip {r['trip']}: {r['dist_km']} km, {r['duration_min']} min, avg {r['avg_speed']} km/h, "
          f"idle {r['idle_pct']}%  road={r['road_type']}  style={r['style']}")
    if measured is not None:
        print(f"  measured {r['measured_l_per_100km_from_maf']} L/100 km (from MAF)   "
              f"model ({fs}) predicted {r['pred_l_per_100km']}")
    else:
        print(f"  predicted {r['pred_l_per_100km']} L/100 km (model: {fs})")
    print(f"  eco-driving estimate {r['eco_l_per_100km']} L/100 km "
          f"({'scaled to measured' if measured is not None else 'behaviour model'}) "
          f"-> up to {r['possible_saving_pct']}% saving")
    if "trip_cost_pkr" in r:
        print(f"  trip cost about PKR {r['trip_cost_pkr']:.0f}; eco driving saves about "
              f"PKR {r['eco_saving_pkr_per_100km']:.0f} per 100 km")
    for tip in r["tips"][:3]:
        print(f"  tip (-{tip['saving_l_per_100km']} L/100 km): {tip['tip']}")

with open(a.out, "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print(f"\nsaved {a.out}")