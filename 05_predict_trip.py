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
ap.add_argument("--feature-set", choices=["full", "behaviour_only"], default=None,
                help="default: full if the log has RPM, else behaviour_only")
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
fs = a.feature_set or ("full" if "rpm" in df and df["rpm"].notna().mean() > 0.5 else "behaviour_only")
bundle = joblib.load(meta["models"][meta[f"best_{fs}"]]["path"])
style = joblib.load(mf.MODELS / "driving_style.joblib")
trips["style"] = mf.predict_style(style, trips)
# eco profile per road type (city / mixed / highway); recommend() picks the one matching each trip
eco_profiles = style["eco_profile"]

results = []
for _, t in trips.iterrows():
    r = mf.recommend(bundle, t, eco_profiles, a.petrol, eco_p75=style.get("eco_p75"))
    r.update({"trip": t["trip_key"], "dist_km": round(t["dist_km"], 2),
              "duration_min": round(t["dur_min"], 1), "avg_speed": round(t["avg_speed"], 1),
              "idle_pct": round(100 * t["idle_frac"], 1), "style": t["style"], "model": meta[f"best_{fs}"]})
    if "l_per_100km" in t and pd.notna(t["l_per_100km"]):
        r["measured_l_per_100km_from_maf"] = round(float(t["l_per_100km"]), 2)
    results.append(r)
    print(f"\nTrip {r['trip']}: {r['dist_km']} km, {r['duration_min']} min, avg {r['avg_speed']} km/h, "
          f"idle {r['idle_pct']}%  road={r['road_type']}  style={r['style']}")
    print(f"  predicted {r['pred_l_per_100km']} L/100 km"
          + (f"  (measured from MAF: {r['measured_l_per_100km_from_maf']})" if "measured_l_per_100km_from_maf" in r else ""))
    print(f"  eco-driving estimate {r['eco_l_per_100km']} L/100 km  -> up to {r['possible_saving_pct']}% saving")
    if "trip_cost_pkr" in r:
        print(f"  trip cost about PKR {r['trip_cost_pkr']:.0f}; eco driving saves about "
              f"PKR {r['eco_saving_pkr_per_100km']:.0f} per 100 km")
    for tip in r["tips"][:3]:
        print(f"  tip (-{tip['saving_l_per_100km']} L/100 km): {tip['tip']}")

with open(a.out, "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print(f"\nsaved {a.out}")