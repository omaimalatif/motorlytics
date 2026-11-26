"""
Predict fuel use for an UNSEEN trip log and give eco-driving tips (the viva demo).

Works with a VED file, an enviroCar export, or a phone OBD log
(Torque / Car Scanner CSV with time, speed, RPM, MAF columns).

    python 05_predict_trip.py my_trip.csv --disp 1.0 --cyl 3 --cls Car --petrol 265
    python 05_predict_trip.py my_trip.csv --disp 1.3 --cyl 4 --temp 31 --car-id my_civic
    python 05_predict_trip.py trip1.csv trip2.csv --disp 1.3 --car-id my_civic --fillup-litres 9.4

--petrol is the current PKR price per litre; enter today's price, it is not built in.

This is a thin command-line wrapper: the logic (model tier, per-car calibration, tips, costs) lives in
core/app_core.py (predict_logs), shared with the Streamlit app. See that module for the details.
"""
import argparse
import json

import numpy as np
import pandas as pd

import motorlytics_features as mf
from core.app_core import predict_logs

APP_ONLY = ("route", "speed_trace", "start_dt", "eco_score", "measured", "calibrated")

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
                help="override the model tier (see core/app_core.py for the defaults)")
ap.add_argument("--car-id", default=None, help="car id for per-car calibration (models/calibration.json)")
ap.add_argument("--fillup-litres", type=float, default=None,
                help="litres filled after driving these logs: calibrates k = L / sum(predicted litres)")
ap.add_argument("--no-update", action="store_true", help="do not add these trips to the calibration store")
ap.add_argument("--calibration", default=str(mf.CALIB_PATH), help="calibration store path")
ap.add_argument("--out", default="prediction.json")
a = ap.parse_args()
if a.fillup_litres is not None and not a.car_id:
    raise SystemExit("--fillup-litres needs --car-id (the calibration is stored per car)")

car = {"disp_L": a.disp, "cylinders": a.cyl, "boost": a.boost, "weight": a.weight, "cls": a.cls}
try:
    res = predict_logs([pd.read_csv(p, low_memory=False) for p in a.csv], car, car_key=a.car_id, petrol=a.petrol,
                       temp=a.temp, feature_set=a.feature_set, calibration_path=a.calibration,
                       update_calibration=not a.no_update, fillup_litres=a.fillup_litres, verbose=True)
except ValueError as e:
    raise SystemExit(str(e))

fs, cal_txt = res["tier"], res["calibration"]
print(f"\nmodel tier: {fs} -> {res['model']} ({res['why']}); tips from {res['tips_model']}")
print(f"calibration: {cal_txt}")
for r in res["trips"]:
    print(f"\nTrip {r['trip']}: {r['dist_km']} km, {r['duration_min']} min, avg {r['avg_speed']} km/h, "
          f"idle {r['idle_pct']}%  road={r['road_type']}  style={r['style']}")
    if "measured_l_per_100km_from_maf" in r:
        print(f"  measured {r['measured_l_per_100km_from_maf']} L/100 km (from MAF)   "
              f"model ({fs}) predicted {r['pred_l_per_100km']} [{'calibrated' if res['calibrated'] else 'uncalibrated'}]")
    else:
        print(f"  predicted {r['pred_l_per_100km']} L/100 km (model: {fs}, {cal_txt})")
    print(f"  eco-driving estimate {r['eco_l_per_100km']} L/100 km -> up to {r['possible_saving_pct']}% saving")
    if "trip_cost_pkr" in r:
        print(f"  trip cost about PKR {r['trip_cost_pkr']:.0f}; eco driving saves about "
              f"PKR {r['eco_saving_pkr_per_100km']:.0f} per 100 km")
    for tip in r["tips"][:3]:
        print(f"  tip (-{tip['saving_l_per_100km']} L/100 km): {tip['tip']}")

u = res["store_update"]
if u:
    print(f"\ncalibration store updated for car '{a.car_id}': {len(u['added'])} entr{'y' if len(u['added']) == 1 else 'ies'} "
          f"added; now " + ", ".join(f"{tr} k={c['k']:.3f} ({c['n_trips']} trips)" for tr, c in u["now"].items()))

with open(a.out, "w") as fh:
    json.dump([{k: v for k, v in r.items() if k not in APP_ONLY} for r in res["trips"]], fh, indent=2, default=str)
print(f"\nsaved {a.out}")
