"""
Live API source #2 (and unseen validation data): enviroCar open driving tracks.

enviroCar is a citizen-science platform where drivers upload OBD-II tracks
(speed, RPM, MAF, consumption) as open data (ODbL licence).
API docs: http://envirocar.github.io/enviroCar-server/api/

    python 01b_fetch_envirocar.py --tracks 300

Saves:
  Data/envirocar_rows.csv              every measurement (counts toward the 10k-record rule)
  Data/processed/envirocar_trips.parquet  trip-level features, same schema as VED trips

Field names below follow the enviroCar JSON as documented; if the API returns
something different, print one track (--debug) and adjust PHEN.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import requests

import motorlytics_features as mf

BASE = "https://envirocar.org/api/stable"
PHEN = {"Speed": "speed", "GPS Speed": "gps_speed", "Rpm": "rpm", "MAF": "maf",
        "Calculated MAF": "calc_maf", "Engine Load": "load", "Consumption": "fuel_rate_obd",
        "Intake Temperature": "iat"}

ap = argparse.ArgumentParser()
ap.add_argument("--tracks", type=int, default=300)
ap.add_argument("--debug", action="store_true")
a = ap.parse_args()
sess = requests.Session()
sess.headers["Accept"] = "application/json"


def get(url, **params):
    for attempt in range(3):
        try:
            r = sess.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("  retry", attempt + 1, e)
            time.sleep(3)
    return None


rows, page = [], 1
seen = 0
while seen < a.tracks:
    lst = get(f"{BASE}/tracks", limit=100, page=page)
    if not lst or not lst.get("tracks"):
        break
    for t in lst["tracks"]:
        if seen >= a.tracks:
            break
        tr = get(f"{BASE}/tracks/{t['id']}")
        if not tr:
            continue
        if a.debug and seen == 0:
            print(json.dumps(tr, indent=1)[:3000])
        props = tr.get("properties", {})
        sensor = (props.get("sensor") or {}).get("properties", {})
        fuel = str(sensor.get("fuelType", "")).lower()
        for f in tr.get("features", []):
            p = f.get("properties", {})
            ph = p.get("phenomenons", {})
            rec = {"trip": t["id"], "time": p.get("time"),
                   "lon": (f.get("geometry") or {}).get("coordinates", [None, None])[0],
                   "lat": (f.get("geometry") or {}).get("coordinates", [None, None])[1],
                   "fuel_type": fuel, "make": sensor.get("manufacturer"),
                   "model": sensor.get("model"), "year": sensor.get("constructionYear"),
                   "disp_ccm": sensor.get("engineDisplacement")}
            for k, v in PHEN.items():
                rec[v] = (ph.get(k) or {}).get("value")
            rows.append(rec)
        seen += 1
        if seen % 25 == 0:
            print(f"{seen} tracks, {len(rows):,} measurements")
        time.sleep(0.3)                        # be polite to a free service
    page += 1

raw = pd.DataFrame(rows)
mf.DATA.mkdir(exist_ok=True)
raw.to_csv(mf.DATA / "envirocar_rows.csv", index=False)
print(f"saved {len(raw):,} measurements from {raw['trip'].nunique() if len(raw) else 0} tracks")
if raw.empty:
    raise SystemExit("no data returned; run with --debug and check the API")

# ---- same feature pipeline as VED, gasoline cars only ----
df = raw[raw["fuel_type"].str.contains("gasoline|petrol", na=False)].copy()
if df["maf"].isna().all() and "calc_maf" in df:
    df["maf"] = df["calc_maf"]      # estimated from MAP; flag this in the report
df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(pd.to_numeric(df["gps_speed"], errors="coerce"))
for c in ["rpm", "maf", "load", "fuel_rate_obd", "lat", "lon"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["vehid"] = (df["make"].astype(str) + "_" + df["model"].astype(str) + "_" + df["year"].astype(str))
df = mf.add_time_and_trips(df.drop(columns=["fuel_type"]))
df = mf.add_row_features(df)
trips = mf.aggregate_trips(df)
spec = (raw.drop_duplicates("trip").set_index("trip")[["disp_ccm"]]
        .assign(disp_L=lambda s: pd.to_numeric(s["disp_ccm"], errors="coerce") / 1000))
trips["track_id"] = trips["trip_key"].str.split("_").str[-1]
trips = trips.merge(spec[["disp_L"]], left_on="track_id", right_index=True, how="left")
trips = trips[(trips["dist_km"] >= 1) & (trips["dur_min"] >= 2)]
if "l_per_100km" in trips:
    trips = trips[trips["l_per_100km"].between(2, 40)]
mf.PROC.mkdir(parents=True, exist_ok=True)
trips.to_parquet(mf.PROC / "envirocar_trips.parquet", index=False)
print(f"usable gasoline trips: {len(trips)}  -> {mf.PROC / 'envirocar_trips.parquet'}")
print(trips[["dist_km", "avg_speed", "l_per_100km"]].describe().round(2).to_string())