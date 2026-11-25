"""
Build the two Motorlytics training sets from the RAW VED files.

  Data/processed/trips.parquet   one row per trip, target l_per_100km   (main task)
  Data/processed/rows.parquet    sampled time-steps, target fuel_Lph     (second-level task)

Why rebuild instead of using ved_features.pkl: the pickle has no trip id or
timestamp, so it cannot be joined to weather or summed per trip, and its rolling
features may cross trip boundaries. Here every feature is computed inside a trip.

    python 02_build_datasets.py                 # all files
    python 02_build_datasets.py --max-files 3   # quick test
"""
import argparse
import time

import numpy as np
import pandas as pd

import motorlytics_features as mf

ap = argparse.ArgumentParser()
ap.add_argument("--max-files", type=int, default=0)
ap.add_argument("--row-frac", type=float, default=0.03, help="share of rows kept for rows.parquet")
ap.add_argument("--rows-per-vehicle", type=int, default=4000)
ap.add_argument("--weather", default=None, help="UTC weather CSV (default: weather_annarbor_utc.csv, else the API file)")
ap.add_argument("--eved", default=None, help="optional eVED trip file with grade/elevation per trip")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
t0 = time.time()

# ---------------- vehicle specs: keep petrol ICE cars ----------------
static = mf.parse_static(pd.read_excel(mf.DATA / "VED_Static_Data_ICE&HEV.xlsx"))
print("vehicle types:", static["veh_type"].value_counts().to_dict())
ice = static[(static["veh_type"] == "ICE") & ~static["is_diesel"]]
ice_ids = set(ice["vehid"])
print(f"petrol ICE vehicles in static file: {len(ice_ids)}")

# ---------------- weather ----------------
wpath = a.weather or next((p for p in [mf.DATA / "weather_annarbor_utc.csv",
                                        mf.DATA / "weather_annarbor_api.csv"] if p.exists()), None)
weather = mf.load_weather(wpath) if wpath else None
print("weather file:", wpath)

# ---------------- dynamic files ----------------
files = sorted(mf.DATA.glob("VED_DynamicData_Part*/**/*.csv"))
if a.max_files:
    files = files[: a.max_files]
print(f"{len(files)} dynamic files")
if not files:
    raise SystemExit("no files found under Data/VED_DynamicData_Part*/")

trip_parts, row_parts = [], []
for i, f in enumerate(files):
    raw = pd.read_csv(f, low_memory=False)
    df = mf.standardize(raw, verbose=(i == 0))
    df["vehid"] = pd.to_numeric(df["vehid"], errors="coerce")
    df = df[df["vehid"].isin(ice_ids) & df["speed"].notna()]
    if df.empty:
        continue
    df = mf.add_time_and_trips(df)
    df = mf.add_row_features(df)
    trip_parts.append(mf.aggregate_trips(df))
    ok = df["fuel_Lph"].notna() & df["accel"].notna() if "fuel_Lph" in df else df["accel"].notna()
    keep = df[ok & (rng.random(len(df)) < a.row_frac)]
    row_parts.append(keep)
    print(f"[{i + 1}/{len(files)}] {f.name}: {len(df):,} rows, "
          f"{df['trip_key'].nunique():,} trips  ({time.time() - t0:.0f}s)")

trips = pd.concat(trip_parts, ignore_index=True)
rows = pd.concat(row_parts, ignore_index=True)

# ---------------- trip cleaning ----------------
n0 = len(trips)
trips = trips[(trips["dist_km"] >= 1) & (trips["dur_min"] >= 2)]
trips = trips[trips["fuel_coverage"] >= 0.8]
trips = trips[trips["l_per_100km"].between(2, 40)]          # physically plausible for cars
print(f"\ntrip filter kept {len(trips):,} of {n0:,} trips")

# ---------------- joins ----------------
spec_cols = ["vehid", "veh_class", "disp_L", "cylinders", "boost", "weight"]
trips = trips.merge(static[spec_cols], on="vehid", how="left")
if weather is not None:
    trips = mf.join_weather(trips, weather)
    print(f"weather matched for {trips['temp_c'].notna().mean():.1%} of trips")
if a.eved:
    ev = pd.read_parquet(a.eved) if a.eved.endswith(".parquet") else pd.read_csv(a.eved)
    trips = trips.merge(ev, on="trip_key", how="left")
classes = sorted(trips["veh_class"].fillna("unknown").unique())
trips = mf.add_class_dummies(trips, classes)

# outliers: per-vehicle robust z-score on the target (flag, then drop extreme ones)
med = trips.groupby("vehid")["l_per_100km"].transform("median")
mad = trips.groupby("vehid")["l_per_100km"].transform(lambda s: (s - s.median()).abs().median())
trips["target_robust_z"] = (trips["l_per_100km"] - med) / (1.4826 * mad.replace(0, np.nan))
n1 = len(trips)
trips = trips[~(trips["target_robust_z"].abs() > 6)]
print(f"robust-z outlier filter removed {n1 - len(trips)} trips")

# ---------------- row-level set ----------------
rows = rows[rows["trip_key"].isin(trips["trip_key"])]
rows = rows.sample(frac=1, random_state=a.seed)                 # shuffle, then cap per vehicle
rows = rows[rows.groupby("vehid").cumcount() < a.rows_per_vehicle]
ctx = ["trip_key", "veh_class", "disp_L", "cylinders", "boost", "weight", "temp_c", "humidity",
       "precip_mm", "wind_kmh"] + [c for c in trips if c.startswith("cls_")]
rows = rows.merge(trips[[c for c in ctx if c in trips]], on="trip_key", how="left")
cap = rows["fuel_Lph"].quantile(0.999)
rows = rows[rows["fuel_Lph"].between(0, cap)]

mf.PROC.mkdir(parents=True, exist_ok=True)
trips.to_parquet(mf.PROC / "trips.parquet", index=False)
rows.to_parquet(mf.PROC / "rows.parquet", index=False)

print(f"\nTRIPS: {len(trips):,} trips, {trips['vehid'].nunique()} vehicles -> {mf.PROC / 'trips.parquet'}")
print(trips[["dist_km", "dur_min", "avg_speed", "idle_frac", "rpa", "l_per_100km"]]
      .describe().round(3).to_string())
print(f"\nROWS: {len(rows):,} rows, {rows['vehid'].nunique()} vehicles -> {mf.PROC / 'rows.parquet'}")
print(f"done in {time.time() - t0:.0f}s")