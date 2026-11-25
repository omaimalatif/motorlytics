"""
Live API source #2 (and unseen validation data): enviroCar open driving tracks.

enviroCar is a citizen-science platform where drivers upload OBD-II tracks
(speed, RPM, MAF, consumption) as open data (ODbL licence).
API docs: http://envirocar.github.io/enviroCar-server/api/

    python 01b_fetch_envirocar.py                       # >= 30 petrol vehicles, <= 2,000 tracks downloaded
    python 01b_fetch_envirocar.py --min-vehicles 40 --per-vehicle 10 --max-tracks 3000

How tracks are chosen: the track list (newest first) already carries each track's car
(manufacturer, model, year, engine size, weight, fuel type, vehicle id). The script pages back
through older tracks, skips non-petrol cars ("gas" in enviroCar = CNG/LPG) and vehicles that
already have --per-vehicle usable trips, downloads the rest, and runs each through the same
feature pipeline and filters as VED (>= 1 km, >= 2 min, >= 80% MAF coverage, 2-40 L/100 km).
It stops once --min-vehicles vehicles have usable trips, or --max-tracks tracks were downloaded.

Target: fuel from real MAF only; tracks with only "Calculated MAF" are skipped unless
--allow-calc-maf (calculated MAF is estimated from intake pressure, flag it in the report).

Context added per track: Open-Meteo hourly weather at the track's start point and hour
(cached in Data/envirocar_weather_cache.json), local time for hour / weekday features,
weight (kg -> lb, as in VED), cylinders estimated from engine size. Turbo is unknown (NaN).

Saves:
  Data/envirocar_rows.csv                  every downloaded measurement (counts toward the 10k-record rule)
  Data/processed/envirocar_trips.parquet   usable trips, same schema as VED trips (+ veh_label, cylinders_estimated)
  Data/processed/envirocar_vehicles.csv    one row per vehicle with usable trips
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import requests

import motorlytics_features as mf

BASE = "https://envirocar.org/api/stable"
METEO = "https://archive-api.open-meteo.com/v1/archive"
PHEN = {"Speed": "speed", "GPS Speed": "gps_speed", "Rpm": "rpm", "MAF": "maf",
        "Calculated MAF": "calc_maf", "Engine Load": "load", "Consumption": "fuel_rate_obd",
        "Intake Temperature": "iat"}
KG_TO_LB = 2.20462

ap = argparse.ArgumentParser()
ap.add_argument("--min-vehicles", type=int, default=30, help="stop once this many vehicles have usable trips")
ap.add_argument("--max-tracks", "--tracks", type=int, default=2000, help="cap on tracks downloaded")
ap.add_argument("--per-vehicle", type=int, default=10, help="max usable trips kept per vehicle")
ap.add_argument("--start-page", type=int, default=1)
ap.add_argument("--allow-calc-maf", action="store_true", help="use Calculated MAF when a track has no MAF")
ap.add_argument("--no-weather", action="store_true")
ap.add_argument("--debug", action="store_true", help="print the first downloaded track's JSON")
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


def estimate_cylinders(disp_L):
    """Typical petrol layouts: < 1.15 L -> 3, <= 2.6 L -> 4, <= 4.0 L -> 6, else 8 (estimate)."""
    if pd.isna(disp_L):
        return np.nan
    return 3.0 if disp_L < 1.15 else 4.0 if disp_L <= 2.6 else 6.0 if disp_L <= 4.0 else 8.0


def track_rows(tr, tid, vid):
    out = []
    for f in tr.get("features", []):
        p = f.get("properties", {})
        ph = p.get("phenomenons", {})
        xy = (f.get("geometry") or {}).get("coordinates", [None, None])
        rec = {"vehid": vid, "trip": tid, "time": p.get("time"), "lon": xy[0], "lat": xy[1]}
        for k, v in PHEN.items():
            rec[v] = (ph.get(k) or {}).get("value")
        out.append(rec)
    return pd.DataFrame(out)


def usable_trips(raw):
    """Same feature pipeline and filters as VED. Returns (trips, maf_source)."""
    df = raw.copy()
    for c in ["speed", "gps_speed", "rpm", "maf", "calc_maf", "load", "fuel_rate_obd", "lat", "lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    src = "MAF"
    if df["maf"].notna().mean() < 0.5:
        if not a.allow_calc_maf or df["calc_maf"].notna().mean() < 0.5:
            return pd.DataFrame(), None
        df["maf"], src = df["calc_maf"], "Calculated MAF"
    df["speed"] = df["speed"].fillna(df["gps_speed"])
    df = df.drop(columns=["calc_maf", "gps_speed", "fuel_rate_obd"])   # target from MAF only
    df = mf.add_time_and_trips(df)
    df = mf.add_row_features(df)
    t = mf.aggregate_trips(df)
    t = t[(t["dist_km"] >= 1) & (t["dur_min"] >= 2) & (t["fuel_coverage"] >= 0.8)
          & t["l_per_100km"].between(2, 40)]
    first = raw.dropna(subset=["lat", "lon"]).head(1)
    if len(first):
        t = t.assign(start_lat=float(first["lat"].iloc[0]), start_lon=float(first["lon"].iloc[0]))
    return t, src


# ---------------- page back through tracks ----------------
mf.DATA.mkdir(exist_ok=True)
rows_path = mf.DATA / "envirocar_rows.csv"       # rewritten by each run (first write overwrites)
trip_parts, vehicles = [], {}
n_listed = n_downloaded = n_rows = 0
skip = {"not petrol": 0, "vehicle full": 0, "no usable trip": 0, "download failed": 0}
page = a.start_page
usable_veh = lambda: sum(1 for v in vehicles.values() if v["n_trips"] > 0)
while usable_veh() < a.min_vehicles and n_downloaded < a.max_tracks:
    lst = get(f"{BASE}/tracks", limit=100, page=page)
    if not lst or not lst.get("tracks"):
        print("no more tracks")
        break
    for t in lst["tracks"]:
        if usable_veh() >= a.min_vehicles or n_downloaded >= a.max_tracks:
            break
        n_listed += 1
        sp = (t.get("sensor") or {}).get("properties", {})
        if str(sp.get("fuelType", "")).lower() != "gasoline":
            skip["not petrol"] += 1
            continue
        vid = str(sp.get("id"))
        v = vehicles.setdefault(vid, {
            "vehid": vid, "veh_label": f"{sp.get('manufacturer')} {sp.get('model')} {sp.get('constructionYear')}",
            "disp_L": pd.to_numeric(sp.get("engineDisplacement"), errors="coerce") / 1000,
            "weight": pd.to_numeric(sp.get("weight"), errors="coerce") * KG_TO_LB,
            "n_trips": 0, "n_tracks_downloaded": 0})
        if v["n_trips"] >= a.per_vehicle:
            skip["vehicle full"] += 1
            continue
        tr = get(f"{BASE}/tracks/{t['id']}")
        n_downloaded += 1
        v["n_tracks_downloaded"] += 1
        if not tr:
            skip["download failed"] += 1
            continue
        if a.debug and n_downloaded == 1:
            print(json.dumps(tr, indent=1)[:3000])
        raw = track_rows(tr, t["id"], vid)
        if raw.empty:
            skip["no usable trip"] += 1
            continue
        raw.to_csv(rows_path, mode="a" if n_rows else "w", header=not n_rows, index=False)
        n_rows += len(raw)
        trips, src = usable_trips(raw)
        trips = trips.head(a.per_vehicle - v["n_trips"])
        if trips.empty:
            skip["no usable trip"] += 1
        else:
            trips = trips.assign(track_id=t["id"], maf_source=src)
            trip_parts.append(trips)
            v["n_trips"] += len(trips)
        if n_downloaded % 25 == 0:
            print(f"page {page}: {n_listed} listed, {n_downloaded} downloaded, {n_rows:,} measurements, "
                  f"{sum(len(p) for p in trip_parts)} usable trips from {usable_veh()} vehicles")
        time.sleep(0.3)                        # be polite to a free service
    page += 1

print(f"\nscanned {n_listed} listed tracks (pages {a.start_page}-{page - 1}), downloaded {n_downloaded}, "
      f"{n_rows:,} measurements -> {rows_path}")
print("skipped:", skip)
if not trip_parts:
    raise SystemExit("no usable trips; run with --debug and check the API")
trips = pd.concat(trip_parts, ignore_index=True)
veh = pd.DataFrame(vehicles.values())
veh = veh[veh["n_trips"] > 0]
veh["cylinders"] = veh["disp_L"].map(estimate_cylinders)
trips = trips.merge(veh[["vehid", "veh_label", "disp_L", "weight", "cylinders"]], on="vehid", how="left")
trips["cylinders_estimated"] = 1.0
trips["boost"] = np.nan                        # not in enviroCar's vehicle data

# ---------------- weather per track (Open-Meteo, cached) ----------------
cache_path = mf.DATA / "envirocar_weather_cache.json"
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
for c in ["temp_c", "humidity", "precip_mm", "wind_kmh"]:
    trips[c] = np.nan
if not a.no_weather:
    n_api = 0
    for i, t in trips.iterrows():
        if pd.isna(t.get("start_lat")) or pd.isna(t["start_dt"]):
            continue
        start = pd.Timestamp(t["start_dt"])               # UTC (enviroCar times end in Z)
        key = f"{t['start_lat']:.2f},{t['start_lon']:.2f},{start.date()}"
        if key not in cache:
            w = get(METEO, latitude=round(t["start_lat"], 2), longitude=round(t["start_lon"], 2),
                    start_date=str(start.date()), end_date=str(start.date()), timezone="auto",
                    hourly="temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m")
            cache[key] = w if w and "hourly" in w else None
            n_api += 1
            time.sleep(0.2)
        w = cache[key]
        if not w:
            continue
        tz = w.get("timezone", "UTC")
        local = start.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        h = pd.DataFrame(w["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        m = h[h["time"] == local.floor("h")]
        trips.loc[i, "start_local"] = local
        if len(m):
            trips.loc[i, ["temp_c", "humidity", "precip_mm", "wind_kmh"]] = m[
                ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]].to_numpy()[0]
    cache_path.write_text(json.dumps(cache))
    sd = pd.to_datetime(trips["start_local"])           # calendar features in local time, as for VED
    trips["hour"], trips["weekday"], trips["month"] = sd.dt.hour, sd.dt.day_of_week, sd.dt.month
    trips["is_weekend"] = (sd.dt.day_of_week >= 5).astype(float).where(sd.notna())
    print(f"weather: {n_api} Open-Meteo calls ({len(cache)} cached locations/days), "
          f"matched {trips['temp_c'].notna().mean():.0%} of trips")
trips["cold"] = (trips["temp_c"] < 5).astype(float).where(trips["temp_c"].notna())

mf.PROC.mkdir(parents=True, exist_ok=True)
trips.to_parquet(mf.PROC / "envirocar_trips.parquet", index=False)
veh.to_csv(mf.PROC / "envirocar_vehicles.csv", index=False)
print(f"\nusable petrol trips: {len(trips)} from {trips['vehid'].nunique()} vehicles "
      f"(max {a.per_vehicle} per vehicle; MAF source: {trips['maf_source'].value_counts().to_dict()})")
print(f"  -> {mf.PROC / 'envirocar_trips.parquet'}, {mf.PROC / 'envirocar_vehicles.csv'}")
print(trips[["dist_km", "avg_speed", "l_per_100km", "disp_L", "temp_c"]].describe().round(2).to_string())
print("\nvehicles:")
print(veh[["veh_label", "disp_L", "cylinders", "weight", "n_trips", "n_tracks_downloaded"]]
      .round(2).to_string(index=False))
