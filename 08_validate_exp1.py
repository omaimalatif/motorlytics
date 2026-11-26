"""
External validation on Data/exp1_14drivers_14cars_dailyRoutes.csv (14 drivers, 14 cars,
daily routes; the car models suggest Brazil).

    python 08_validate_exp1.py

What this data allows (checked before writing the script):
  - MAF is missing per CAR, not interleaved between rows: only car11 and car12 log MAF on nearly
    every row (car1 on ~20%). Forward-filling within trips (reported below) does not help, so
    only cars with >= 80% MAF coverage in a trip can be scored.
  - Rows are ~8 s apart (VED ~1 s). The shared feature code ignores gaps > 5 s when integrating,
    so MAX_DT is raised to 12 s here. Acceleration is then taken over ~8 s, which smooths out
    hard accelerations and RPA (behaviour features are biased low compared with VED).
  - Fuel type is mostly empty, and car1 says "Biodiesel_Ethanol". Brazilian pump petrol is E27
    (27% ethanol by volume), which changes the air-fuel ratio. Measured fuel is therefore
    computed twice: pure petrol (AFR 14.7, 737 g/L, as in VED) and E27 (AFR ~13.1, ~753 g/L).
  - No GPS -> no weather; temp_c stays missing. Timestamps are epoch ms (UTC); hour / weekday use
    America/Sao_Paulo time.

Scores behaviour_only (main tier for non-VED logs), full_no_load and full with the saved best course
models (and the best reference models for comparison), same trip filters as the VED build, and writes
Data/processed/results_trip/external_exp1.txt (+ per-vehicle CSV and predictions parquet).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # single-threaded BLAS: ~2x faster here for small matrices (MLP, LWR)

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import motorlytics_features as mf

SRC = mf.DATA / "exp1_14drivers_14cars_dailyRoutes.csv"
REP = mf.PROC / "results_trip"
TZ = "America/Sao_Paulo"
MAX_DT_EXP1 = 12.0                 # s; exp1 logs every ~8 s
FUELS = {                          # name -> (stoichiometric AFR, density g/L)
    "pure petrol (AFR 14.7)": (14.7, 737.0),
    # E27: ethanol mass fraction ~0.28 -> AFR = 14.7*0.72 + 9.0*0.28 ~ 13.1; density 0.27*789 + 0.73*740
    "E27 petrol (AFR 13.1)": (13.1, 753.0),
}
TIERS = ["behaviour_only", "full_no_load", "full"]
out_lines = []


def say(s=""):
    print(s)
    out_lines.append(str(s))


def num(s):
    """'4,49' / '33,30%' / '1,0%' -> float."""
    return pd.to_numeric(s.astype(str).str.strip().str.rstrip("%").str.replace(",", ".", regex=False)
                         .replace({"nan": np.nan, "": np.nan}), errors="coerce")


def estimate_cylinders(disp_L):
    """Same rule as 01b_fetch_envirocar.py: < 1.15 L -> 3, <= 2.6 L -> 4, <= 4.0 L -> 6, else 8."""
    if pd.isna(disp_L):
        return np.nan
    return 3.0 if disp_L < 1.15 else 4.0 if disp_L <= 2.6 else 6.0 if disp_L <= 4.0 else 8.0


# ---------------- load + clean ----------------
raw = pd.read_csv(SRC, low_memory=False)
say(f"{SRC.name}: {len(raw):,} rows, {raw['VEHICLE_ID'].nunique()} vehicles")
say(f"FUEL_TYPE values: {raw['FUEL_TYPE'].value_counts(dropna=False).to_dict()}")
d = pd.DataFrame({
    "vehid": raw["VEHICLE_ID"].astype(str),
    "time": pd.to_datetime(raw["TIMESTAMP"], unit="ms", utc=True).dt.tz_localize(None),   # UTC
    "speed": num(raw["SPEED"]), "rpm": num(raw["ENGINE_RPM"]), "maf": num(raw["MAF"]),
    "load": num(raw["ENGINE_LOAD"]), "oat": num(raw["AMBIENT_AIR_TEMP"]),
    "equiv_ratio": num(raw["EQUIV_RATIO"]),
}).dropna(subset=["time"]).sort_values(["vehid", "time"]).reset_index(drop=True)
veh = (raw.assign(vehid=raw["VEHICLE_ID"].astype(str)).groupby("vehid")
       [["MARK", "MODEL", "CAR_YEAR", "ENGINE_POWER", "FUEL_TYPE"]].first())
veh["disp_L"] = num(veh["ENGINE_POWER"])              # values are litres with a decimal comma
veh["cylinders"] = veh["disp_L"].map(estimate_cylinders)
veh["label"] = (veh["MARK"].fillna("?") + " " + veh["MODEL"].fillna("?") + " "
                + veh["CAR_YEAR"].astype("Int64").astype(str))
er = d["equiv_ratio"].dropna()
say(f"EQUIV_RATIO: n={len(er):,}, mean={er.mean():.3f}, min={er.min():.3f}, max={er.max():.3f} "
    "(constant 1.0 = commanded stoichiometric; it does not reveal the ethanol share)")
dt = d.groupby("vehid")["time"].diff().dt.total_seconds()
say(f"sampling interval: median {dt.median():.1f} s (VED ~1 s); integration gap limit raised to {MAX_DT_EXP1:.0f} s")

# ---------------- interleaving check: MAF coverage before / after a 3 s forward fill in trips ----------------
d["trip"] = (dt.isna() | (dt > 300)).groupby(d["vehid"]).cumsum()          # 5-minute gaps
grp = d.groupby(["vehid", "trip"])
before = d["maf"].notna().groupby(d["vehid"]).mean()
filled = grp["maf"].ffill()
age = d["time"] - d["time"].where(d["maf"].notna()).groupby([d["vehid"], d["trip"]]).ffill()
filled = filled.where(age.dt.total_seconds() <= 3)
after = filled.notna().groupby(d["vehid"]).mean()
say("\nMAF coverage per vehicle, before -> after 3 s forward fill within trips:")
say(pd.DataFrame({"label": veh["label"], "before": before.round(2), "after": after.round(2)}).to_string())
say("-> MAF is missing per car (not interleaved), so no resampling; only trips with >= 80% MAF are scored.")

# ---------------- EPA match on make / model / year (reported only; not a model feature here) ----------------
try:
    epa = pd.read_csv(mf.DATA / "database.csv", usecols=["Year", "Make", "Model"], low_memory=False)
    epa["mk"], epa["md"] = epa["Make"].str.lower(), epa["Model"].str.lower()
    hits = {}
    for v, r in veh.iterrows():
        if pd.isna(r["MARK"]):
            continue
        mk, md = str(r["MARK"]).lower(), str(r["MODEL"]).lower().replace("_", " ").split()[0]
        m = epa[(epa["mk"] == mk) & epa["md"].str.contains(md, regex=False, na=False)
                & epa["Year"].between(r["CAR_YEAR"] - 1, r["CAR_YEAR"] + 1)]
        hits[v] = len(m)
    matched = [v for v, n in hits.items() if n > 0]
    say(f"\nEPA match on make/model/year (+-1): {len(matched)} of {len(hits)} vehicles "
        f"({', '.join(veh.loc[matched, 'label']) or 'none'}); Brazilian models are mostly not sold in the US")
except FileNotFoundError:
    say("\nEPA match skipped: Data/database.csv not found")

# ---------------- trips, scored under each fuel assumption ----------------
meta = json.loads((mf.MODELS / "meta_trip.json").read_text())
# "<tier>" = best COURSE model of the tier (what the app uses); "ref:<tier>" = best reference model
model_keys = {fs: meta[f"best_{fs}"] for fs in TIERS if f"best_{fs}" in meta}
model_keys.update({f"ref:{fs}": meta[f"best_reference_{fs}"] for fs in TIERS if f"best_reference_{fs}" in meta})
bundles = {k: joblib.load(meta["models"][v]["path"]) for k, v in model_keys.items()}
mf.MAX_DT = MAX_DT_EXP1                               # read at call time by add_row_features
base = d.drop(columns=["trip", "equiv_ratio"])
results, per_vehicle, preds = [], None, []
for fuel, (afr, dens) in FUELS.items():
    mf.AFR, mf.G_PER_L = afr, dens
    x = mf.add_time_and_trips(base, gap_s=300.0)     # 5-minute gaps per vehicle
    x = mf.add_row_features(x)
    t = mf.aggregate_trips(x)
    n0 = len(t)
    t = t[(t["dist_km"] >= 1) & (t["dur_min"] >= 2) & (t["fuel_coverage"] >= 0.8)
          & t["l_per_100km"].between(2, 40)].copy()
    if len(t) > 2:                                    # per-vehicle robust z, as in the VED build
        med = t.groupby("vehid")["l_per_100km"].transform("median")
        mad = t.groupby("vehid")["l_per_100km"].transform(lambda s: (s - s.median()).abs().median())
        t = t[~(((t["l_per_100km"] - med) / (1.4826 * mad.replace(0, np.nan))).abs() > 6)]
    sl = pd.to_datetime(t["start_dt"]).dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)
    t["hour"], t["weekday"], t["month"] = sl.dt.hour, sl.dt.day_of_week, sl.dt.month
    t["is_weekend"] = (sl.dt.day_of_week >= 5).astype(float)
    t = t.merge(veh[["disp_L", "cylinders", "label"]], left_on="vehid", right_index=True, how="left")
    t["boost"], t["weight"], t["temp_c"] = np.nan, np.nan, np.nan
    say(f"\n=== {fuel} ===")
    say(f"trips: {n0} after splitting, {len(t)} usable (>= 1 km, >= 2 min, >= 80% MAF, 2-40 L/100 km) "
        f"from {t['vehid'].nunique()} vehicles: {t.groupby('label').size().to_dict()}")
    if t.empty:
        continue
    y = t["l_per_100km"].to_numpy()
    say(f"measured L/100 km: median {np.median(y):.2f}; behaviour features: hard accel/km median "
        f"{t['hard_accel_per_km'].median():.2f}, RPA median {t['rpa'].median():.3f} (VED: 0.34, 0.148)")
    for fs, b in bundles.items():
        p = b["model"].predict(mf.to_matrix(t, b["features"]))
        t[f"pred_{fs}"] = p
        missing = [c for c in b["features"] if c not in t or t[c].notna().mean() < 0.5]
        res = {"fuel": fuel, "tier": fs.split(":")[-1],
               "group": "reference" if fs.startswith("ref:") else "course",
               "model": model_keys[fs].split("/")[1], "n_vehicles": t["vehid"].nunique(), "n_trips": len(t),
               "rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
               "r2": float(r2_score(y, p)) if len(t) > 1 else np.nan,
               "mape": float(np.mean(np.abs((y - p) / y)) * 100), "mean_bias": float(np.mean(p - y))}
        results.append(res)
        say(f"  {res['tier']:15s} {res['group']:9s} {res['model']:12s} n_vehicles={res['n_vehicles']} n_trips={res['n_trips']}  RMSE={res['rmse']:.3f} "
            f"MAE={res['mae']:.3f} R2={res['r2']:.3f} MAPE={res['mape']:.1f}% bias={res['mean_bias']:+.2f}  "
            f"missing features: {missing or 'none'}")
    t["err"] = t["pred_behaviour_only"] - t["l_per_100km"]
    pv = t.groupby("label").agg(disp_L=("disp_L", "first"), trips=("l_per_100km", "size"),
                                measured_median=("l_per_100km", "median"),
                                predicted_median=("pred_behaviour_only", "median"),
                                mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"))
    pv["bias_pct"] = pv["bias"] / pv["measured_median"] * 100
    say(f"\nper vehicle, behaviour_only ({fuel}):\n{pv.round(2).to_string()}")
    per_vehicle = pd.concat([per_vehicle, pv.assign(fuel=fuel)]) if per_vehicle is not None else pv.assign(fuel=fuel)
    preds.append(t.assign(fuel=fuel))

REP.mkdir(parents=True, exist_ok=True)
if results:
    summary = pd.DataFrame(results)
    say("\n=== summary ===")
    say(summary.round(3).to_string(index=False))
    summary.to_csv(REP / "external_exp1_metrics.csv", index=False)
    per_vehicle.round(3).to_csv(REP / "external_exp1_per_vehicle.csv")
    pd.concat(preds, ignore_index=True).drop(columns=["start_local"], errors="ignore").to_parquet(
        REP / "external_exp1_predictions.parquet", index=False)
(REP / "external_exp1.txt").write_text("\n".join(out_lines), encoding="utf-8")
print(f"\nsaved {REP / 'external_exp1.txt'}")
