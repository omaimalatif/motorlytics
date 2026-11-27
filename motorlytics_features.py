"""
Shared feature code for Motorlytics.

The dataset builder, the training scripts and the app all import this file,
so a trip is turned into features the same way everywhere (no train/serve skew).
"""
import re

import numpy as np
import pandas as pd
from pathlib import Path

# anchored to this file and spelled exactly as on disk ("Data", capital D): Linux paths are case-sensitive
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
PROC = DATA / "processed"
MODELS = ROOT / "models"

AFR = 14.7          # stoichiometric air-fuel ratio, gasoline
G_PER_L = 737.0     # gasoline density, g/L
MAX_DT = 5.0        # gaps longer than this (s) are logging gaps, not driving
HARD_ACCEL = 2.5    # m/s^2
HARD_BRAKE = -3.0   # m/s^2
ACC_MIN_DT = 0.9    # s: acceleration is taken over at least ~1 s (VED can log 0.2 s apart, integer km/h)
HIGH_RPM = 2500     # engine speed counted as "high RPM"

# VED: DayNum counts days from this date. CHECK the VED README and change if needed.
VED_BASE_DATE = pd.Timestamp("2017-11-01")
# VED timestamps are UTC (checked against Open-Meteo: OAT lines up with the UTC weather file).
# Weather is joined on UTC; hour / weekday / month features use local Ann Arbor time.
VED_TZ = "America/Detroit"

# ------------------------------------------------------------------------------
# 1. Column standardisation: VED, Torque / Car Scanner exports, enviroCar, allcars
# ------------------------------------------------------------------------------
# canonical name -> regex patterns tried in order on lower-cased column names
PATTERNS = {
    "vehid": [r"^vehid$", r"^vehicle.?id$", r"^deviceid$"],
    "trip": [r"^trip$", r"^trip.?id$", r"^track.?id$"],
    "daynum": [r"^daynum$"],
    "t_ms": [r"^timestamp\s*\(ms\)$"],
    "time": [r"^time$", r"^timestamp$", r"device time", r"gps time", r"^datetime$", r"^date.?time"],
    "lat": [r"^lat(itude)?\b", r"latitude"],
    "lon": [r"^lon(gitude)?\b", r"^lng$", r"longitude"],
    "speed": [r"^vehicle speed", r"speed \(obd\)", r"^speed\b", r"obd.*speed", r"gps.?speed"],
    "maf": [r"^maf\b", r"mass air flow", r"air flow rate"],
    "rpm": [r"^engine rpm", r"^rpm\b", r"engine speed", r"^engine_rpm$"],
    "load": [r"^absolute load", r"^engine.?load", r"^eload$", r"^load\b"],
    "oat": [r"^oat\b", r"ambient air temp", r"outside"],
    "fuel_rate_obd": [r"^fuel rate", r"^consumption\b"],
}


def standardize(raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Rename known columns to canonical names and drop the rest."""
    lower = {c: c.strip().lower() for c in raw.columns}
    used, mapping = set(), {}
    for canon, pats in PATTERNS.items():
        for p in pats:
            hit = next((c for c, lc in lower.items() if c not in used and re.search(p, lc)), None)
            if hit is not None:
                mapping[hit] = canon
                used.add(hit)
                break
    if verbose:
        print("column mapping:", {v: k for k, v in mapping.items()})
    df = raw[list(mapping)].rename(columns=mapping)
    for c in ["speed", "maf", "rpm", "load", "oat", "fuel_rate_obd", "lat", "lon"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_time_and_trips(df: pd.DataFrame, gap_s: float = 300.0) -> pd.DataFrame:
    """Give every row `trip_key`, `t` (seconds from trip start) and `start_dt`.

    VED: trip id + DayNum + Timestamp(ms).
    Phone-app logs: a time column; trips are split where the log pauses > gap_s.
    """
    df = df.copy()
    if "vehid" not in df:
        df["vehid"] = "user"
    if "t_ms" in df and "trip" in df:                          # VED
        df["trip_key"] = df["vehid"].astype(str) + "_" + df["trip"].astype(str)
        df["t"] = pd.to_numeric(df["t_ms"], errors="coerce") / 1000.0
        if "daynum" in df:
            day = df.groupby("trip_key")["daynum"].transform("min")
            df["start_dt"] = VED_BASE_DATE + pd.to_timedelta(day - 1, unit="D")   # UTC
            df["start_local"] = (df["start_dt"].dt.tz_localize("UTC")
                                 .dt.tz_convert(VED_TZ).dt.tz_localize(None))
        else:
            df["start_dt"] = pd.NaT
            df["start_local"] = pd.NaT
        return df
    if "time" not in df:
        raise ValueError("need a time column (or VED Timestamp(ms) + Trip)")
    ts = df["time"]
    if pd.api.types.is_numeric_dtype(ts):                   # epoch seconds or ms
        unit = "ms" if ts.median() > 1e11 else "s"
        ts = pd.to_datetime(ts, unit=unit, errors="coerce")
    else:
        ts = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_localize(None)
    df["_ts"] = ts
    df = df.dropna(subset=["_ts"]).sort_values(["vehid", "_ts"])
    if "trip" not in df:
        gap = df.groupby("vehid")["_ts"].diff().dt.total_seconds()
        df["trip"] = (gap.isna() | (gap > gap_s)).groupby(df["vehid"]).cumsum()
    df["trip_key"] = df["vehid"].astype(str) + "_" + df["trip"].astype(str)
    start = df.groupby("trip_key")["_ts"].transform("min")
    df["t"] = (df["_ts"] - start).dt.total_seconds()
    df["start_dt"] = start
    df["start_local"] = start          # phone-app logs: device time is treated as local time
    return df.drop(columns="_ts")


# ------------------------------------------------------------------------------
# 2. Row-level features (computed inside each trip, never across trips)
# ------------------------------------------------------------------------------
def _accel_over_1s(df: pd.DataFrame, g) -> np.ndarray:
    """Acceleration (m/s^2) from smoothed speed, over a span of at least ACC_MIN_DT seconds.

    Speed is smoothed with a centred 3-sample mean inside each trip. Each row is compared with
    the latest earlier sample of the same trip that is >= ACC_MIN_DT s back; spans longer than
    MAX_DT (logging gaps) give NaN. df must be sorted by trip_key, t.
    """
    t = df["t"].to_numpy(dtype=float)
    if len(t) == 0 or np.isnan(t).all():                       # nothing usable left (e.g. all glitches)
        return np.full(len(t), np.nan)
    vs = (g["speed"].rolling(3, center=True, min_periods=1).mean()
          .reset_index(level=0, drop=True).to_numpy() / 3.6)
    code = pd.factorize(df["trip_key"], sort=False)[0].astype(float)
    span = np.nanmax(t) - np.nanmin(t) + 10 * MAX_DT if len(t) else 1.0
    key = code * span + (t - np.nanmin(t))                      # increasing within and across trips
    j = np.searchsorted(key, key - ACC_MIN_DT, side="right") - 1
    valid = (j >= 0) & ~np.isnan(t)
    j = np.clip(j, 0, None)
    same = valid & (code[j] == code)
    dtj = t - t[j]
    ok = same & (dtj >= ACC_MIN_DT) & (dtj <= MAX_DT)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, (vs - vs[j]) / dtj, np.nan)


MAX_VALID_SPEED = 250.0    # km/h; 255 is the OBD "no data" byte
SPIKE_ACCEL = 4.0          # m/s^2; a single reading needing more than this into AND out of it is a glitch
SPIKE_MIN_JUMP = 10.0      # km/h; and the jump must be at least this big (whole-km/h rounding at 10 Hz is not a glitch)


def clean_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Remove speed glitches before any feature is computed (used by add_row_features, so by training, the
    app, 05_predict_trip.py and the enviroCar scripts):
      - speed >= MAX_VALID_SPEED (e.g. 255 = OBD 'no data'),
      - single-sample spikes: |acceleration| > SPIKE_ACCEL m/s^2 both into and out of one reading, in
        opposite directions (up then straight down, or the reverse), within a trip;
      - one-sided spikes: |acceleration| > SPIKE_ACCEL into (or out of) a reading whose other neighbour is a
        removed glitch or the trip's start / end (e.g. the 223 in 0 -> 223 -> 255 -> 255).
    Each jump must also be at least SPIKE_MIN_JUMP km/h, so whole-km/h rounding steps between readings 0.1 s apart
    (2 km/h in 0.1 s is 5.6 m/s^2) are not treated as glitches.
    Bad readings are set missing and filled by linear interpolation over time inside the trip; a bad reading at
    the start or end of a trip (nothing to interpolate from) is dropped. Needs trip_key, t (s) and speed.
    The number of readings fixed / dropped is stored in df.attrs["speed_cleaning"]."""
    d = df.sort_values(["trip_key", "t"]).reset_index(drop=True)
    if "speed" not in d or d.empty:
        d.attrs["speed_cleaning"] = {"fixed": 0, "dropped": 0}
        return d
    v = pd.to_numeric(d["speed"], errors="coerce").to_numpy(float).copy()
    t = pd.to_numeric(d["t"], errors="coerce").to_numpy(float)
    g = pd.factorize(d["trip_key"])[0]
    bad = v >= MAX_VALID_SPEED
    v[bad] = np.nan
    same_prev = np.r_[False, g[1:] == g[:-1]]
    same_next = np.r_[g[:-1] == g[1:], False]
    dtp, dtn = np.r_[np.nan, np.diff(t)], np.r_[np.diff(t), np.nan]
    jump_in = v - np.r_[np.nan, v[:-1]]
    jump_out = np.r_[v[1:], np.nan] - v
    with np.errstate(invalid="ignore", divide="ignore"):
        a_in = np.where(same_prev & (dtp > 0), jump_in / 3.6 / dtp, np.nan)
        a_out = np.where(same_next & (dtn > 0), jump_out / 3.6 / dtn, np.nan)
    big_in = (np.abs(a_in) > SPIKE_ACCEL) & (np.abs(jump_in) >= SPIKE_MIN_JUMP)
    big_out = (np.abs(a_out) > SPIKE_ACCEL) & (np.abs(jump_out) >= SPIKE_MIN_JUMP)
    spike = big_in & big_out & (np.sign(a_in) != np.sign(a_out))
    # one-sided: an impossible jump into / out of a reading whose other neighbour is missing (removed glitch or
    # the trip's start / end), e.g. 0 -> 223 -> 255 -> 255: the 223 is part of the glitch burst
    spike |= big_in & np.isnan(a_out)
    spike |= big_out & np.isnan(a_in)
    bad |= spike
    v[spike] = np.nan
    d["speed"] = v
    drop = np.zeros(len(d), bool)
    if bad.any():
        for key in np.unique(g[bad]):                      # interpolate only trips that had a glitch
            idx = np.where(g == key)[0]
            s = pd.Series(v[idx], index=t[idx])
            filled = s.interpolate(method="index", limit_area="inside").to_numpy()
            d.loc[idx, "speed"] = filled
            drop[idx] = bad[idx] & np.isnan(filled)        # glitch at the trip's start / end
    d.attrs["speed_cleaning"] = {"fixed": int((bad & ~drop).sum()), "dropped": int(drop.sum())}
    attrs = d.attrs
    d = d[~drop].reset_index(drop=True)
    d.attrs = attrs
    return d


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_speed(df)                                  # speed glitches out before any feature is computed
    df = df.sort_values(["trip_key", "t"]).reset_index(drop=True)
    g = df.groupby("trip_key", sort=False)
    df["dt"] = g["t"].diff()
    ok = (df["dt"] > 0) & (df["dt"] <= MAX_DT)
    df["dt_int"] = df["dt"].where(ok, 0.0)                      # seconds used for integration
    df["accel"] = _accel_over_1s(df, g)
    df["accel"] = df["accel"].clip(-8, 8)                      # m/s^2, sensor spikes clipped
    v = df["speed"] / 3.6
    df["speed_sq"] = df["speed"] ** 2
    df["power_proxy"] = v * df["accel"]                         # ~ specific power, W/kg
    df["is_idle"] = (df["speed"] <= 1).astype(np.float32)
    for w in (5, 15):
        r = g["speed"].rolling(w, min_periods=1)
        df[f"speed_roll{w}"] = r.mean().reset_index(level=0, drop=True)
        df[f"speed_std{w}"] = r.std().reset_index(level=0, drop=True)
    df["accel_std15"] = (g["accel"].rolling(15, min_periods=2).std()
                         .reset_index(level=0, drop=True))
    df["speed_lag1"] = g["speed"].shift(1)
    df["speed_lag3"] = g["speed"].shift(3)
    if "rpm" in df:
        df["rpm_per_kmh"] = np.where(df["speed"] > 5, df["rpm"] / df["speed"].clip(lower=1), np.nan)
        df["rpm_change"] = g["rpm"].diff()
        df["rpm_roll5"] = (g["rpm"].rolling(5, min_periods=1).mean()
                           .reset_index(level=0, drop=True))
    if "maf" in df and df["maf"].notna().any():
        df["fuel_Lph"] = df["maf"] * 3600 / (AFR * G_PER_L)    # target (row level)
    elif "fuel_rate_obd" in df:
        df["fuel_Lph"] = df["fuel_rate_obd"]
    return df


ROW_FEATURES = ["speed", "speed_sq", "accel", "power_proxy", "is_idle",
                "speed_roll5", "speed_std5", "speed_roll15", "speed_std15",
                "accel_std15", "speed_lag1", "speed_lag3",
                "rpm", "rpm_per_kmh", "rpm_change", "rpm_roll5", "load"]


# ------------------------------------------------------------------------------
# 3. Trip-level features (one row per trip; target = L/100 km)
# ------------------------------------------------------------------------------
def aggregate_trips(df: pd.DataFrame) -> pd.DataFrame:
    d = df
    dt = d["dt_int"]
    g = d.groupby("trip_key", sort=False)
    prev_speed = g["speed"].shift(1)
    prev_acc = g["accel"].shift(1)
    v = d["speed"] / 3.6
    h = pd.DataFrame({"trip_key": d["trip_key"]})
    h["dist_m"] = v * dt
    h["dur_s"] = dt
    h["idle_s"] = (d["speed"] <= 1) * dt
    h["rpa_num"] = v * d["accel"].clip(lower=0).fillna(0) * dt   # relative positive acceleration
    h["stops"] = ((d["speed"] <= 1) & (prev_speed > 1)).astype(int)
    h["hard_acc"] = ((d["accel"] > HARD_ACCEL) & ~(prev_acc > HARD_ACCEL)).astype(int)
    h["hard_brk"] = ((d["accel"] < HARD_BRAKE) & ~(prev_acc < HARD_BRAKE)).astype(int)
    for name, lo, hi in [("0_30", 1, 30), ("30_60", 30, 60), ("60_90", 60, 90), ("gt90", 90, 999)]:
        h[f"t_{name}"] = ((d["speed"] > lo) & (d["speed"] <= hi)) * dt
    if "fuel_Lph" in d:
        h["fuel_L"] = d["fuel_Lph"].fillna(0) * dt / 3600
        h["fuel_ok"] = d["fuel_Lph"].notna().astype(int)
    if "rpm" in d:
        h["rpm_t"] = d["rpm"].fillna(0) * dt
        h["rpm_hi_s"] = (d["rpm"] > HIGH_RPM) * dt
        h["rpm_t_n"] = d["rpm"].notna() * dt
    if "load" in d:
        h["load_t"] = d["load"].fillna(0) * dt
        h["load_t_n"] = d["load"].notna() * dt
    if "oat" in d:
        h["oat"] = d["oat"]
    s = h.groupby("trip_key", sort=False).sum(numeric_only=True)

    first = g.first()
    out = pd.DataFrame(index=s.index)
    out["vehid"] = first["vehid"]
    out["start_dt"] = first["start_dt"] if "start_dt" in first else pd.NaT
    out["start_local"] = first["start_local"] if "start_local" in first else out["start_dt"]
    out["n_rows"] = g.size()
    out["dist_km"] = s["dist_m"] / 1000
    out["dur_min"] = s["dur_s"] / 60
    km = out["dist_km"].where(out["dist_km"] > 0)
    out["avg_speed"] = out["dist_km"] / (out["dur_min"] / 60)
    moving_h = (s["dur_s"] - s["idle_s"]) / 3600
    out["avg_moving_speed"] = out["dist_km"] / moving_h.where(moving_h > 0)
    out["max_speed"] = g["speed"].max()
    out["speed_std"] = g["speed"].std()
    out["accel_std"] = g["accel"].std()
    out["mean_pos_accel"] = d["accel"].where(d["accel"] > 0.1).groupby(d["trip_key"]).mean()
    out["idle_frac"] = s["idle_s"] / s["dur_s"]
    out["stops_per_km"] = s["stops"] / km
    out["hard_accel_per_km"] = s["hard_acc"] / km
    out["hard_brake_per_km"] = s["hard_brk"] / km
    out["rpa"] = s["rpa_num"] / s["dist_m"]
    for name in ["0_30", "30_60", "60_90", "gt90"]:
        out[f"frac_speed_{name}"] = s[f"t_{name}"] / s["dur_s"]
    if "rpm_t" in s:
        out["mean_rpm"] = s["rpm_t"] / s["rpm_t_n"].where(s["rpm_t_n"] > 0)
        out["rpm_p90"] = g["rpm"].quantile(0.9)
        out["frac_rpm_gt2500"] = s["rpm_hi_s"] / s["dur_s"]
        mv = d["rpm_per_kmh"] if "rpm_per_kmh" in d else d["rpm"] / d["speed"].where(d["speed"] > 5)
        out["mean_rpm_per_kmh"] = mv.groupby(d["trip_key"]).mean()
    if "load_t" in s:
        out["mean_load"] = s["load_t"] / s["load_t_n"].where(s["load_t_n"] > 0)
    if "oat" in d:
        out["oat_mean"] = g["oat"].mean()
    if "fuel_L" in s:
        out["fuel_L"] = s["fuel_L"]
        out["fuel_coverage"] = s["fuel_ok"] / out["n_rows"]
        out["l_per_100km"] = s["fuel_L"] / km * 100           # TARGET
    sd = pd.to_datetime(out["start_local"])                  # local time for calendar features
    out["hour"] = sd.dt.hour
    out["weekday"] = sd.dt.weekday
    out["is_weekend"] = (sd.dt.weekday >= 5).astype(float).where(sd.notna())
    out["month"] = sd.dt.month
    out["short_trip"] = (out["dur_min"] < 10).astype(float)
    return out.reset_index()


TRIP_BEHAVIOUR = ["avg_speed", "avg_moving_speed", "max_speed", "speed_std", "accel_std",
                  "mean_pos_accel", "idle_frac", "stops_per_km", "hard_accel_per_km",
                  "hard_brake_per_km", "rpa", "frac_speed_0_30", "frac_speed_30_60",
                  "frac_speed_60_90", "frac_speed_gt90", "dist_km", "dur_min", "short_trip"]
TRIP_ENGINE = ["mean_rpm", "rpm_p90", "frac_rpm_gt2500", "mean_rpm_per_kmh", "mean_load"]
TRIP_CONTEXT = ["temp_c", "humidity", "precip_mm", "wind_kmh", "oat_mean",
                "hour", "is_weekend", "month", "grade_abs_mean", "elev_gain_per_km"]
VEHICLE = ["disp_L", "cylinders", "boost", "weight"]          # cls_* dummies not used: ~99% "unknown" in VED

# ------------------------------------------------------------------------------
# 4. Vehicle specs (VED static file; also used for the app's car picker)
# ------------------------------------------------------------------------------
def estimate_cylinders(disp_L):
    """Typical petrol layouts: < 1.15 L -> 3, <= 2.6 L -> 4, <= 4.0 L -> 6, else 8 (an estimate)."""
    if disp_L is None or pd.isna(disp_L):
        return np.nan
    return 3.0 if disp_L < 1.15 else 4.0 if disp_L <= 2.6 else 6.0 if disp_L <= 4.0 else 8.0


def parse_static(static: pd.DataFrame) -> pd.DataFrame:
    s = static.copy()
    s.columns = [c.strip() for c in s.columns]
    eng = s["Engine Configuration & Displacement"].astype(str)
    out = pd.DataFrame({"vehid": s["VehId"]})
    out["veh_type"] = s["Vehicle Type"].astype(str).str.strip()
    out["veh_class"] = s["Vehicle Class"].astype(str).str.strip().replace({"NO DATA": "unknown"})
    # VED mostly writes "4-FI 2.0L", but some rows use "I4 2.4L", "V6 3.5L", "H-4 2.5L",
    # "2.3L Gasoline I4" or "3.0L 6cyl", so try those forms too (never a leading "2.3L")
    cyl = eng.str.extract(r"^\s*(\d{1,2})\s*-")[0]
    cyl = cyl.fillna(eng.str.extract(r"(?i)\b[IVHW]\s*-?\s*(\d{1,2})\b")[0])
    cyl = cyl.fillna(eng.str.extract(r"(?i)(\d{1,2})\s*-?\s*cyl")[0])
    out["cylinders"] = pd.to_numeric(cyl, errors="coerce")
    out["disp_L"] = pd.to_numeric(eng.str.extract(r"(\d+(?:\.\d+)?)\s*L")[0], errors="coerce")
    out["boost"] = eng.str.contains(r"T/C|S/C", regex=True).astype(float)
    out["is_diesel"] = eng.str.contains("DIESEL", case=False)
    w = s["Generalized_Weight"].astype(str).str.replace(",", "").str.extract(r"(\d+(?:\.\d+)?)")[0]
    out["weight"] = pd.to_numeric(w, errors="coerce")
    return out


def add_class_dummies(df: pd.DataFrame, classes=None) -> pd.DataFrame:
    classes = classes or sorted(df["veh_class"].fillna("unknown").unique())
    for c in classes:
        df[f"cls_{c}"] = (df["veh_class"].fillna("unknown") == c).astype(np.float32)
    return df


# ------------------------------------------------------------------------------
# 5. Weather join (hourly, on trip start)
# ------------------------------------------------------------------------------
def load_weather(path) -> pd.DataFrame:
    w = pd.read_csv(path)
    w["hour_key"] = pd.to_datetime(w["time"]).dt.floor("h")
    w = w.rename(columns={"temperature_2m": "temp_c", "relative_humidity_2m": "humidity",
                          "precipitation": "precip_mm", "wind_speed_10m": "wind_kmh"})
    return w[["hour_key", "temp_c", "humidity", "precip_mm", "wind_kmh"]].drop_duplicates("hour_key")


def join_weather(trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()
    trips["hour_key"] = pd.to_datetime(trips["start_dt"]).dt.floor("h")
    trips = trips.merge(weather, on="hour_key", how="left")
    trips["cold"] = (trips["temp_c"] < 5).astype(float).where(trips["temp_c"].notna())
    return trips.drop(columns="hour_key")


# ------------------------------------------------------------------------------
# 5b. EPA fuel-economy ratings (Data/database.csv), matched on engine specs
# ------------------------------------------------------------------------------
MPG_TO_L100 = 235.215                      # L/100 km = 235.215 / US mpg
EPA_FEATURES = ["epa_city_l100", "epa_hwy_l100", "epa_comb_l100"]
EPA_COLS = ["Year", "Make", "Model", "Class", "Engine Cylinders", "Engine Displacement",
            "Turbocharger", "Supercharger", "Fuel Type 1", "Fuel Type 2", "Alternative Fuel/Technology",
            "Electric Motor", "City MPG (FT1)", "Highway MPG (FT1)", "Combined MPG (FT1)"]


def _flag(s: pd.Series) -> pd.Series:
    """EPA turbo/supercharger columns: a marker such as 'T' / 'S' means yes, empty means no."""
    t = s.astype(str).str.strip().str.lower()
    return s.notna() & ~t.isin(["", "nan", "0", "false", "n", "no"])


def epa_coarse_class(cls: pd.Series) -> pd.Series:
    """EPA size class -> Car / SUV / Truck-Van (VED's static file uses Car / SUV)."""
    c = cls.astype(str).str.lower()
    out = pd.Series("Car", index=cls.index, dtype=object)
    out[c.str.contains("sport utility")] = "SUV"
    out[c.str.contains("pickup|van|special purpose")] = "Truck-Van"
    return out.where(cls.notna(), None)


def load_epa(path, years=(2005, 2018)) -> pd.DataFrame:
    """Gasoline, non-hybrid, non-electric EPA rows within the model-year window, with L/100 km."""
    e = pd.read_csv(path, usecols=lambda c: c in EPA_COLS, low_memory=False)
    e = e[e["Year"].between(*years)]
    gas = e["Fuel Type 1"].astype(str).str.contains("Gasoline", case=False)
    alt = e.get("Alternative Fuel/Technology", pd.Series(np.nan, index=e.index)).astype(str)
    hybrid = alt.str.contains("hybrid|electric|plug", case=False) | e.get(
        "Electric Motor", pd.Series(np.nan, index=e.index)).notna()
    e = e[gas & ~hybrid].copy()
    e["disp_L"] = pd.to_numeric(e["Engine Displacement"], errors="coerce").round(1)
    e["cylinders"] = pd.to_numeric(e["Engine Cylinders"], errors="coerce")
    e["boost"] = (_flag(e["Turbocharger"]) | _flag(e["Supercharger"])).astype(float)
    e["epa_class"] = epa_coarse_class(e["Class"])
    for src, dst in [("City MPG (FT1)", "epa_city_l100"), ("Highway MPG (FT1)", "epa_hwy_l100"),
                     ("Combined MPG (FT1)", "epa_comb_l100")]:
        mpg = pd.to_numeric(e[src], errors="coerce")
        e[dst] = MPG_TO_L100 / mpg.where(mpg > 0)
    return e.dropna(subset=["disp_L", "cylinders", "epa_comb_l100"]).reset_index(drop=True)


def match_epa(epa: pd.DataFrame, disp_L, cylinders, boost, veh_class=None, min_n: int = 3) -> dict:
    """Median EPA ratings for one vehicle, trying strict to loose matches:
      1. same displacement (0.1 L), cylinders, turbo/supercharged, and class (if known)
      2. same without class
      3. displacement within +-0.1 L, same cylinders and boost
    A level is used once it has at least `min_n` EPA rows (otherwise the next level is tried;
    the loosest level is accepted with >= 1 row). Returns {} when nothing matches."""
    if pd.isna(disp_L) or pd.isna(cylinders):
        return {}
    d, cyl, b = round(float(disp_L), 1), float(cylinders), float(boost or 0)
    base = (epa["cylinders"] == cyl) & (epa["boost"] == b)
    known_cls = veh_class if veh_class in ("Car", "SUV", "Truck-Van") else None
    levels = []
    if known_cls:
        levels.append((1, base & (epa["disp_L"] == d) & (epa["epa_class"] == known_cls)))
    levels.append((2, base & (epa["disp_L"] == d)))
    levels.append((3, base & (epa["disp_L"] - d).abs().le(0.1 + 1e-9)))
    for i, (lvl, m) in enumerate(levels):
        c = epa[m]
        if len(c) >= min_n or (i == len(levels) - 1 and len(c) >= 1):
            q = c["epa_comb_l100"].quantile([0.25, 0.75])
            return {"epa_city_l100": float(c["epa_city_l100"].median()),
                    "epa_hwy_l100": float(c["epa_hwy_l100"].median()),
                    "epa_comb_l100": float(c["epa_comb_l100"].median()),
                    "epa_comb_l100_iqr": float(q.iloc[1] - q.iloc[0]),
                    "epa_class": c["epa_class"].mode().iloc[0] if c["epa_class"].notna().any() else None,
                    "epa_n_candidates": int(len(c)), "epa_match_level": lvl}
    return {}


# ------------------------------------------------------------------------------
# 6. Prediction + recommendations (used by predict_trip.py and the app)
# ------------------------------------------------------------------------------
TIPS = {
    "stops_per_km": "You stop often for this kind of road. Leave more gap and roll towards lights so you can keep moving.",
    "idle_frac": "Idling is a large share of this trip. Switch the engine off at stops longer than about a minute.",
    "hard_accel_per_km": "Accelerate more gently. Hard starts from lights burn the most extra fuel.",
    "hard_brake_per_km": "Look further ahead and brake earlier and softer; every hard stop wastes the speed you paid fuel for.",
    "rpa": "Keep a steadier pace. Repeated speeding up and slowing down costs fuel.",
    "speed_std": "Hold a more constant speed; use cruise control on open roads.",
    "frac_rpm_gt2500": "Shift up earlier and keep RPM under about 2,500.",
    "rpm_p90": "Your engine often runs at high RPM. Shift up sooner.",
    "frac_speed_gt90": "Above 90 km/h air drag rises sharply. Easing off a little saves fuel on the highway.",
}


def to_matrix(df: pd.DataFrame, features) -> np.ndarray:
    X = pd.DataFrame(index=df.index)
    for f in features:
        X[f] = df[f] if f in df else (0.0 if f.startswith("cls_") else np.nan)
    return X.to_numpy(dtype=np.float32)


ROAD_TYPES = ["city", "mixed", "highway"]

# minimum value a trip must exceed before a tip is considered (eco p75 can be 0 for these);
# features not listed have floor 0
TIP_FLOORS = {"hard_brake_per_km": 0.2, "hard_accel_per_km": 0.2, "frac_rpm_gt2500": 0.02}


def road_type(avg_speed) -> pd.Series:
    """city (< 30 km/h), mixed (30-60), highway (> 60), from trip average speed."""
    s = pd.Series(avg_speed, dtype=float)
    out = pd.cut(s, [-np.inf, 30, 60, np.inf], labels=ROAD_TYPES, right=False).astype(object)
    return out.where(s.notna(), None)


def pick_eco_profile(eco_profile: dict, trip: pd.Series) -> tuple:
    """eco_profile is either one flat profile or {road_type: profile}. Returns (profile, road_type)."""
    rt = road_type([trip.get("avg_speed", np.nan)]).iloc[0]
    if any(k in eco_profile for k in ROAD_TYPES + ["all"]):
        prof = eco_profile.get(rt) or eco_profile.get("all") or next(iter(eco_profile.values()))
        return prof, rt
    return eco_profile, rt


def recommend(bundle: dict, trip: pd.Series, eco_profile: dict, petrol_pkr_per_l=None,
              eco_p75: dict = None) -> dict:
    """What-if analysis: set each behaviour feature to the eco-driver median
    (only if the user is worse) and see how much predicted L/100 km drops.
    eco_profile / eco_p75 may be dicts per road type; the trip is then compared with eco
    drivers on the same road type (city / mixed / highway).
    A feature is flagged only if the trip is above the eco 75th percentile (fallback:
    eco median x 1.1); the what-if target stays the eco median.
    Features are correlated, so per-tip savings are estimates, not additive."""
    eco_profile, rt = pick_eco_profile(eco_profile, trip)
    p75 = pick_eco_profile(eco_p75, trip)[0] if eco_p75 else {}
    model, feats = bundle["model"], bundle["features"]
    row = trip.to_frame().T
    base = float(model.predict(to_matrix(row, feats))[0])
    tips, better = [], row.copy()
    for f, text in TIPS.items():
        if f not in feats or f not in eco_profile or pd.isna(row[f].iloc[0]):
            continue
        thr = p75.get(f)
        if thr is None or pd.isna(thr):
            thr = eco_profile[f] * 1.1
        thr = max(thr, TIP_FLOORS.get(f, 0.0))
        if row[f].iloc[0] > thr:
            r2 = row.copy()
            r2[f] = eco_profile[f]
            better[f] = eco_profile[f]
            saving = base - float(model.predict(to_matrix(r2, feats))[0])
            if saving > 0.05:
                tips.append({"feature": f, "your_value": round(float(row[f].iloc[0]), 3),
                             "eco_value": round(float(eco_profile[f]), 3),
                             "saving_l_per_100km": round(saving, 2), "tip": text})
    eco_pred = float(model.predict(to_matrix(better, feats))[0])
    tips.sort(key=lambda t: -t["saving_l_per_100km"])
    res = {"road_type": rt, "pred_l_per_100km": round(base, 2), "eco_l_per_100km": round(eco_pred, 2),
           "possible_saving_pct": round(max(0.0, (base - eco_pred) / base * 100), 1), "tips": tips}
    if petrol_pkr_per_l:
        dist = float(trip.get("dist_km", np.nan))
        res["trip_cost_pkr"] = round(base / 100 * dist * petrol_pkr_per_l, 0)
        res["cost_per_100km_pkr"] = round(base * petrol_pkr_per_l, 0)
        res["eco_saving_pkr_per_100km"] = round((base - eco_pred) * petrol_pkr_per_l, 0)
    return res


# ------------------------------------------------------------------------------
# 6b. Locally Weighted Regression (course model; lives here so saved models unpickle anywhere)
# ------------------------------------------------------------------------------
from sklearn.base import BaseEstimator, RegressorMixin


class LocallyWeightedRegression(BaseEstimator, RegressorMixin):
    """Locally weighted linear regression with a Gaussian kernel (sklearn-style).

    For each query x: weights w_i = exp(-||x - x_i||^2 / (2 * bandwidth^2)) over the training points
    (features median-imputed and standardised), then a weighted least-squares line
    beta = (X'WX + ridge*I)^-1 X'Wy is fitted and evaluated at x. The weights are normalised per
    query, so `ridge` is a small relative penalty that keeps near-singular cases solvable (the
    intercept is not penalised). The training set is subsampled to `max_train` points for speed.
    """

    def __init__(self, bandwidth=3.0, ridge=1e-2, max_train=5000, batch=64, random_state=42):
        self.bandwidth = bandwidth
        self.ridge = ridge
        self.max_train = max_train
        self.batch = batch
        self.random_state = random_state

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isnan(X), self.med_, X)
        return (X - self.mu_) / self.sd_

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.med_ = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isnan(self.med_), 0.0, self.med_)
        Xi = np.where(np.isnan(X), self.med_, X)
        self.mu_, self.sd_ = Xi.mean(axis=0), Xi.std(axis=0)
        self.sd_ = np.where(self.sd_ > 0, self.sd_, 1.0)
        Z = (Xi - self.mu_) / self.sd_
        if self.max_train and len(Z) > self.max_train:
            idx = np.random.default_rng(self.random_state).choice(len(Z), self.max_train, replace=False)
            Z, y = Z[idx], y[idx]
        self.Xa_ = np.hstack([np.ones((len(Z), 1)), Z])       # with intercept column
        self.y_ = y
        self.sq_ = (Z ** 2).sum(axis=1)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        Q = self._prep(X)
        Qa = np.hstack([np.ones((len(Q), 1)), Q])
        P = np.eye(self.Xa_.shape[1]) * self.ridge
        P[0, 0] = 0.0
        out = np.empty(len(Q))
        Z = self.Xa_[:, 1:]
        for s in range(0, len(Q), self.batch):
            q = Q[s:s + self.batch]
            d2 = np.maximum((q ** 2).sum(axis=1)[:, None] + self.sq_[None, :] - 2 * q @ Z.T, 0.0)
            d2 -= d2.min(axis=1, keepdims=True)                  # avoid underflow; scale-free after norm
            W = np.exp(-d2 / (2.0 * self.bandwidth ** 2))
            W /= W.sum(axis=1, keepdims=True)
            A = np.matmul(self.Xa_.T[None] * W[:, None, :], self.Xa_) + P   # batched X'WX (BLAS)
            b = (W * self.y_) @ self.Xa_
            beta = np.linalg.solve(A, b[..., None])[..., 0]
            out[s:s + self.batch] = (Qa[s:s + self.batch] * beta).sum(axis=1)
        return out


class ClipInputs(BaseEstimator, RegressorMixin):
    """Wraps any regressor: learns each feature's q-th and (100-q)-th percentile on the training data
    and clips inputs to that range before every predict. Limits extrapolation on inputs outside the
    training range (e.g. other countries' cars or loggers); NaN stays NaN for the model's own imputer.
    """

    def __init__(self, estimator=None, q=1.0):
        self.estimator = estimator
        self.q = q

    def fit(self, X, y):
        from sklearn.base import clone
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            lo, hi = np.nanpercentile(X, [self.q, 100 - self.q], axis=0) if len(X) else (None, None)
        self.clip_lo_ = np.where(np.isnan(lo), -np.inf, lo)
        self.clip_hi_ = np.where(np.isnan(hi), np.inf, hi)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.estimator_.predict(np.clip(X, self.clip_lo_, self.clip_hi_))


# ------------------------------------------------------------------------------
# 7. Per-car calibration store (models/calibration.json)
# ------------------------------------------------------------------------------
# External tests showed most of the error on unseen cars is a per-car offset, so each car keeps a
# factor k = median(measured / predicted) per model tier, from its most recent calibration entries:
#   "maf"    one trip with MAF:   ratio = measured L/100 km / predicted L/100 km
#   "fillup" a fill-up over trips: ratio = litres filled / sum(predicted litres over those trips)
CALIB_PATH = MODELS / "calibration.json"
CALIB_MAX_TRIPS = 10


def load_calibration(path=None) -> dict:
    p = Path(path or CALIB_PATH)
    if p.exists():
        import json
        return json.loads(p.read_text())
    return {"cars": {}}


def save_calibration(store: dict, path=None) -> None:
    import json
    p = Path(path or CALIB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, indent=2, default=str))


def add_calibration(store: dict, car_id: str, ratios: dict, n_trips: int, kind: str,
                    when: str, detail: dict = None) -> None:
    """Append one calibration entry (ratios per tier), then keep only the entries covering the
    CALIB_MAX_TRIPS most recent trips (the newest entry is always kept)."""
    car = store.setdefault("cars", {}).setdefault(str(car_id), {"entries": []})
    car["entries"].append({"kind": kind, "when": str(when), "trips": int(n_trips),
                           "ratios": {k: float(v) for k, v in ratios.items() if v and np.isfinite(v)},
                           **(detail or {})})
    car["entries"].sort(key=lambda e: e["when"])
    kept, total = [], 0
    for e in reversed(car["entries"]):
        if kept and total + e["trips"] > CALIB_MAX_TRIPS:
            break
        kept.append(e)
        total += e["trips"]
    car["entries"] = list(reversed(kept))
    car["updated"] = pd.Timestamp.now().isoformat(timespec="seconds")


def car_k(store: dict, car_id, tier: str) -> dict:
    """{'k', 'n_trips', 'n_entries', 'last'} for a car and tier, or {} if not calibrated for that tier."""
    car = store.get("cars", {}).get(str(car_id)) if car_id is not None else None
    if not car:
        return {}
    es = [e for e in car["entries"] if tier in e["ratios"]]
    if not es:
        return {}
    return {"k": float(np.median([e["ratios"][tier] for e in es])), "n_trips": sum(e["trips"] for e in es),
            "n_entries": len(es), "last": es[-1]["when"]}


def predict_style(bundle: dict, trips_df: pd.DataFrame) -> list:
    """Assign eco / normal / aggressive with the saved k-means model."""
    X = pd.DataFrame({c: trips_df[c] if c in trips_df else np.nan for c in bundle["features"]}).astype(float)
    X = X.clip(pd.Series(bundle["clip_lo"]), pd.Series(bundle["clip_hi"]), axis=1)
    X = X.fillna(pd.Series(bundle["scaler"].mean_, index=bundle["features"]))  # missing -> average
    c = bundle["kmeans"].predict(bundle["scaler"].transform(X))
    return [bundle["label_map"][int(i)] for i in c]