"""Car catalogue and car profiles.

Catalogue: make / model / year -> suggested engine size, transmission, engine type, body type and
assembly (most common value among Pakistani used-car listings in Data/CAR_Prediction_System_DATASET.csv).
Cylinders are estimated from engine size. Every suggestion can be changed, and manual entry is allowed.
Profile: per-car statistics from the trips table (car_profile).
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd

import motorlytics_features as mf

SRC = mf.DATA / "CAR_Prediction_System_DATASET.csv"
TWO_WORD_MAKES = ("Mercedes Benz", "Range Rover", "Land Rover", "Alfa Romeo", "Aston Martin", "Rolls Royce")
FUEL_TYPES = ["petrol", "CNG", "diesel", "hybrid"]
TRANSMISSIONS = ["Manual", "Automatic"]
ASSEMBLY = ["Local", "Imported"]
BODY_TYPES = ["Hatchback", "Sedan", "SUV", "Crossover", "Mini Van", "Van", "Double Cabin", "Pick Up", "Coupe",
              "Station Wagon", "Compact sedan", "Compact SUV", "MPV", "Micro Van", "Convertible"]


def _first(s):
    """"['1300 cc']" -> "1300 cc"."""
    m = re.findall(r"'([^']*)'", str(s))
    return m[0].strip() if m else str(s).strip("[] ")


def _split_name(name: str):
    name = " ".join(str(name).split())
    make = next((m for m in TWO_WORD_MAKES if name.startswith(m + " ")), name.split(" ")[0])
    rest = name[len(make):].strip().split(" ")
    model = rest[0] if rest else ""
    if len(rest) > 1 and rest[1].lower() == "class":       # "E Class", "GLB Class"
        model = f"{rest[0]} Class"
    return make, model


@lru_cache(maxsize=1)
def catalogue() -> pd.DataFrame:
    """One row per listing: make, model, year, disp_L, fuel. Empty if the file is missing."""
    if not SRC.exists():
        return pd.DataFrame(columns=["make", "model", "year", "disp_L", "fuel"])
    d = pd.read_csv(SRC, usecols=["NAME", "Model", "Engine Capacity", "Engine Type", "Transmission", "Body type",
                                  "Assembled"], low_memory=False)
    mm = d["NAME"].map(_split_name)
    trans = d["Transmission"].map(_first).str.title()
    fuel = d["Engine Type"].map(_first).str.lower()
    body = d["Body type"].map(_first)
    asm = d["Assembled"].map(_first).str.title()
    cat = pd.DataFrame({                     # junk values ("local" in the wrong column, empty) become NaN
        "make": mm.str[0], "model": mm.str[1],
        "year": pd.to_numeric(d["Model"].map(_first), errors="coerce"),
        "disp_L": pd.to_numeric(d["Engine Capacity"].map(_first).str.extract(r"(\d+)")[0], errors="coerce") / 1000,
        "fuel": fuel.where(fuel.isin(["petrol", "diesel", "hybrid", "electric"])),
        "transmission": trans.where(trans.isin(TRANSMISSIONS)),
        "body_type": body.where(body.notna() & ~body.str.lower().isin(["", "local", "nan"])),
        "assembled": asm.where(asm.isin(ASSEMBLY)),
    })
    cat = cat[cat["disp_L"].between(0.5, 7.0) & cat["year"].notna()]
    cat["year"] = cat["year"].astype(int)
    return cat.reset_index(drop=True)


def makes() -> list:
    return sorted(catalogue()["make"].dropna().unique().tolist())


def models(make: str) -> list:
    c = catalogue()
    return sorted(c.loc[c["make"] == make, "model"].dropna().unique().tolist())


def years(make: str, model: str) -> list:
    c = catalogue()
    return sorted(c.loc[(c["make"] == make) & (c["model"] == model), "year"].unique().tolist(), reverse=True)


def engine_options(make: str, model: str, year=None) -> list:
    """Engine sizes (L) seen for this car, most common first: [(disp_L, listings), ...].
    Falls back to all years of the model if the year has no listings."""
    c = catalogue()
    m = c[(c["make"] == make) & (c["model"] == model)]
    if year is not None and (m["year"] == year).any():
        m = m[m["year"] == year]
    vc = m["disp_L"].map(round_disp).value_counts()
    return [(float(k), int(v)) for k, v in vc.items()]


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def lookup(make: str, model: str, year=None) -> dict:
    """Suggestions for make/model/year: most common engine size (+ all sizes seen), estimated cylinders,
    transmission, fuel type, body type, assembly, and how many listings they come from. {} if unknown.
    Each field falls back to all years of the model if the year has no value for it."""
    opts = engine_options(make, model, year)
    if not opts:
        return {}
    c = catalogue()
    allyears = c[(c["make"] == make) & (c["model"] == model)]
    m = allyears[allyears["year"] == year] if year is not None and (allyears["year"] == year).any() else allyears
    pick = {f: _mode(m[f]) if m[f].notna().any() else _mode(allyears[f])
            for f in ("transmission", "fuel", "body_type", "assembled")}
    disp = opts[0][0]
    fuel = pick["fuel"] if pick["fuel"] in FUEL_TYPES else None        # electric is not supported
    return {"disp_L": disp, "cylinders": estimate_cylinders(disp), "options": opts, "fuel_type": fuel,
            "transmission": pick["transmission"], "body_type": pick["body_type"], "assembled": pick["assembled"],
            "listings": int(len(m))}


def estimate_cylinders(disp_L) -> int | None:
    c = mf.estimate_cylinders(disp_L)
    return None if c != c else int(c)


def round_disp(disp_L):
    """Engine size in L: 2 decimals below 1 L (660 cc -> 0.66), 1 decimal above (1,298 cc -> 1.3)."""
    if disp_L is None or disp_L != disp_L:
        return disp_L
    return round(float(disp_L), 2 if disp_L < 1 else 1)


def fmt_disp(disp_L, unit: str = " L") -> str:
    """'0.66 L', '0.8 L', '1.3 L'; '' if unknown."""
    if disp_L is None or disp_L != disp_L:
        return ""
    d = round_disp(disp_L)
    s = f"{d:.2f}".rstrip("0") if d < 1 else f"{d:.1f}"
    return s + unit


def car_title(car: dict) -> str:
    """Nickname if set, else make model year."""
    if not car:
        return "no car"
    return car.get("nickname") or " ".join(str(car.get(k)) for k in ("make", "model", "year") if car.get(k))


def mask_registration(reg: str) -> str:
    """Registration with the middle hidden: first letters and last 3 characters, e.g. LEA-..-123."""
    if not reg:
        return ""
    s = "".join(ch for ch in str(reg).upper() if ch.isalnum())
    if len(s) <= 4:
        return "\u2022\u2022" + s[-2:]
    letters = "".join(ch for ch in s[:3] if ch.isalpha()) or s[:2]
    return f"{letters}-\u2022\u2022-{s[-3:]}"


def car_profile(car_id) -> dict:
    """Per-car statistics from the trips table and the calibration store."""
    from core import app_core, db
    car = db.get_car(car_id)
    if not car:
        return {}
    t = pd.DataFrame(db.get_trips(car["user_id"], car_id))
    out = {"car": car, "trips": 0, "km": 0.0, "litres": 0.0, "pkr": None, "avg_measured_l100": None,
           "avg_predicted_l100": None, "avg_l100": None, "best_trip": None, "worst_trip": None, "long_trips": 0,
           "calibration_k": None, "calibration_trips": 0, "calibration_tier": None, "range_km": None}
    status = app_core.calibration_status(car["car_key"])
    if status:
        tier = "behaviour_only" if "behaviour_only" in status else next(iter(status))
        out.update(calibration_k=status[tier]["k"], calibration_trips=status[tier]["n_trips"], calibration_tier=tier)
    if t.empty:
        return out
    t["l100"] = t["measured_l100"].fillna(t["pred_l100"])                # measured where available
    t["litres"] = t["l100"] * t["dist_km"] / 100
    km, litres = float(t["dist_km"].sum()), float(t["litres"].sum())
    out.update(trips=len(t), km=km, litres=litres,
               pkr=float(t["cost_pkr"].sum()) if t["cost_pkr"].notna().any() else None,
               avg_measured_l100=float(t["measured_l100"].mean()) if t["measured_l100"].notna().any() else None,
               avg_predicted_l100=float(t["pred_l100"].mean()) if t["pred_l100"].notna().any() else None,
               avg_l100=litres / km * 100 if km > 0 else None)        # distance-weighted
    long = t[t["dist_km"] >= 3]
    out["long_trips"] = int(len(long))                     # best / worst only make sense with 2 or more
    if len(long):
        keep = ["id", "started_at", "dist_km", "l100", "eco_score"]
        out["best_trip"] = long.loc[long["l100"].idxmin(), keep].to_dict()
        out["worst_trip"] = long.loc[long["l100"].idxmax(), keep].to_dict()
    if car.get("tank_litres") and out["avg_l100"]:
        out["range_km"] = float(car["tank_litres"]) / out["avg_l100"] * 100
    return out


def car_label(car: dict) -> str:
    if not car:
        return "no car"
    parts = [str(car.get(k)) for k in ("make", "model", "year") if car.get(k)]
    disp = car.get("disp_L")
    return " ".join(parts) + (f" ({fmt_disp(disp)})" if disp is not None and not np.isnan(disp) else "")
