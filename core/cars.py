"""Car catalogue: make / model / year -> engine size, from Data/CAR_Prediction_System_DATASET.csv
(Pakistani used-car listings). Cylinders are estimated from engine size. Manual entry is always allowed.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd

import motorlytics_features as mf

SRC = mf.DATA / "CAR_Prediction_System_DATASET.csv"
TWO_WORD_MAKES = ("Mercedes Benz", "Range Rover", "Land Rover", "Alfa Romeo", "Aston Martin", "Rolls Royce")


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
    d = pd.read_csv(SRC, usecols=["NAME", "Model", "Engine Capacity", "Engine Type"], low_memory=False)
    mm = d["NAME"].map(_split_name)
    cat = pd.DataFrame({
        "make": mm.str[0], "model": mm.str[1],
        "year": pd.to_numeric(d["Model"].map(_first), errors="coerce"),
        "disp_L": pd.to_numeric(d["Engine Capacity"].map(_first).str.extract(r"(\d+)")[0], errors="coerce") / 1000,
        "fuel": d["Engine Type"].map(_first).str.lower(),
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
    vc = m["disp_L"].round(1).value_counts()
    return [(float(k), int(v)) for k, v in vc.items()]


def lookup(make: str, model: str, year=None) -> dict:
    """Most common engine size and estimated cylinders for make/model/year, or {} if unknown."""
    opts = engine_options(make, model, year)
    if not opts:
        return {}
    disp = opts[0][0]
    return {"disp_L": disp, "cylinders": estimate_cylinders(disp), "options": opts}


def estimate_cylinders(disp_L) -> float:
    return mf.estimate_cylinders(disp_L)


def car_label(car: dict) -> str:
    if not car:
        return "no car"
    parts = [str(car.get(k)) for k in ("make", "model", "year") if car.get(k)]
    disp = car.get("disp_L")
    return " ".join(parts) + (f" ({disp:.1f} L)" if disp is not None and not np.isnan(disp) else "")
