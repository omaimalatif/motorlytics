"""
Join EPA fuel-economy ratings (Data/database.csv) to the VED vehicles.

VED's static file has no make, model or model year (vehicles are anonymised), so the match
uses engine specs: displacement, cylinders and turbo/supercharger, plus vehicle class when
VED has one (about 5 of 357 vehicles), within a model-year window (--years, default
2005-2018; VED was recorded Nov 2017 - Nov 2018, so this window is an assumption).
Many EPA models share an engine, so each VED vehicle gets the MEDIAN city / highway /
combined rating of all matching EPA rows, with the number of candidates and the IQR of the
combined rating as match-quality columns. See motorlytics_features.match_epa for the levels.

    python 07_join_epa.py                         # petrol ICE vehicles (as in 02_build_datasets.py)
    python 07_join_epa.py --years 2008-2018 --min-n 5
    python 02_build_datasets.py --epa Data/processed/epa_lookup.csv    # use as optional features

Output: Data/processed/epa_lookup.csv, one row per VehId:
  vehid, epa_city_l100, epa_hwy_l100, epa_comb_l100, epa_comb_l100_iqr, epa_class,
  epa_n_candidates, epa_match_level (1 = with class, 2 = exact engine, 3 = disp +-0.1 L)
"""
import argparse

import pandas as pd

import motorlytics_features as mf

ap = argparse.ArgumentParser()
ap.add_argument("--epa", default=str(mf.DATA / "database.csv"))
ap.add_argument("--years", default="2005-2018", help="EPA model-year window, e.g. 2005-2018")
ap.add_argument("--min-n", type=int, default=3, help="EPA rows needed before a match level is accepted")
ap.add_argument("--all-types", action="store_true", help="also try HEV / diesel vehicles (default: petrol ICE)")
ap.add_argument("--out", default=str(mf.PROC / "epa_lookup.csv"))
a = ap.parse_args()
y0, y1 = (int(x) for x in a.years.split("-"))

# ---------------- VED vehicles ----------------
static = mf.parse_static(pd.read_excel(mf.DATA / "VED_Static_Data_ICE&HEV.xlsx"))
veh = static if a.all_types else static[(static["veh_type"] == "ICE") & ~static["is_diesel"]]
print(f"VED vehicles to match: {len(veh)} ({'all types' if a.all_types else 'petrol ICE'})")
print(f"  with displacement: {veh['disp_L'].notna().sum()}, cylinders: {veh['cylinders'].notna().sum()}, "
      f"boosted: {int(veh['boost'].sum())}, known class: {(veh['veh_class'] != 'unknown').sum()}")

# ---------------- EPA ratings ----------------
epa = mf.load_epa(a.epa, years=(y0, y1))
print(f"\nEPA rows used: {len(epa):,} (gasoline, non-hybrid, model years {y0}-{y1})")
print(f"  boosted share: {epa['boost'].mean():.1%}   classes: {epa['epa_class'].value_counts().to_dict()}")
print(f"  combined L/100 km: median {epa['epa_comb_l100'].median():.1f}, "
      f"IQR {epa['epa_comb_l100'].quantile(0.25):.1f}-{epa['epa_comb_l100'].quantile(0.75):.1f}")

# ---------------- match per vehicle ----------------
rows = []
for _, v in veh.iterrows():
    m = mf.match_epa(epa, v["disp_L"], v["cylinders"], v["boost"], v["veh_class"], min_n=a.min_n)
    rows.append({"vehid": v["vehid"], "disp_L": v["disp_L"], "cylinders": v["cylinders"],
                 "boost": v["boost"], "veh_class": v["veh_class"], **m})
lk = pd.DataFrame(rows)
for c in mf.EPA_FEATURES + ["epa_comb_l100_iqr", "epa_class", "epa_n_candidates", "epa_match_level"]:
    if c not in lk:
        lk[c] = pd.NA

# ---------------- report ----------------
matched = lk["epa_comb_l100"].notna()
print(f"\nmatch rate: {matched.sum()} of {len(lk)} vehicles ({matched.mean():.1%})")
print("by match level (1 = with class, 2 = exact engine, 3 = disp +-0.1 L):")
print(lk["epa_match_level"].value_counts(dropna=False).sort_index().to_string())
if matched.any():
    print("\ncandidates per matched vehicle:",
          lk.loc[matched, "epa_n_candidates"].describe()[["min", "50%", "max"]].to_dict())
    print("IQR of combined L/100 km among candidates (spread = how ambiguous the engine match is):",
          lk.loc[matched, "epa_comb_l100_iqr"].describe()[["50%", "max"]].round(2).to_dict())
    if (lk["veh_class"] != "unknown").any():
        known = lk[matched & (lk["veh_class"] != "unknown")]
        print(f"class agreement where VED has a class: "
              f"{(known['veh_class'] == known['epa_class']).sum()} of {len(known)}")
unm = lk[~matched]
if len(unm):
    print("\nunmatched engine specs (disp_L, cylinders, boost -> vehicles):")
    print(unm.groupby(["disp_L", "cylinders", "boost"], dropna=False).size().to_string())

mf.PROC.mkdir(parents=True, exist_ok=True)
lk.to_csv(a.out, index=False)
print(f"\nsaved {a.out}")
