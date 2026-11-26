"""
Generalisation experiments on the external data and the choice of the app's model per tier
(run after 03_train.py --external ... and 08_validate_exp1.py).

    python 09_generalisation.py

Models scored per tier: every COURSE model (DecisionTree, MLP, kNN, LWR) and LightGBM as reference.
All saved models clip their inputs to the training 1st-99th percentile (motorlytics_features.ClipInputs).

1. Cleaned enviroCar test set: drop the Ford Fusion Hybrid (a hybrid, excluded from VED training),
   "d dd 2014" (unidentifiable), and vehicles with < 3 trips. Per-model results.
2. Per-vehicle calibration: for each external vehicle (cleaned enviroCar + exp1 under both fuel
   assumptions) take its first n trips by start time, k = median(measured / predicted), predict
   the remaining trips as k x prediction. Before vs after on the SAME remaining trips, n = 3 and 1.
   Per-vehicle error = median(prediction / measured - 1) over its remaining trips.
3. Model table + app choice per tier: VED test MAPE, enviroCar MAPE before / after 3-trip
   calibration, exp1 MAPE. Rule: among course models whose VED test MAPE is within 2 points of the
   best course VED test MAPE, pick the lowest calibrated enviroCar MAPE. Written to
   models/meta_trip.json as best_<tier> (03_train.py's CV choice is kept as best_cv_<tier>).
4. Multi-source training for the chosen models (+ LightGBM): cleaned enviroCar vehicles split 50/50
   by vehicle (seed 42); refit on the VED training vehicles (same 80% split as 03_train.py) with and
   without the first enviroCar half; test on the second half and on exp1.

Output: Data/processed/results_trip/generalisation.txt (+ CSVs), plots/generalisation_calibration_mape.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # single-threaded BLAS: ~2x faster here for small matrices (MLP, LWR)

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

import motorlytics_features as mf

SEED = 42
REP = mf.PROC / "results_trip"
TIERS = ["behaviour_only", "full_no_load", "full"]
COURSE = ["DecisionTree", "MLP", "kNN", "LWR"]
REFERENCE_MODEL = "LightGBM"
VED_TOLERANCE = 2.0                 # MAPE points
TARGET = "l_per_100km"
EXCLUDE = ["Ford Fusion Hybrid 2017", "d dd 2014"]
ENV = "enviroCar (cleaned)"
lines = []
pd.set_option("display.width", 250)


def say(s=""):
    print(s)
    lines.append(str(s))


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": len(y), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mape": float(np.mean(np.abs(p - y) / y) * 100), "bias": float(np.mean(p - y))}


meta_path = mf.MODELS / "meta_trip.json"
meta = json.loads(meta_path.read_text())
# keys are meta model keys "<tier>/<model>": all course models + the reference model, per tier
KEYS = [f"{fs}/{m}" for fs in TIERS for m in COURSE + [REFERENCE_MODEL] if f"{fs}/{m}" in meta["models"]]
TIER_OF = {k: k.split("/")[0] for k in KEYS}
NAME = {k: k.split("/")[1] for k in KEYS}
GROUP_OF = {k: meta["models"][k].get("group", "course" if NAME[k] in COURSE else "reference") for k in KEYS}
bundles = {k: joblib.load(meta["models"][k]["path"]) for k in KEYS}
say(f"models: {len(KEYS)} ({', '.join(COURSE)} + {REFERENCE_MODEL} reference, per tier); "
    f"inputs clipped to the training {bundles[KEYS[0]].get('clip_q', '?')}-{100 - bundles[KEYS[0]].get('clip_q', 0)} "
    "percentile range inside each model")


def predict(df, k):
    b = bundles[k]
    return b["model"].predict(mf.to_matrix(df, b["features"]))


# ---------------- 1. cleaned enviroCar ----------------
ec = pd.read_parquet(mf.PROC / "envirocar_trips.parquet").dropna(subset=[TARGET])
n0_t, n0_v = len(ec), ec["vehid"].nunique()
ec = ec[~ec["veh_label"].isin(EXCLUDE)]
ec = ec[ec.groupby("vehid")[TARGET].transform("size") >= 3].reset_index(drop=True)
ec["label"], ec["source"] = ec["veh_label"].str.strip(), "enviroCar"
for k in KEYS:
    ec[f"pred_{k}"] = predict(ec, k)
say("\n=== 1. Cleaned enviroCar test set (no calibration) ===")
say(f"{n0_t} trips / {n0_v} vehicles -> {len(ec)} trips / {ec['vehid'].nunique()} vehicles "
    f"(dropped {EXCLUDE} and vehicles with < 3 trips)")
rows = []
for k in KEYS:
    s = scores(ec[TARGET], ec[f"pred_{k}"])
    rows.append({"tier": TIER_OF[k], "group": GROUP_OF[k], "model": NAME[k], "vehicles": ec["vehid"].nunique(),
                 "trips": s["n"], "rmse": s["rmse"], "mape": s["mape"], "bias": s["bias"]})
say(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------- external sets ----------------
ext_sets = {ENV: ec}
exp1_path = REP / "external_exp1_predictions.parquet"
if exp1_path.exists():
    x1 = pd.read_parquet(exp1_path)
    for fuel, g in x1.groupby("fuel"):
        g = g.dropna(subset=[TARGET]).reset_index(drop=True)
        for k in KEYS:                        # predict with the current saved models
            g[f"pred_{k}"] = predict(g, k)
        ext_sets[f"exp1, {fuel}"] = g
else:
    say("\n(exp1 trips not found: run 08_validate_exp1.py first)")


# ---------------- 2. per-vehicle calibration ----------------
def calibrate(df, k, n_cal):
    out = []
    for v, g in df.sort_values("start_dt").groupby("vehid", sort=False):
        if len(g) <= n_cal:
            continue
        cal, test = g.iloc[:n_cal], g.iloc[n_cal:]
        kk = float(np.median(cal[TARGET] / cal[f"pred_{k}"]))
        out.append(test.assign(k=kk, p_before=test[f"pred_{k}"], p_after=kk * test[f"pred_{k}"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def vehicle_err(t, col):
    return t.groupby("vehid").apply(lambda g: float(np.median(g[col] / g[TARGET] - 1) * 100),
                                    include_groups=False)


say("\n=== 2. Per-vehicle calibration (k = median(measured / predicted) over the first n trips) ===")
say("before / after are scored on the same remaining trips; vehicle error = median(pred / measured - 1)")
cal_rows = []
for name, df in ext_sets.items():
    for n_cal in (3, 1):
        for k in KEYS:
            t = calibrate(df, k, n_cal)
            if t.empty:
                continue
            sb, sa = scores(t[TARGET], t["p_before"]), scores(t[TARGET], t["p_after"])
            eb, ea = vehicle_err(t, "p_before"), vehicle_err(t, "p_after")
            cal_rows.append({
                "data": name, "cal_trips": n_cal, "tier": TIER_OF[k], "group": GROUP_OF[k], "model": NAME[k],
                "vehicles": t["vehid"].nunique(), "test_trips": len(t),
                "mape_before": sb["mape"], "mape_after": sa["mape"],
                "rmse_before": sb["rmse"], "rmse_after": sa["rmse"],
                "bias_before": sb["bias"], "bias_after": sa["bias"],
                "veh_within10_before": int((eb.abs() <= 10).sum()), "veh_within10_after": int((ea.abs() <= 10).sum()),
                "veh_within15_before": int((eb.abs() <= 15).sum()), "veh_within15_after": int((ea.abs() <= 15).sum()),
                "k_median": float(t.groupby("vehid")["k"].first().median())})
cal = pd.DataFrame(cal_rows)
for name in ext_sets:
    c = cal[cal["data"] == name]
    if not c.empty:
        say(f"\n[{name}]")
        say(c.drop(columns="data").round(3).to_string(index=False))
cal.round(4).to_csv(REP / "generalisation_calibration.csv", index=False)

# ---------------- 3. model table + the app's model per tier ----------------
say(f"\n=== 3. Model table and the app's model per tier ===")
say(f"rule: among course models with VED test MAPE <= best course VED test MAPE + {VED_TOLERANCE:.0f} points, "
    "pick the lowest enviroCar MAPE after 3-trip calibration")
test_m = pd.read_csv(REP / "test_metrics.csv").set_index(["feature_set", "model"])
c3 = cal[(cal["cal_trips"] == 3) & (cal["data"] == ENV)].set_index(["tier", "model"])
ext_unc = {}
for name, df in ext_sets.items():
    for k in KEYS:
        ext_unc[(name, TIER_OF[k], NAME[k])] = scores(df[TARGET], df[f"pred_{k}"])["mape"]
table, choice, selection = [], {}, {"rule": f"course model with the lowest enviroCar MAPE after 3-trip "
                                            f"calibration among those within {VED_TOLERANCE} MAPE points of the best "
                                            "course VED test MAPE", "tiers": {}}
for fs in TIERS:
    trs = []
    for m in COURSE + [REFERENCE_MODEL]:
        if f"{fs}/{m}" not in KEYS:
            continue
        r = {"tier": fs, "model": m, "group": GROUP_OF[f"{fs}/{m}"],
             "ved_test_mape": float(test_m.loc[(fs, m), "mape"]),
             "envirocar_mape": ext_unc[(ENV, fs, m)],
             "envirocar_mape_cal3": float(c3.loc[(fs, m), "mape_after"]) if (fs, m) in c3.index else np.nan}
        for name in ext_sets:
            if name.startswith("exp1"):
                r[f"{name} mape"] = ext_unc[(name, fs, m)]
        trs.append(r)
    t = pd.DataFrame(trs)
    course = t[t["group"] == "course"]
    best_ved = course["ved_test_mape"].min()
    eligible = course[course["ved_test_mape"] <= best_ved + VED_TOLERANCE]
    chosen = eligible.sort_values(["envirocar_mape_cal3", "ved_test_mape"]).iloc[0]["model"]
    t["within_2pts_of_best_course_ved"] = (t["group"] == "course") & (t["ved_test_mape"] <= best_ved + VED_TOLERANCE)
    t["app_choice"] = t["model"] == chosen
    choice[fs] = chosen
    table.append(t)
    selection["tiers"][fs] = {"chosen": f"{fs}/{chosen}", "best_course_ved_test_mape": round(float(best_ved), 3),
                              "eligible": eligible["model"].tolist(),
                              "cv_choice": meta.get(f"best_cv_{fs}", meta.get(f"best_{fs}"))}
table = pd.concat(table, ignore_index=True)
say(table.round(2).to_string(index=False))
table.round(4).to_csv(REP / "model_selection_table.csv", index=False)

# write the choice; keep 03_train.py's CV-based choice for the record
for fs in TIERS:
    meta.setdefault(f"best_cv_{fs}", meta.get(f"best_{fs}"))
    meta[f"best_{fs}"] = f"{fs}/{choice[fs]}"
meta["selection"] = selection
meta_path.write_text(json.dumps(meta, indent=2))
say("\napp model per tier (written to models/meta_trip.json best_<tier>):")
for fs in TIERS:
    s = selection["tiers"][fs]
    say(f"  {fs}: {choice[fs]}  (eligible: {s['eligible']}; best course VED test MAPE {s['best_course_ved_test_mape']:.2f}; "
        f"CV choice was {s['cv_choice']})")

# ---------------- 4. multi-source training (chosen models + reference) ----------------
say("\n=== 4. Multi-source training: VED + half of enviroCar vs VED only ===")
ved = pd.read_parquet(mf.PROC / "trips.parquet").dropna(subset=[TARGET]).reset_index(drop=True)
tr_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(ved, groups=ved["vehid"]))
ved_train = ved.iloc[tr_idx]                          # same training vehicles as 03_train.py
vids = np.array(sorted(ec["vehid"].unique()))
rng = np.random.default_rng(SEED)
rng.shuffle(vids)
half_a, half_b = set(vids[: len(vids) // 2]), set(vids[len(vids) // 2:])
ec_a, ec_b = ec[ec["vehid"].isin(half_a)], ec[ec["vehid"].isin(half_b)]
say(f"enviroCar split by vehicle (seed {SEED}): train half {len(half_a)} vehicles / {len(ec_a)} trips, "
    f"test half {len(half_b)} vehicles / {len(ec_b)} trips; VED training: {len(ved_train):,} trips / "
    f"{ved_train['vehid'].nunique()} vehicles (enviroCar adds {len(ec_a) / len(ved_train):.1%} more trips)")
say(f"test half vehicles: {sorted(ec_b['label'].unique())}")
tests = {"enviroCar test half": ec_b}
tests.update({k: v for k, v in ext_sets.items() if k.startswith("exp1")})
ms_rows = []
for fs in ["behaviour_only", "full_no_load"]:
    for k in [f"{fs}/{choice[fs]}", f"{fs}/{REFERENCE_MODEL}"]:
        feats = bundles[k]["features"]
        for train_name, train_df in [("VED only", ved_train), ("VED + enviroCar half", pd.concat([ved_train, ec_a]))]:
            m = clone(bundles[k]["model"]).fit(mf.to_matrix(train_df, feats), train_df[TARGET].to_numpy())
            for test_name, td in tests.items():
                s = scores(td[TARGET], m.predict(mf.to_matrix(td, feats)))
                ms_rows.append({"tier": fs, "group": GROUP_OF[k], "model": NAME[k], "training": train_name,
                                "test": test_name, "vehicles": td["vehid"].nunique(), "trips": s["n"],
                                "rmse": s["rmse"], "mape": s["mape"], "bias": s["bias"]})
ms = pd.DataFrame(ms_rows)
say(ms.round(3).to_string(index=False))
ms.round(4).to_csv(REP / "generalisation_multisource.csv", index=False)
say("note: one 50/50 split with few vehicles; differences of a few MAPE points are within split-to-split noise")

(REP / "generalisation.txt").write_text("\n".join(lines), encoding="utf-8")

# ---------------- plot: MAPE before / after 3-trip calibration, the app's models ----------------
cp = cal[cal["cal_trips"] == 3].copy()
cp = cp[cp.apply(lambda r: r["model"] == choice[r["tier"]], axis=1)]
if not cp.empty:
    datas = list(dict.fromkeys(cp["data"]))
    fig, axs = plt.subplots(1, len(datas), figsize=(4.2 * len(datas), 4), sharey=True, squeeze=False)
    for ax, name in zip(axs[0], datas):
        c = cp[cp["data"] == name].set_index("tier").reindex(TIERS)
        x = np.arange(len(TIERS))
        ax.bar(x - 0.18, c["mape_before"], 0.36, color="#9fb4cf", label="before")
        ax.bar(x + 0.18, c["mape_after"], 0.36, color="#2a78d6", label="after (k from 3 trips)")
        for i, (b, a_) in enumerate(zip(c["mape_before"], c["mape_after"])):
            ax.text(i - 0.18, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + 0.18, a_, f"{a_:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [t.replace("_", " ") + "\n(" + str(c.loc[t, "model"]) + ")" for t in TIERS])
        ax.set_title(f"{name}\n{int(c['vehicles'].max())} vehicles", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df")
        ax.set_axisbelow(True)
    axs[0][0].set_ylabel("MAPE on remaining trips (%)")
    axs[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Per-vehicle calibration on external data (the app's course models)", fontweight="bold")
    fig.tight_layout()
    out = mf.Path("plots")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "generalisation_calibration_mape.png", dpi=130, bbox_inches="tight")
    print(f"saved {out / 'generalisation_calibration_mape.png'}")
print(f"saved {REP / 'generalisation.txt'}")
