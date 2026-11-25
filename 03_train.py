"""
Train and evaluate Motorlytics models.

    python 03_train.py --task trip              # main task: trip L/100 km
    python 03_train.py --task row               # second task: instantaneous L/h
    python 03_train.py --task trip --external Data/processed/envirocar_trips.parquet

Protocol (vehicles never appear in both train and test):
  1. 20% of vehicles held out as the TEST set, untouched until the end.
  2. Feature selection on the training vehicles with 3 methods
     (mutual information filter, Lasso embedded, permutation importance);
     a feature is kept if at least 2 methods keep it.
  3. 10-fold GroupKFold CV (by vehicle) on the training vehicles for 3 models
     -> RMSE, MAE, R2, MAPE, fit time, inference time.
  4. Friedman test + pairwise Wilcoxon and paired t-tests (Holm-corrected).
  5. Final fit on all training vehicles, evaluation on the held-out vehicles,
     error analysis by vehicle class, speed band, temperature band, trip length.

Also (trip task):
  - light tuning: RandomizedSearchCV (n_iter, GroupKFold(5)) on the training vehicles for
    LightGBM and RandomForest per feature set; the best params are used for CV, tests and the
    final models (CV scores are therefore slightly optimistic; the held-out test is not affected)
  - ablation "full_no_load" = selected full features without mean_load
  - baselines on the held-out vehicles: DummyRegressor (mean) and median L/100 km per disp_L
"""
import argparse
import json
import time
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import motorlytics_features as mf

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["trip", "row"], default="trip")
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--test-size", type=float, default=0.2)
ap.add_argument("--max-rows", type=int, default=300_000, help="row task: cap for speed")
ap.add_argument("--external", default=None, help="trip parquet from another source (e.g. enviroCar)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tune-iter", type=int, default=20, help="RandomizedSearchCV iterations (0 = no tuning)")
ap.add_argument("--tune-folds", type=int, default=5)
ap.add_argument("--reuse-params", action="store_true",
                help="skip tuning; load tuned_params from models/meta_<task>.json "
                     "(or results_<task>/best_params.json)")
a = ap.parse_args()
SEED = a.seed
rng = np.random.default_rng(SEED)
OUT = mf.MODELS
OUT.mkdir(exist_ok=True)
REP = mf.PROC / f"results_{a.task}"
REP.mkdir(parents=True, exist_ok=True)

# ---------------- data + candidate features ----------------
if a.task == "trip":
    df = pd.read_parquet(mf.PROC / "trips.parquet")
    target = "l_per_100km"
    behaviour = mf.TRIP_BEHAVIOUR + mf.TRIP_CONTEXT
    engine = mf.TRIP_ENGINE
else:
    df = pd.read_parquet(mf.PROC / "rows.parquet")
    if len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=SEED)
    target = "fuel_Lph"
    behaviour = [f for f in mf.ROW_FEATURES if not f.startswith(("rpm", "load"))] + \
                ["temp_c", "humidity", "precip_mm", "wind_kmh"]
    engine = ["rpm", "rpm_per_kmh", "rpm_change", "rpm_roll5", "load"]
vehicle = list(mf.VEHICLE)   # cls_* dropped: VED vehicle class is ~99% "unknown"
df = df.dropna(subset=[target]).reset_index(drop=True)


def present(cols):
    return [c for c in cols if c in df and df[c].notna().mean() > 0.5]


FEATURE_SETS = {
    "behaviour_only": present(behaviour + vehicle),          # works with a phone + car spec
    "full": present(behaviour + vehicle + engine),            # needs an OBD-II adapter
}
print(f"task={a.task}  rows={len(df):,}  vehicles={df['vehid'].nunique()}  target={target}")
for k, v in FEATURE_SETS.items():
    print(f"  {k}: {len(v)} candidate features")

# ---------------- 1. hold out 20% of vehicles ----------------
gss = GroupShuffleSplit(n_splits=1, test_size=a.test_size, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=df["vehid"]))
train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
print(f"train: {len(train):,} rows / {train['vehid'].nunique()} vehicles | "
      f"test: {len(test):,} rows / {test['vehid'].nunique()} vehicles")


def make_models(params=None):
    # Swap these for algorithms from your course outline if needed
    # (e.g. KNeighborsRegressor, DecisionTreeRegressor, SVR).
    # params: {"LightGBM": {...}, "RandomForest": {...}} from tuning (pipeline-style names for RF)
    params = params or {}
    models = {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=200, min_samples_leaf=5 if a.task == "trip" else 20,
                                  max_features=0.5, max_samples=0.5 if a.task == "row" else None,
                                  n_jobs=-1, random_state=SEED)),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                                  random_state=SEED, verbose=-1),
    }
    for name, p in params.items():
        models[name].set_params(**p)
    return models


SEARCH_SPACES = {
    "LightGBM": {"num_leaves": [15, 31, 63, 127], "learning_rate": loguniform(0.01, 0.1),
                 "n_estimators": randint(200, 1001), "min_child_samples": [10, 20, 40, 80],
                 "colsample_bytree": uniform(0.5, 0.5)},
    "RandomForest": {"randomforestregressor__max_features": [0.3, 0.5, 0.7, 1.0, "sqrt"],
                     "randomforestregressor__min_samples_leaf": [1, 2, 5, 10, 20],
                     "randomforestregressor__n_estimators": [100, 200, 300, 400]},
}


def tune(data, cols, tag):
    """Light RandomizedSearchCV on the training vehicles only (grouped by vehicle)."""
    if a.tune_iter <= 0:
        return {}
    X, y = data[cols].to_numpy(np.float32), data[target].to_numpy()
    best_params = {}
    for name, space in SEARCH_SPACES.items():
        t = time.perf_counter()
        rs = RandomizedSearchCV(make_models()[name], space, n_iter=a.tune_iter,
                                cv=GroupKFold(n_splits=a.tune_folds),
                                scoring="neg_root_mean_squared_error", n_jobs=1,  # 1 job: low RAM
                                random_state=SEED, refit=False)
        rs.fit(X, y, groups=data["vehid"])
        bp = {k: (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
        best_params[name] = bp
        print(f"  [{tag}] {name}: best {a.tune_folds}-fold CV RMSE={-rs.best_score_:.3f}  "
              f"{bp}  ({time.perf_counter() - t:.0f}s)")
        pd.DataFrame(rs.cv_results_).to_csv(REP / f"tuning_{tag}_{name}.csv", index=False)
    return best_params


# ---------------- 2. feature selection (3 methods, keep if >= 2 agree) ----------------
def select_features(data, cols, tag):
    X = data[cols].astype(float)
    y = data[target].to_numpy()
    # 0. drop near-constant columns
    nunq = X.nunique()
    top_share = X.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
    cols = [c for c in cols if nunq[c] > 1 and top_share[c] < 0.99]
    X = X[cols]
    Xi = X.fillna(X.median())
    sub = rng.choice(len(Xi), size=min(30_000, len(Xi)), replace=False)
    # A. filter: mutual information
    mi = pd.Series(mutual_info_regression(Xi.iloc[sub], y[sub], random_state=SEED), index=cols)
    # redundancy: of any pair with |r| > 0.95 keep the one with higher MI
    corr = Xi.iloc[sub].corr().abs()
    redundant = set()
    for c1, c2 in combinations(cols, 2):
        if corr.loc[c1, c2] > 0.95:
            redundant.add(c2 if mi[c1] >= mi[c2] else c1)
    keep_mi = (mi > 0.01) & ~mi.index.isin(redundant)
    # B. embedded: Lasso on standardised features
    Xs = StandardScaler().fit_transform(Xi.iloc[sub])
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000).fit(Xs, y[sub])
    coef = pd.Series(lasso.coef_, index=cols)
    keep_lasso = coef.abs() > 1e-6
    # C. wrapper-style: permutation importance on a grouped validation split
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    a_idx, b_idx = next(g2.split(X, groups=data["vehid"]))
    m = make_models()["LightGBM"].fit(X.iloc[a_idx].to_numpy(np.float32), y[a_idx])
    bsub = b_idx if len(b_idx) <= 20_000 else rng.choice(b_idx, 20_000, replace=False)
    pi = permutation_importance(m, X.iloc[bsub].to_numpy(np.float32), y[bsub], n_repeats=5,
                                random_state=SEED, scoring="neg_root_mean_squared_error")
    perm = pd.Series(pi.importances_mean, index=cols)
    keep_perm = perm > 0
    table = pd.DataFrame({"mutual_info": mi, "redundant": mi.index.isin(redundant),
                          "lasso_coef": coef, "perm_rmse_increase": perm,
                          "keep_mi": keep_mi, "keep_lasso": keep_lasso, "keep_perm": keep_perm})
    table["votes"] = table[["keep_mi", "keep_lasso", "keep_perm"]].sum(axis=1)
    table["selected"] = table["votes"] >= 2
    if table["selected"].sum() < 5:                       # safety net
        table.loc[perm.nlargest(5).index, "selected"] = True
    table = table.sort_values("perm_rmse_increase", ascending=False)
    table.to_csv(REP / f"feature_selection_{tag}.csv")
    sel = table.index[table["selected"]].tolist()
    print(f"\n[{tag}] feature selection: {len(sel)} of {len(cols)} kept")
    print(table[["mutual_info", "lasso_coef", "perm_rmse_increase", "votes", "selected"]]
          .round(4).to_string())
    return sel


SELECTED = {k: select_features(train, v, k) for k, v in FEATURE_SETS.items()}
# ablation: the selected full set without mean_load (engine load needs a capable OBD adapter)
if "mean_load" in SELECTED["full"]:
    SELECTED["full_no_load"] = [c for c in SELECTED["full"] if c != "mean_load"]
    print(f"\n[full_no_load] {len(SELECTED['full_no_load'])} features (full minus mean_load)")
else:
    print("\n[full_no_load] skipped: mean_load was not selected in the full set")
# EPA ratings per vehicle (07_join_epa.py + 02_build_datasets.py --epa): full + EPA columns
epa_cols = [c for c in mf.EPA_FEATURES if c in df and df[c].notna().mean() > 0.5]
if a.task == "trip" and epa_cols:
    SELECTED["full_epa"] = SELECTED["full"] + [c for c in epa_cols if c not in SELECTED["full"]]
    print(f"[full_epa] {len(SELECTED['full_epa'])} features (full + {epa_cols}), "
          f"EPA coverage {df[epa_cols[0]].notna().mean():.1%} of trips")
else:
    print("[full_epa] skipped: no EPA columns in the data (run 02_build_datasets.py --epa ...)")


def load_params():
    """Tuned params from an earlier run; feature sets without saved params (e.g. full_epa)
    reuse the full set's params."""
    for p, key in [(OUT / f"meta_{a.task}.json", "tuned_params"), (REP / "best_params.json", None)]:
        if p.exists():
            d = json.loads(p.read_text())
            d = d.get(key) if key else d
            if d:
                print(f"reusing tuned params from {p}")
                break
    else:
        raise SystemExit("--reuse-params: no tuned params found; run once without it")
    out = {}
    for fs in SELECTED:
        src = fs if fs in d else ("full" if fs.startswith("full") and "full" in d else None)
        out[fs] = d.get(src, {}) if src else {}
        print(f"  {fs}: params from {src or 'model defaults'}")
    return out


# ---------------- 2b. light hyper-parameter tuning (training vehicles only) ----------------
if a.reuse_params:
    print("\n=== Tuning skipped (--reuse-params) ===")
    PARAMS = load_params()
else:
    print("\n=== Tuning (RandomizedSearchCV, GroupKFold) ===")
    PARAMS = {fs: tune(train, cols, fs) for fs, cols in SELECTED.items()}
(REP / "best_params.json").write_text(json.dumps(PARAMS, indent=2))
print(f"best params saved -> {REP / 'best_params.json'}")


# ---------------- 3. grouped cross-validation ----------------
def metrics(y, p):
    out = {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
           "r2": float(r2_score(y, p))}
    if a.task == "trip":
        out["mape"] = float(np.mean(np.abs((y - p) / y)) * 100)
    return out


gkf = GroupKFold(n_splits=a.folds)
y_tr = train[target].to_numpy()
rows = []
for fs, cols in SELECTED.items():
    X = train[cols].to_numpy(np.float32)
    for fold, (i, j) in enumerate(gkf.split(X, y_tr, train["vehid"])):
        for name, model in make_models(PARAMS.get(fs)).items():
            t = time.perf_counter(); model.fit(X[i], y_tr[i]); fit_s = time.perf_counter() - t
            t = time.perf_counter(); p = model.predict(X[j]); inf = (time.perf_counter() - t) / len(j) * 1e6
            rows.append({"feature_set": fs, "model": name, "fold": fold, **metrics(y_tr[j], p),
                         "fit_s": fit_s, "infer_ms_per_1k": inf})
    print(f"CV done: {fs}")
cv = pd.DataFrame(rows)
cv.to_csv(REP / "cv_folds.csv", index=False)
summary = cv.groupby(["feature_set", "model"]).agg(["mean", "std"]).drop(columns="fold", level=0).round(3)
summary.to_csv(REP / "cv_summary.csv")
print("\n=== CV summary (training vehicles, mean and std over folds) ===")
print(summary.to_string())


# ---------------- 4. significance tests ----------------
def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0
    for rank, k in enumerate(order):
        run = max(run, (m - rank) * pvals[k]); adj[k] = min(1, run)
    return adj


lines = []
for fs in SELECTED:
    piv = cv[cv.feature_set == fs].pivot(index="fold", columns="model", values="rmse")
    chi, p = stats.friedmanchisquare(*[piv[c] for c in piv])
    lines.append(f"[{fs}] Friedman chi2={chi:.3f} p={p:.5f}")
    pairs = list(combinations(piv.columns, 2))
    pw = [stats.wilcoxon(piv[x], piv[y]).pvalue for x, y in pairs]
    pt = [stats.ttest_rel(piv[x], piv[y]).pvalue for x, y in pairs]
    for (x, y), w, wa, t, ta in zip(pairs, pw, holm(np.array(pw)), pt, holm(np.array(pt))):
        diff = (piv[x] - piv[y]).mean()
        lines.append(f"  {x} vs {y}: mean RMSE diff={diff:+.4f} | Wilcoxon p={w:.5f} (Holm {wa:.5f})"
                     f" | paired t p={t:.5f} (Holm {ta:.5f})")
# behaviour_only vs full for the best model family
for name in ["LightGBM", "RandomForest", "Ridge"]:
    b = cv[(cv.model == name) & (cv.feature_set == "behaviour_only")].sort_values("fold")["rmse"].to_numpy()
    f = cv[(cv.model == name) & (cv.feature_set == "full")].sort_values("fold")["rmse"].to_numpy()
    lines.append(f"{name}: behaviour_only vs full Wilcoxon p={stats.wilcoxon(b, f).pvalue:.5f}")
    for other in ["full_no_load", "full_epa"]:
        if other in SELECTED:
            n = cv[(cv.model == name) & (cv.feature_set == other)].sort_values("fold")["rmse"].to_numpy()
            lines.append(f"{name}: {other} vs full mean RMSE diff={np.mean(n - f):+.4f} "
                         f"Wilcoxon p={stats.wilcoxon(n, f).pvalue:.5f}")
print("\n=== Significance tests (per-fold RMSE) ===\n" + "\n".join(lines))
(REP / "significance.txt").write_text("\n".join(lines))

# ---------------- 5. final fit, held-out test, error analysis ----------------
best = cv.groupby(["feature_set", "model"])["rmse"].mean()
test_rows, meta = [], {"task": a.task, "target": target, "models": {}}
y_te = test[target].to_numpy()
for fs, cols in SELECTED.items():
    for name, model in make_models(PARAMS.get(fs)).items():
        t = time.perf_counter(); model.fit(train[cols].to_numpy(np.float32), y_tr); fit_s = time.perf_counter() - t
        t = time.perf_counter(); p = model.predict(test[cols].to_numpy(np.float32)); inf = time.perf_counter() - t
        m = metrics(y_te, p)
        test_rows.append({"feature_set": fs, "model": name, **m, "fit_s": fit_s,
                          "infer_ms_per_1k": inf / len(test) * 1e6})
        path = OUT / f"{a.task}_{fs}_{name}.joblib"
        joblib.dump({"model": model, "features": cols, "target": target, "task": a.task,
                     "feature_set": fs, "test_metrics": m,
                     "params": PARAMS.get(fs, {}).get(name, {})}, path)
        meta["models"][f"{fs}/{name}"] = {"path": str(path), "cv_rmse": float(best[(fs, name)]),
                                          "params": PARAMS.get(fs, {}).get(name, {}), **m}
        test[f"pred_{fs}_{name}"] = p

# baselines on the same held-out vehicles
t = time.perf_counter()
dummy = DummyRegressor(strategy="mean").fit(np.zeros((len(train), 1)), y_tr)
p = dummy.predict(np.zeros((len(test), 1)))
test_rows.append({"feature_set": "baseline", "model": "Dummy(mean)", **metrics(y_te, p),
                  "fit_s": time.perf_counter() - t, "infer_ms_per_1k": np.nan})
if "disp_L" in train:
    t = time.perf_counter()
    by_disp = train.groupby("disp_L")[target].median()
    p = test["disp_L"].map(by_disp).fillna(float(np.median(y_tr))).to_numpy()
    n_unseen = int((~test["disp_L"].isin(by_disp.index)).sum())
    test_rows.append({"feature_set": "baseline", "model": "median per disp_L", **metrics(y_te, p),
                      "fit_s": time.perf_counter() - t, "infer_ms_per_1k": np.nan})
    print(f"baseline median per disp_L: {n_unseen} test trips with an unseen disp_L -> global median")
meta["tuned_params"] = PARAMS
test_res = pd.DataFrame(test_rows).round(3)
test_res.to_csv(REP / "test_metrics.csv", index=False)
print("\n=== Held-out TEST vehicles ===")
print(test_res.to_string(index=False))

for fs in SELECTED:
    name = best.loc[fs].idxmin()
    meta[f"best_{fs}"] = f"{fs}/{name}"
(OUT / f"meta_{a.task}.json").write_text(json.dumps(meta, indent=2))
print("\nbest per feature set:", {k: v for k, v in meta.items() if k.startswith("best_")})

bf, bm = meta["best_full"].split("/")
test["pred"] = test[f"pred_{bf}_{bm}"]
test["abs_err"] = (test[target] - test["pred"]).abs()
test["sq_err"] = (test[target] - test["pred"]) ** 2
keep = ["vehid", target, "pred", "abs_err"] + [c for c in ["trip_key", "veh_class", "avg_speed", "speed",
                                                           "temp_c", "dist_km", "idle_frac"] if c in test]
test[keep + [c for c in test if c.startswith("pred_")]].to_parquet(REP / "test_predictions.parquet", index=False)

spd = "avg_speed" if a.task == "trip" else "speed"
bands = {
    "vehicle class": test.get("veh_class"),
    "speed band": pd.cut(test[spd], [-1, 1, 20, 40, 60, 80, 300]),
    "temperature band": pd.cut(test["temp_c"], [-40, 0, 10, 20, 40]) if "temp_c" in test else None,
    "trip length": pd.cut(test["dist_km"], [0, 3, 10, 30, 1000]) if "dist_km" in test else None,
}
ea_lines = [f"Error analysis on held-out vehicles, model = {bf}/{bm}"]
for label, key in bands.items():
    if key is None:
        continue
    t = test.groupby(key, observed=True).agg(n=(target, "size"), mean_target=(target, "mean"),
                                             mae=("abs_err", "mean"),
                                             rmse=("sq_err", lambda s: np.sqrt(s.mean())))
    ea_lines.append(f"\nby {label}:\n{t.round(3).to_string()}")
pv = test.groupby("vehid").agg(n=(target, "size"), mean_target=(target, "mean"),
                               mae=("abs_err", "mean")).sort_values("mae", ascending=False)
ea_lines.append(f"\nworst 10 vehicles:\n{pv.head(10).round(3).to_string()}")
(REP / "error_analysis.txt").write_text("\n".join(ea_lines))
print("\n" + "\n".join(ea_lines))

# ---------------- 6. optional external validation (unseen data source) ----------------
if a.external and a.task == "trip":
    ext = pd.read_parquet(a.external).dropna(subset=[target])
    lines = [f"External validation on {a.external}: {len(ext)} trips"]
    for fs in SELECTED:
        b = joblib.load(OUT / f"{a.task}_{fs}_{meta['best_' + fs].split('/')[1]}.joblib")
        p = b["model"].predict(mf.to_matrix(ext, b["features"]))
        m = metrics(ext[target].to_numpy(), p)
        bias = float(np.mean(p - ext[target]))
        lines.append(f"  {fs}: {json.dumps({k: round(v, 3) for k, v in m.items()})}  mean bias={bias:+.2f}")
    (REP / "external_validation.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))

print(f"\nsaved models in {OUT}/ and reports in {REP}/")