"""
Motorlytics baseline: predict fuel rate (L/h) from driving features.
Run from your project root:  python train_baseline.py
Needs: pip install pandas numpy scikit-learn lightgbm scipy joblib
"""
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- settings ----------------
DATA = Path("Data") / "ved_features.pkl"
OUT = Path("models")
N_VEHICLES = 150        # vehicles sampled
MAX_ROWS = 2000         # target rows per vehicle (strided, so up to ~2x)
N_FOLDS = 5
SEED = 42
AFR = 14.7              # gasoline air-fuel ratio
G_PER_L = 737           # gasoline density, g/L

OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------- load + target ----------------
df = pd.read_pickle(DATA)
print("loaded", df.shape)

# maf is g/s  ->  fuel rate in litres per hour
df["fuel_Lph"] = df["maf"] * 3600 / (AFR * G_PER_L)

# ---------------- cleaning ----------------
n0 = len(df)
df = df[df["maf"].notna() & df["speed"].notna() & df["rpm"].notna()]
df = df[(df["dt"] > 0) & (df["dt"] <= 5)]          # drop logging gaps
df = df[df["maf"] >= 0]
cap = df["fuel_Lph"].quantile(0.999)                # cap extreme sensor spikes
df = df[df["fuel_Lph"] <= cap]
print(f"cleaning kept {len(df):,} of {n0:,} rows (fuel cap {cap:.1f} L/h)")

# ---------------- sample vehicles, then thin rows per vehicle ----------------
ids = df["VehId"].unique()
chosen = rng.choice(ids, size=min(N_VEHICLES, len(ids)), replace=False)
df = df[df["VehId"].isin(chosen)].sort_index()
n = df.groupby("VehId")["VehId"].transform("size")
step = (n // MAX_ROWS).clip(lower=1)
df = df[(df.groupby("VehId").cumcount() % step) == 0].copy()
print(f"sample: {df['VehId'].nunique()} vehicles, {len(df):,} rows")

# ---------------- extra features ----------------
df["accel_ms2"] = df["accel"] / 3.6                 # km/h/s -> m/s^2
df["accel_abs_ms2"] = df["accel_abs"] / 3.6
df["power_proxy"] = (df["speed"] / 3.6) * df["accel_ms2"]
df["speed_cubed"] = df["speed"] ** 3

BEHAVIOUR = ["speed", "speed_sq", "speed_cubed", "accel_ms2", "accel_abs_ms2",
             "power_proxy", "is_idle", "is_highway", "speed_roll5",
             "speed_std5", "disp_L", "dt"]
WITH_RPM = BEHAVIOUR + ["rpm", "rpm_per_kmh", "rpm_change", "rpm_roll5"]
FEATURE_SETS = {"behaviour_only": BEHAVIOUR, "with_rpm": WITH_RPM}
# note: maf is NOT a feature, the target is derived from it (would leak)

print("\ntarget fuel_Lph:\n", df["fuel_Lph"].describe().round(3))


def make_models():
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"),
                               StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=100, max_depth=16,
                                  min_samples_leaf=20, n_jobs=-1,
                                  random_state=SEED)),
        "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  num_leaves=63, subsample=0.8,
                                  colsample_bytree=0.8, n_jobs=-1,
                                  random_state=SEED, verbose=-1),
    }


# ---------------- cross-validation grouped by vehicle ----------------
y = df["fuel_Lph"].to_numpy()
groups = df["VehId"].to_numpy()
gkf = GroupKFold(n_splits=N_FOLDS)
rows = []

for fs_name, cols in FEATURE_SETS.items():
    X = df[cols].to_numpy(dtype=np.float32)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        for m_name, model in make_models().items():
            t0 = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = model.predict(X[te])
            inf_ms_per_1k = (time.perf_counter() - t0) / len(te) * 1000 * 1000
            rows.append(dict(
                feature_set=fs_name, model=m_name, fold=fold,
                rmse=np.sqrt(mean_squared_error(y[te], pred)),
                mae=mean_absolute_error(y[te], pred),
                r2=r2_score(y[te], pred),
                fit_s=fit_s, infer_ms_per_1k_rows=inf_ms_per_1k))
        print(f"{fs_name}: fold {fold + 1}/{N_FOLDS} done")

res = pd.DataFrame(rows)
res.to_csv("cv_results.csv", index=False)

summary = (res.groupby(["feature_set", "model"])
              [["rmse", "mae", "r2", "fit_s", "infer_ms_per_1k_rows"]]
              .agg(["mean", "std"]).round(3))
print("\n=== CV summary (mean, std over folds) ===")
print(summary.to_string())

# ---------------- significance tests on per-fold RMSE ----------------
print("\n=== Significance tests (per-fold RMSE) ===")
for fs_name in FEATURE_SETS:
    piv = (res[res.feature_set == fs_name]
           .pivot(index="fold", columns="model", values="rmse"))
    chi, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
    print(f"\n[{fs_name}] Friedman chi2={chi:.3f}, p={p:.4f}")
    for a, b in combinations(piv.columns, 2):
        t, pt = stats.ttest_rel(piv[a], piv[b])
        print(f"  paired t-test {a} vs {b}: t={t:.3f}, p={pt:.4f}")
print("(with 5 folds, Wilcoxon cannot go below p=0.0625; use 10 folds for it)")

# ---------------- final fit for the app + importances ----------------
for fs_name, cols in FEATURE_SETS.items():
    X = df[cols].to_numpy(dtype=np.float32)
    final = make_models()["LightGBM"].fit(X, y)
    joblib.dump({"model": final, "features": cols}, OUT / f"lgbm_{fs_name}.joblib")
    imp = pd.Series(final.feature_importances_, index=cols).sort_values(ascending=False)
    print(f"\nLightGBM importances [{fs_name}]:\n{imp.to_string()}")

print("\nsaved: cv_results.csv, models/lgbm_*.joblib")