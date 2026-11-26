"""
Driving-style profiles with K-Means (unsupervised), used for the eco-score and tips.

    python 04_driving_styles.py            # k = 3 (eco / normal / aggressive)

Clusters only on driving-behaviour features (rpa, hard accelerations/brakes per km, mean
positive acceleration). Idle share, stops, speed variation and RPM mostly reflect road type
and traffic, so they are not used for clustering.

Each trip also gets a road type from avg_speed: city (< 30 km/h), mixed (30-60), highway (> 60).
Saves models/driving_style.joblib with the scaler, k-means model, cluster->label map and the
eco profile per road type (median of eco trips on that road type) that recommend() compares
users to.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import motorlytics_features as mf

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()

trips = pd.read_parquet(mf.PROC / "trips.parquet")
# higher = more aggressive for all four, so the mean standardised centroid ranks clusters
STYLE = [c for c in ["rpa", "hard_accel_per_km", "hard_brake_per_km", "mean_pos_accel"]
         if c in trips and trips[c].notna().mean() > 0.8]
CONTEXT = [c for c in ["idle_frac", "stops_per_km", "frac_rpm_gt2500", "speed_std"] if c in trips]
X = trips[STYLE].replace([np.inf, -np.inf], np.nan)
ok = X.notna().all(axis=1)
X = X[ok]
# winsorise so a handful of extreme trips do not own a cluster
X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
scaler = StandardScaler().fit(X)
Z = scaler.transform(X)

rng = np.random.default_rng(42)
sub = rng.choice(len(Z), size=min(10_000, len(Z)), replace=False)
print("silhouette by k (higher = better separated):")
for k in range(2, 7):
    lab = KMeans(k, n_init=10, random_state=42).fit_predict(Z[sub])
    print(f"  k={k}: {silhouette_score(Z[sub], lab):.3f}")

km = KMeans(a.k, n_init=20, random_state=42).fit(Z)
lab = km.labels_
# order clusters by aggressiveness = mean standardised centroid value
score = km.cluster_centers_.mean(axis=1)
order = np.argsort(score)
names = ["eco", "normal", "aggressive"] if a.k == 3 else [f"style_{i}" for i in range(a.k)]
label_map = {int(c): names[r] for r, c in enumerate(order)}

t = trips[ok].copy()
t["style"] = [label_map[int(c)] for c in lab]
t["road_type"] = mf.road_type(t["avg_speed"]).to_numpy()
order_s = [n for n in names if n in set(t["style"])]
prof = t.groupby("style")[STYLE + CONTEXT + ["l_per_100km", "avg_speed"]].median().reindex(order_s)
prof["share_of_trips"] = t["style"].value_counts(normalize=True)
print("\nmedian profile per style:")
print(prof.round(3).to_string())

ct = pd.crosstab(t["style"], t["road_type"], margins=True).reindex(
    index=order_s + ["All"], columns=mf.ROAD_TYPES + ["All"])
print("\nstyle vs road type (trips):")
print(ct.to_string())
print("\nstyle vs road type (share of each road type):")
print(pd.crosstab(t["style"], t["road_type"], normalize="columns")
      .reindex(index=order_s, columns=mf.ROAD_TYPES).round(3).to_string())
print("\nmedian L/100 km by style and road type:")
print(t.pivot_table(index="style", columns="road_type", values="l_per_100km", aggfunc="median")
      .reindex(index=order_s, columns=mf.ROAD_TYPES).round(2).to_string())

groups = [g["l_per_100km"].to_numpy() for _, g in t.groupby("style")]
h, p = stats.kruskal(*groups)
print(f"\nKruskal-Wallis on L/100km across styles: H={h:.1f}, p={p:.2e}")

# ---------------- EM clustering (Gaussian mixture) next to k-Means ----------------
# Same standardised behaviour features. k-Means stays the app's style model; EM is the comparison.
em_lines = ["EM (GaussianMixture, full covariance) vs k-Means on the same standardised features",
            f"features: {STYLE}; {len(Z):,} trips", "", "k   silhouette k-Means   silhouette EM   EM BIC"]
for k in range(2, 7):
    lk = KMeans(k, n_init=10, random_state=42).fit_predict(Z[sub])
    g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=42).fit(Z[sub])
    le = g.predict(Z[sub])
    s_em = silhouette_score(Z[sub], le) if len(set(le)) > 1 else np.nan
    em_lines.append(f"{k}   {silhouette_score(Z[sub], lk):.3f}               {s_em:.3f}           {g.bic(Z[sub]):,.0f}")
gm = GaussianMixture(a.k, covariance_type="full", n_init=5, random_state=42).fit(Z)
lab_em = gm.predict(Z)
em_order = np.argsort(gm.means_.mean(axis=1))          # same ranking rule as k-Means
em_map = {int(c): names[r] for r, c in enumerate(em_order)}
t["style_em"] = [em_map[int(c)] for c in lab_em]
ari = adjusted_rand_score(t["style"], t["style_em"])
size = pd.DataFrame({"k-Means trips": t["style"].value_counts(), "EM trips": t["style_em"].value_counts(),
                     "k-Means median L/100km": t.groupby("style")["l_per_100km"].median(),
                     "EM median L/100km": t.groupby("style_em")["l_per_100km"].median()}).reindex(order_s)
size["k-Means share"] = size["k-Means trips"] / len(t)
size["EM share"] = size["EM trips"] / len(t)
h_em, p_em = stats.kruskal(*[g_["l_per_100km"].to_numpy() for _, g_ in t.groupby("style_em")])
em_lines += ["", f"k={a.k}: silhouette k-Means {silhouette_score(Z[sub], lab[sub]):.3f}, "
             f"EM {silhouette_score(Z[sub], lab_em[sub]):.3f}",
             f"adjusted Rand index (k-Means vs EM labels): {ari:.3f}",
             f"Kruskal-Wallis on L/100km across EM clusters: H={h_em:.1f}, p={p_em:.2e}", "",
             "cluster sizes and fuel:", size.round(3).to_string(), "",
             "k-Means (rows) vs EM (columns), trips:",
             pd.crosstab(t["style"], t["style_em"]).reindex(index=order_s, columns=order_s).to_string(), "",
             "EM median profile:",
             t.groupby("style_em")[STYLE + ["l_per_100km", "avg_speed"]].median().reindex(order_s).round(3).to_string()]
print("\n" + "\n".join(em_lines))
(mf.PROC / "style_em_comparison.txt").write_text("\n".join(em_lines), encoding="utf-8")

eco_name = names[0]
eco = t[t["style"] == eco_name]
# median = what-if target for tips; 75th percentile = threshold for flagging a feature
# (many eco medians are 0, e.g. hard brakes, so "above median" would flag almost everyone)
eco_profile = {"all": eco.median(numeric_only=True).to_dict()}
eco_p75 = {"all": eco.quantile(0.75, numeric_only=True).to_dict()}
print("\neco profile per road type (used for tips):")
for rt in mf.ROAD_TYPES:
    e = eco[eco["road_type"] == rt]
    if len(e) >= 20:
        eco_profile[rt] = e.median(numeric_only=True).to_dict()
        eco_p75[rt] = e.quantile(0.75, numeric_only=True).to_dict()
        print(f"  {rt:8s} p75      " + "  ".join(
            f"{c}={eco_p75[rt][c]:.3f}" for c in STYLE + CONTEXT + ["l_per_100km"]))
        print(f"  {rt:8s} n={len(e):5d}  " + "  ".join(
            f"{c}={eco_profile[rt][c]:.3f}" for c in STYLE + CONTEXT + ["l_per_100km"]))
    else:
        print(f"  {rt:8s} n={len(e):5d}  too few eco trips, falls back to the overall eco profile")
mf.MODELS.mkdir(exist_ok=True)
joblib.dump({"scaler": scaler, "kmeans": km, "features": STYLE, "label_map": label_map,
             "clip_lo": X.quantile(0.01).to_dict(), "clip_hi": X.quantile(0.99).to_dict(),
             "eco_profile": eco_profile, "eco_p75": eco_p75, "profile": prof}, mf.MODELS / "driving_style.joblib")
t[["trip_key", "vehid", "style", "style_em", "road_type"]].to_parquet(mf.PROC / "trip_styles.parquet", index=False)
prof.to_csv(mf.PROC / "style_profiles.csv")
print(f"\nsaved {mf.MODELS / 'driving_style.joblib'}")