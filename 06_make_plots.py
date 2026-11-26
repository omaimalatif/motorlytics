"""
Matplotlib figures for the report (16, plus 2 real-world vs EPA figures when the trips have EPA ratings), saved to plots/, plus plots/plot_notes.txt
with the numbers behind each figure so you can write the interpretations.

    python 06_make_plots.py          (run after 02, 03 --task trip, 04; 03 --task row optional)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import motorlytics_features as mf

OUT = Path("plots"); OUT.mkdir(exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.prop_cycle": plt.cycler(color=C)})
notes = []
n = [0]


def save(fig, name, note):
    n[0] += 1
    fname = f"{n[0]:02d}_{name}.png"
    fig.savefig(OUT / fname); plt.close(fig)
    notes.append(f"{fname}\n  {note}\n")
    print("saved", fname)


def binned(x, y, bins):
    b = pd.cut(x, bins)
    g = pd.DataFrame({"b": b, "y": y}).groupby("b", observed=True)["y"]
    mids = [iv.mid for iv in g.mean().index]
    return np.array(mids), g.mean().to_numpy(), g.quantile(0.25).to_numpy(), g.quantile(0.75).to_numpy(), g.size().to_numpy()


T = pd.read_parquet(mf.PROC / "trips.parquet")
y = T["l_per_100km"]

# 1 target distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(y, bins=60, color=C[0], edgecolor="white", linewidth=0.5)
ax.axvline(y.median(), color=INK, lw=1.5, ls="--")
ax.text(y.median(), ax.get_ylim()[1] * 0.92, f"  median {y.median():.1f}", color=INK)
ax.set(title="Trip fuel consumption is right-skewed", xlabel="L/100 km", ylabel="Trips")
save(fig, "target_distribution", f"n={len(y)} trips; mean={y.mean():.2f}, median={y.median():.2f}, "
     f"std={y.std():.2f}, skew={y.skew():.2f}; 90th pct={y.quantile(.9):.2f}")

# 2 average speed vs consumption
fig, ax = plt.subplots(figsize=(7, 4))
hb = ax.hexbin(T["avg_speed"], y, gridsize=45, cmap="Blues", mincnt=1, bins="log")
m, mu, lo, hi, _ = binned(T["avg_speed"], y, np.arange(0, 125, 5))
ax.plot(m, mu, color=C[1], lw=2, label="mean per 5 km/h bin")
ax.legend(frameon=False); fig.colorbar(hb, ax=ax, label="trips (log)")
ax.set(title="Consumption falls with average speed, then flattens", xlabel="Average trip speed (km/h)", ylabel="L/100 km")
best = m[np.nanargmin(mu)] if len(mu) else np.nan
save(fig, "speed_vs_consumption", f"lowest mean L/100km at about {best:.0f} km/h avg speed; "
     f"corr={T['avg_speed'].corr(y):.2f}. Slow, stop-start trips are the least efficient.")

# 3-5 behaviour drivers
for col, xl, title in [("idle_frac", "Share of trip time idling", "More idling, more fuel per km"),
                       ("rpa", "Relative positive acceleration (m/s²)", "Aggressive acceleration costs fuel"),
                       ("stops_per_km", "Stops per km", "Stop-and-go traffic raises consumption")]:
    x = T[col].clip(upper=T[col].quantile(0.99))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=4, alpha=0.15, color=C[0], linewidths=0)
    m, mu, lo, hi, _ = binned(x, y, np.linspace(x.min(), x.max(), 16))
    ax.fill_between(m, lo, hi, color=C[1], alpha=0.15, linewidth=0)
    ax.plot(m, mu, color=C[1], lw=2, label="mean (band = middle 50%)")
    ax.legend(frameon=False); ax.set(title=title, xlabel=xl, ylabel="L/100 km", ylim=(0, y.quantile(0.995)))
    save(fig, f"{col}_vs_consumption", f"Spearman rho={T[col].corr(y, method='spearman'):.2f}; "
         f"mean L/100km in top vs bottom quartile of {col}: "
         f"{y[T[col] >= T[col].quantile(.75)].mean():.2f} vs {y[T[col] <= T[col].quantile(.25)].mean():.2f}")

# 6 temperature
tcol = "temp_c" if "temp_c" in T and T["temp_c"].notna().any() else "oat_mean"
if tcol in T and T[tcol].notna().any():
    m, mu, lo, hi, cnt = binned(T[tcol], y, np.arange(-25, 40, 5))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(m, lo, hi, color=C[0], alpha=0.15, linewidth=0)
    ax.plot(m, mu, "o-", color=C[0], lw=2, ms=6)
    ax.set(title="Cold weather increases consumption", xlabel="Air temperature at trip start (°C)", ylabel="L/100 km")
    save(fig, "temperature_vs_consumption", f"{tcol}: mean L/100km below 0°C = {y[T[tcol] < 0].mean():.2f}, "
         f"above 20°C = {y[T[tcol] > 20].mean():.2f}; Spearman rho={T[tcol].corr(y, method='spearman'):.2f}")

# 7 vehicle class
if "veh_class" in T:
    cls = T.groupby("veh_class")["l_per_100km"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot([T.loc[T.veh_class == c, "l_per_100km"] for c in cls], showfliers=False,
                    patch_artist=True, widths=0.5)
    for b in bp["boxes"]: b.set(facecolor="#cde2fb", edgecolor=C[0])
    for md in bp["medians"]: md.set(color=INK, lw=1.5)
    ax.set_xticks(range(1, len(cls) + 1), cls)
    ax.set(title="Consumption by vehicle class", ylabel="L/100 km")
    med = T.groupby("veh_class")["l_per_100km"].agg(["median", "size"])
    save(fig, "consumption_by_class", "median and trips per class: " + med.round(2).to_dict("index").__repr__())

# 8 engine size
fig, ax = plt.subplots(figsize=(7, 4))
g = T.groupby(T["disp_L"].round(1))["l_per_100km"].agg(["median", "size"])
g = g[g["size"] >= 20]
ax.bar(g.index.astype(str), g["median"], color=C[0], width=0.6)
ax.set(title="Bigger engines use more fuel", xlabel="Engine displacement (L)", ylabel="Median L/100 km")
save(fig, "consumption_by_engine_size", f"Spearman rho(disp_L, L/100km)={T['disp_L'].corr(y, method='spearman'):.2f}; "
     f"note how few VED cars are below 1.5 L (Pakistani cars are often 0.66 to 1.3 L)")

# 9 correlation heatmap
cols = [c for c in ["l_per_100km", "avg_speed", "idle_frac", "stops_per_km", "rpa", "hard_accel_per_km",
                    "speed_std", "frac_speed_gt90", "mean_rpm", "rpm_p90", "disp_L", "temp_c", "dist_km"]
        if c in T and T[c].notna().any()]
corr = T[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right"); ax.set_yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else INK)
ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
ax.set_title("Feature correlations (trip level)")
top = corr["l_per_100km"].drop("l_per_100km").abs().sort_values(ascending=False).head(5)
save(fig, "correlation_heatmap", f"strongest with target: {top.round(2).to_dict()}")

# 10 feature selection
fsp = mf.PROC / "results_trip" / "feature_selection_full.csv"
if fsp.exists():
    s = pd.read_csv(fsp, index_col=0).sort_values("perm_rmse_increase")
    s = s.tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = [C[0] if v else "#b8b6af" for v in s["selected"]]
    ax.barh(s.index, s["perm_rmse_increase"], color=colors, height=0.6)
    ax.set(title="Permutation importance (blue = selected by ≥2 of 3 methods)",
           xlabel="RMSE increase when shuffled (L/100 km)")
    save(fig, "feature_importance", f"top 5: {s['perm_rmse_increase'].tail(5)[::-1].round(3).to_dict()}; "
         f"selected {int(s['selected'].sum())} shown")

# 11 model comparison
cvp = mf.PROC / "results_trip" / "cv_folds.csv"
if cvp.exists():
    cv = pd.read_csv(cvp)
    agg = cv.groupby(["model", "feature_set"])["rmse"].agg(["mean", "std"]).reset_index()
    course = [m for m in ["DecisionTree", "MLP", "kNN", "LWR"] if m in set(agg["model"])]
    reference = [m for m in ["Ridge", "RandomForest", "LightGBM"] if m in set(agg["model"])]
    models = course + reference
    sets = [s for s in ["behaviour_only", "full_no_load", "full"] if s in set(agg["feature_set"])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(sets)
    x = np.arange(len(models))
    if reference:                                       # shade the reference block
        ax.axvspan(len(course) - 0.5, len(models) - 0.5, color="#f1efe8", zorder=0)
        ax.text(len(course) - 0.4, 0.98, "reference models (beyond the syllabus)", transform=ax.get_xaxis_transform(),
                va="top", fontsize=9, color=INK2)
        ax.text(-0.4, 0.98, "course models", transform=ax.get_xaxis_transform(), va="top", fontsize=9,
                color=INK, fontweight="bold")
    for k, fs in enumerate(sets):
        r = agg[agg.feature_set == fs].set_index("model").reindex(models)
        ax.bar(x + (k - (len(sets) - 1) / 2) * w, r["mean"], w * 0.92, yerr=r["std"], capsize=2,
               color=C[k], label=fs.replace("_", " "), error_kw={"ecolor": INK2, "lw": 0.8}, zorder=2)
    ax.set_xticks(x, models)
    ax.set_ylim(0, agg["mean"].max() * 1.35)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.set(title="10-fold grouped CV: RMSE by model (course vs reference)", ylabel="RMSE (L/100 km), lower is better")
    best_c = agg[agg.model.isin(course)].sort_values("mean").groupby("feature_set").head(1)
    best_r = agg[agg.model.isin(reference)].sort_values("mean").groupby("feature_set").head(1)
    save(fig, "model_comparison", "best course model per feature set: "
         + ", ".join(f"{r.feature_set} {r.model} {r['mean']:.3f}" for _, r in best_c.iterrows())
         + "; best reference: " + ", ".join(f"{r.feature_set} {r.model} {r['mean']:.3f}" for _, r in best_r.iterrows())
         + "\n  " + agg.round(3).to_string(index=False).replace("\n", "\n  "))

# 12 predicted vs actual + 13 residuals by speed band
tp = mf.PROC / "results_trip" / "test_predictions.parquet"
if tp.exists():
    P = pd.read_parquet(tp)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = (0, P["l_per_100km"].quantile(0.995))
    ax.scatter(P["l_per_100km"], P["pred"], s=5, alpha=0.25, color=C[0], linewidths=0)
    ax.plot(lim, lim, color=INK, lw=1, ls="--")
    ax.set(xlim=lim, ylim=lim, title="Held-out vehicles: predicted vs actual",
           xlabel="Actual L/100 km", ylabel="Predicted L/100 km")
    rmse = np.sqrt(((P["l_per_100km"] - P["pred"]) ** 2).mean())
    save(fig, "predicted_vs_actual", f"test RMSE={rmse:.2f}, MAE={P['abs_err'].mean():.2f} on "
         f"{P['vehid'].nunique()} unseen vehicles; points below the line are under-predictions")
    P["band"] = pd.cut(P["avg_speed"], [0, 20, 40, 60, 80, 200])
    bands = [b for b in P["band"].cat.categories if (P["band"] == b).any()]
    fig, ax = plt.subplots(figsize=(7, 4))
    res = [(P.loc[P.band == b, "pred"] - P.loc[P.band == b, "l_per_100km"]) for b in bands]
    bp = ax.boxplot(res, showfliers=False, patch_artist=True, widths=0.5)
    for b in bp["boxes"]: b.set(facecolor="#cde2fb", edgecolor=C[0])
    for md in bp["medians"]: md.set(color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(1, len(bands) + 1), [str(b) for b in bands])
    ax.set(title="Errors are largest on slow, congested trips", xlabel="Average speed band (km/h)",
           ylabel="Prediction − actual (L/100 km)")
    save(fig, "residuals_by_speed", "MAE by band: " + P.groupby("band", observed=True)["abs_err"].mean().round(2).to_dict().__repr__())

# 14 driving styles
sp = mf.PROC / "trip_styles.parquet"
if sp.exists():
    S = T.merge(pd.read_parquet(sp)[["trip_key", "style"]], on="trip_key")
    order = [s for s in ["eco", "normal", "aggressive"] if s in S["style"].unique()]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for k, s in enumerate(order):
        d = S[S["style"] == s]
        axs[0].scatter(d["rpa"], d["hard_accel_per_km"], s=4, alpha=0.25, color=C[k], label=s, linewidths=0)
    axs[0].set(xlabel="RPA (m/s²)", ylabel="Hard accelerations per km", title="Driving-style clusters (K-Means)",
               xlim=(0, S["rpa"].quantile(0.99)), ylim=(0, S["hard_accel_per_km"].quantile(0.99)))
    leg = axs[0].legend(frameon=False, markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    med = S.groupby("style")["l_per_100km"].median().reindex(order)
    axs[1].bar(order, med, color=[C[k] for k in range(len(order))], width=0.55)
    for i, v in enumerate(med):
        axs[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", color=INK)
    axs[1].set(title="Median consumption per style", ylabel="L/100 km")
    save(fig, "driving_styles", f"median L/100km per style: {med.round(2).to_dict()}; "
         f"share: {S['style'].value_counts(normalize=True).round(2).to_dict()}")

# 15 seasonality
if "month" in T and T["month"].notna().any():
    g = T.groupby("month")["l_per_100km"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g.index, g.values, "o-", color=C[0], lw=2, ms=6)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set(title="Winter months use more fuel", xlabel="Month", ylabel="Mean L/100 km")
    save(fig, "monthly_consumption", f"monthly means: {g.round(2).to_dict()}")

# 16 row level: instantaneous fuel rate vs speed and acceleration
rp = mf.PROC / "rows.parquet"
if rp.exists():
    R = pd.read_parquet(rp, columns=["speed", "accel", "fuel_Lph"]).dropna()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hexbin(R["speed"], R["fuel_Lph"], gridsize=50, cmap="Blues", mincnt=1, bins="log")
    axs[0].set(title="Fuel rate vs speed (each 1 s sample)", xlabel="Speed (km/h)", ylabel="Fuel rate (L/h)")
    m, mu, lo, hi, _ = binned(R["accel"].clip(-4, 4), R["fuel_Lph"], np.linspace(-4, 4, 17))
    axs[1].fill_between(m, lo, hi, color=C[1], alpha=0.15, linewidth=0)
    axs[1].plot(m, mu, "o-", color=C[1], lw=2, ms=5)
    axs[1].set(title="Fuel rate rises steeply with acceleration", xlabel="Acceleration (m/s²)", ylabel="Fuel rate (L/h)")
    save(fig, "row_level_fuel_rate", f"{len(R):,} samples; fuel rate at accel>2 m/s² = "
         f"{R.loc[R.accel > 2, 'fuel_Lph'].mean():.2f} L/h vs cruising (|a|<0.3) = "
         f"{R.loc[R.accel.abs() < 0.3, 'fuel_Lph'].mean():.2f} L/h")

# 17-18 real-world vs EPA ratings (needs 07_join_epa.py + 02_build_datasets.py --epa)
if "epa_comb_l100" in T and T["epa_comb_l100"].notna().any():
    E = T[T["epa_comb_l100"].notna()].copy()
    E["road_type"] = mf.road_type(E["avg_speed"]).to_numpy()
    # gap % = (measured - EPA) / EPA; road-type version compares city trips with the EPA city
    # rating, highway trips with the EPA highway rating, mixed trips with EPA combined
    E["gap_comb"] = (E["l_per_100km"] / E["epa_comb_l100"] - 1) * 100
    ref = np.select([E["road_type"] == "city", E["road_type"] == "highway"],
                    [E["epa_city_l100"], E["epa_hwy_l100"]], E["epa_comb_l100"])
    E["gap_road"] = (E["l_per_100km"] / ref - 1) * 100

    # 17 vehicle-level scatter: median measured vs EPA combined
    V = E.groupby("vehid").agg(measured=("l_per_100km", "median"), epa=("epa_comb_l100", "first"),
                               trips=("l_per_100km", "size"))
    V = V[V["trips"] >= 5]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(V["epa"], V["measured"], s=np.clip(V["trips"], 10, 120), alpha=0.55, color=C[0],
               edgecolor="white", linewidth=0.5)
    lim = [min(V["epa"].min(), V["measured"].min()) * 0.9, max(V["epa"].max(), V["measured"].max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=1.2, ls="--")
    ax.text(lim[1], lim[1], " y = x", color=INK2, va="center")
    ax.set(xlim=lim, ylim=lim, title="Real-world consumption vs EPA rating, per vehicle",
           xlabel="EPA combined rating (L/100 km, median of matching engines)",
           ylabel="Measured median (L/100 km)")
    vg = (V["measured"] / V["epa"] - 1) * 100
    r = V["measured"].corr(V["epa"])
    save(fig, "real_world_vs_epa", f"{len(V)} vehicles with >=5 trips (point size = trips); "
         f"Pearson r={r:.2f}; vehicle median gap={vg.median():+.1f}% (IQR {vg.quantile(.25):+.1f} to "
         f"{vg.quantile(.75):+.1f}); {(vg > 0).mean():.0%} of vehicles above their EPA rating. "
         "EPA values are the median of all EPA engines with the same displacement/cylinders/boost, "
         "so they rate the engine type, not the exact car.")

    # 18 median gap by road type and by driving style
    road = E.groupby("road_type")["gap_road"].agg(["median", "size"]).reindex(mf.ROAD_TYPES)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs[0].bar(road.index, road["median"], color=C[:3], width=0.55)
    for i, (v, k) in enumerate(zip(road["median"], road["size"])):
        axs[0].text(i, v, f"{v:+.0f}%\nn={k}", ha="center", va="bottom" if v >= 0 else "top", color=INK)
    axs[0].axhline(0, color=INK2, lw=1)
    axs[0].set(title="Gap vs EPA by road type", ylabel="Median gap, measured vs EPA (%)",
               xlabel="city vs EPA city, mixed vs combined, highway vs EPA highway")
    note_style = ""
    sp = mf.PROC / "trip_styles.parquet"
    if sp.exists():
        E = E.merge(pd.read_parquet(sp)[["trip_key", "style"]], on="trip_key", how="left")
        order = [s for s in ["eco", "normal", "aggressive"] if s in set(E["style"].dropna())]
        sty = E.groupby("style")["gap_comb"].agg(["median", "size"]).reindex(order)
        axs[1].bar(sty.index, sty["median"], color=[C[2], C[0], C[1]][:len(order)], width=0.55)
        for i, (v, k) in enumerate(zip(sty["median"], sty["size"])):
            axs[1].text(i, v, f"{v:+.0f}%\nn={k}", ha="center", va="bottom" if v >= 0 else "top", color=INK)
        axs[1].axhline(0, color=INK2, lw=1)
        axs[1].set(title="Gap vs EPA combined by style", xlabel="Driving style (K-Means)")
        note_style = "; by style (vs EPA combined): " + ", ".join(
            f"{s} {v:+.1f}% (n={k})" for s, v, k in zip(sty.index, sty["median"], sty["size"]))
    lo, hi = axs[0].get_ylim()                       # room for the value labels above/below bars
    pad = (hi - lo) * 0.25
    axs[0].set_ylim(min(lo, 0) - pad, max(hi, 0) + pad)
    ov = E["gap_comb"]
    save(fig, "epa_gap_by_road_and_style",
         f"trip-level gap % = (measured - EPA) / EPA x 100, {len(E):,} trips with an EPA match. "
         f"Overall median vs EPA combined {ov.median():+.1f}% (IQR {ov.quantile(.25):+.1f} to "
         f"{ov.quantile(.75):+.1f}); by road type: " + ", ".join(
             f"{r_} {v:+.1f}% (n={k})" for r_, v, k in zip(road.index, road["median"], road["size"]))
         + note_style)

(OUT / "plot_notes.txt").write_text("\n".join(notes))
print(f"\n{n[0]} plots in {OUT}/, numbers for interpretation in {OUT / 'plot_notes.txt'}")