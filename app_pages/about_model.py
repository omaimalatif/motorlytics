"""How the predictions are made, and how well they did (models/meta_trip.json and Data/processed/results_trip)."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import motorlytics_features as mf
from ui import components as ui
from ui import session
from ui import theme

session.require_user()
ui.page_header("About the model", "Course models (Decision Tree, ANN, k-NN, LWR) trained on the Vehicle Energy "
               "Dataset (Ann Arbor, 2017-18), tested on unseen vehicles and on external data.")
REP = mf.PROC / "results_trip"
meta_p = mf.MODELS / "meta_trip.json"
meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
tiers = session.MODEL_TYPES
NAMES = {"MLP": "MLP (ANN)", "kNN": "k-NN", "DecisionTree": "Decision Tree", "RandomForest": "Random Forest"}

# ---------------- the app model per input type ----------------
cols = st.columns(3, gap="medium")
for c, (fs, label) in zip(cols, tiers.items()):
    with c:
        with ui.card(label, kind="hero"):
            key = meta.get(f"best_{fs}", "–")
            m = meta.get("models", {}).get(key, {})
            with st.container(key=theme.next_key("mlkpi-accent")):
                st.metric("App model", NAMES.get(key.split("/")[-1], key.split("/")[-1]))
            if m:
                st.caption(f"Held-out test: MAPE {m['mape']:.1f}% · RMSE {m['rmse']:.2f} L/100 km · R² {m['r2']:.2f}")
            ref = meta.get(f"best_reference_{fs}")
            if ref:
                st.caption(f"Reference (beyond the syllabus): {NAMES.get(ref.split('/')[-1], ref.split('/')[-1])}")
if meta.get("selection"):
    st.caption(":material/rule: Choice rule: " + meta["selection"]["rule"] + ".")

# ---------------- error comparison: course vs reference models ----------------
models = pd.DataFrame([{"tier": k.split("/")[0], "model": k.split("/")[1], **v} for k, v in meta.get("models", {}).items()
                       if "mape" in v])
with ui.card("Model error on unseen vehicles", "MAPE on the held-out VED vehicles (lower is better). Course models in "
             "orange, reference models in teal; the outlined bar is the model the app uses."):
    if models.empty:
        st.info("Run 03_train.py to create the results.", icon=":material/info:")
    else:
        tier = st.segmented_control("Input type", list(tiers), default="behaviour_only", format_func=tiers.get,
                                    key="about_tier") or "behaviour_only"
        d = models[models["tier"] == tier].sort_values("mape", ascending=False)
        app = meta.get(f"best_{tier}", "").split("/")[-1]
        fig = go.Figure()
        for group, color, name in [("course", theme.CATEGORICAL[0], "course model"),
                                   ("reference", theme.CATEGORICAL[1], "reference model")]:
            g = d[d["group"] == group]
            fig.add_trace(go.Bar(
                y=[NAMES.get(x, x) for x in g["model"]], x=g["mape"], orientation="h", name=name, marker_color=color,
                marker_line=dict(width=[2.5 if x == app else 0 for x in g["model"]], color=theme.TEXT),
                text=[f"{v:.1f}%" + ("  · app model" if x == app else "") for x, v in zip(g["model"], g["mape"])],
                textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}% MAPE<extra></extra>"))
        fig.update_yaxes(categoryorder="array", categoryarray=[NAMES.get(x, x) for x in d["model"]])
        fig.update_xaxes(title="MAPE %", range=[0, float(d["mape"].max()) * 1.3], showgrid=True)
        fig.update_layout(barmode="overlay", meta="model_errors")
        theme.plot(fig, 330)

# ---------------- external validation ----------------
sel = REP / "model_selection_table.csv"
with ui.card("External validation: 21 European cars (enviroCar)",
             "MAPE before and after per-car calibration with 3 trips, for the course models."):
    if sel.exists():
        d = pd.read_csv(sel)
        tier2 = st.session_state.get("about_tier") or "behaviour_only"
        g = d[(d["tier"] == tier2) & (d["group"] == "course")]
        fig = go.Figure()
        for col, color, name in [("envirocar_mape", theme.CATEGORICAL[0], "before calibration"),
                                 ("envirocar_mape_cal3", theme.CATEGORICAL[1], "after 3-trip calibration")]:
            fig.add_trace(go.Bar(x=[NAMES.get(x, x) for x in g["model"]], y=g[col], name=name, marker_color=color,
                                 text=[f"{v:.1f}%" for v in g[col]], textposition="outside", cliponaxis=False,
                                 hovertemplate="%{x}: %{y:.1f}% MAPE<extra></extra>"))
        fig.update_yaxes(title="MAPE %", range=[0, float(g[["envirocar_mape", "envirocar_mape_cal3"]].max().max()) * 1.25])
        fig.update_layout(barmode="group", meta="external_validation")
        theme.plot(fig, 300)
        st.caption(f"Shown for: {tiers[tier2]}. Calibration multiplies each car's predictions by its own factor k.")
    else:
        st.caption("Run 09_generalisation.py to create the external-validation table; the text report is below.")

tab1, tab2, tab3, tab4 = st.tabs(["Test results", "Model choice", "Generalisation", "Figures"])
with tab1:
    p = REP / "test_metrics.csv"
    if p.exists():
        d = pd.read_csv(p).replace({"feature_set": {**session.MODEL_TYPES, "baseline": "Baseline"}})
        d = d.rename(columns={"feature_set": "Data", "group": "Group", "model": "Model", "rmse": "RMSE", "mae": "MAE",
                              "r2": "R²", "mape": "MAPE %", "fit_s": "Fit (s)", "infer_ms_per_1k": "ms per 1,000 trips"})
        st.dataframe(d, hide_index=True, width="stretch",
                     column_config={"MAPE %": st.column_config.ProgressColumn("MAPE %", min_value=0, max_value=25,
                                                                              format="%.1f")})
        st.caption("Held-out VED vehicles (never seen in training). Baselines: mean, and median per engine size.")
    else:
        st.info("Run 03_train.py to create the results.")
with tab2:
    p = REP / "model_selection_table.csv"
    if p.exists():
        d = pd.read_csv(p).round(2).replace({"tier": session.MODEL_TYPES})
        d.columns = [session.friendly(c).replace("_", " ") for c in d.columns]
        st.dataframe(d, hide_index=True, width="stretch")
        st.caption("MAPE %: VED test, enviroCar (21 European cars) before / after 3-trip calibration, exp1 (3 cars).")
    else:
        st.info("Run 09_generalisation.py to create this table.")
with tab3:
    for name in ["generalisation.txt", "external_exp1.txt"]:
        p = REP / name
        if p.exists():
            with st.expander(name):
                st.code(session.friendly(p.read_text(encoding="utf-8")), language=None)
with tab4:
    figs = [Path("plots") / f for f in ["11_model_comparison.png", "generalisation_calibration_mape.png",
                                        "14_driving_styles.png", "18_epa_gap_by_road_and_style.png"]]
    shown = [f for f in figs if f.exists()]
    for f in shown:
        st.image(str(f), width="stretch")
    if not shown:
        st.info("Run 06_make_plots.py to create the figures.")
