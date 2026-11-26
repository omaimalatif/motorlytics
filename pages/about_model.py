"""How the predictions are made, and how well they did (from Data/processed/results_trip)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import motorlytics_features as mf
from ui import components as ui
from ui import session

session.require_user()
ui.page_header("About the model", "Course models (Decision Tree, ANN, k-NN, LWR) trained on the Vehicle Energy "
               "Dataset (Ann Arbor, 2017-18), tested on unseen vehicles and on external data.")
REP = mf.PROC / "results_trip"
meta_p = mf.MODELS / "meta_trip.json"
meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

tiers = {"behaviour_only": "Speed only (phone / any OBD log)", "full_no_load": "Speed + RPM",
         "full": "Speed + RPM + engine load"}
cols = st.columns(3)
for c, (fs, label) in zip(cols, tiers.items()):
    with c:
        with ui.card(label):
            key = meta.get(f"best_{fs}", "–")
            m = meta.get("models", {}).get(key, {})
            st.metric("App model", key.split("/")[-1])
            if m:
                st.caption(f"Held-out test: MAPE {m['mape']:.1f}% · RMSE {m['rmse']:.2f} L/100 km · R² {m['r2']:.2f}")
            ref = meta.get(f"best_reference_{fs}")
            if ref:
                st.caption(f"Reference (beyond the syllabus): {ref.split('/')[-1]}")
if meta.get("selection"):
    st.caption("Choice rule: " + meta["selection"]["rule"])

tab1, tab2, tab3, tab4 = st.tabs(["Test results", "Model choice", "Generalisation", "Figures"])
with tab1:
    p = REP / "test_metrics.csv"
    if p.exists():
        d = pd.read_csv(p)
        st.dataframe(d, hide_index=True, width="stretch")
        st.caption("Held-out VED vehicles (never seen in training). Baselines: mean, and median per engine size.")
    else:
        st.info("Run 03_train.py to create the results.")
with tab2:
    p = REP / "model_selection_table.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p).round(2), hide_index=True, width="stretch")
        st.caption("MAPE %: VED test, enviroCar (21 European cars) before / after 3-trip calibration, exp1 (3 cars).")
    else:
        st.info("Run 09_generalisation.py to create this table.")
with tab3:
    for name in ["generalisation.txt", "external_exp1.txt"]:
        p = REP / name
        if p.exists():
            with st.expander(name):
                st.code(p.read_text(encoding="utf-8"), language=None)
with tab4:
    figs = [Path("plots") / f for f in ["11_model_comparison.png", "generalisation_calibration_mape.png",
                                        "14_driving_styles.png", "18_epa_gap_by_road_and_style.png"]]
    shown = [f for f in figs if f.exists()]
    for f in shown:
        st.image(str(f), width="stretch")
    if not shown:
        st.info("Run 06_make_plots.py to create the figures.")
