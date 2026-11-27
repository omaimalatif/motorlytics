"""Public Welcome page (shown before sign-in): hero with the logo, real results, how it works, features,
the ML behind it, data sources. Numbers are read from the results files; the fallbacks are the same results
for a checkout without Data/ (results files are not all in git)."""
import json
import re

import streamlit as st

import motorlytics_features as mf
from ui import theme

E = theme.esc
FALLBACK = {"trips": 12793, "vehicles": 196, "test_mape": 8.7, "cal_mape": 10.3}


@st.cache_data(ttl=3600, show_spinner=False)
def headline_stats() -> dict:
    """Training trips and vehicles (trips.parquet), best course-model error on held-out vehicles
    (meta_trip.json), and the app model's enviroCar error after 3-trip calibration (model selection results)."""
    out = dict(FALLBACK)
    try:
        import pyarrow.parquet as pq
        p = mf.PROC / "trips.parquet"
        if p.exists():
            out["trips"] = pq.ParquetFile(p).metadata.num_rows
            out["vehicles"] = int(pq.read_table(p, columns=["vehid"]).column("vehid").to_pandas().nunique())
    except Exception:
        pass
    meta = {}
    try:
        meta = json.loads((mf.MODELS / "meta_trip.json").read_text())
        course = [m["mape"] for m in meta.get("models", {}).values() if m.get("group") == "course" and m.get("mape")]
        if course:
            out["test_mape"] = min(course)
    except Exception:
        pass
    app_tier, app_algo = (meta.get("best_behaviour_only") or "behaviour_only/LWR").split("/")
    sel = mf.PROC / "results_trip" / "model_selection_table.csv"
    gen = mf.PROC / "results_trip" / "generalisation.txt"
    try:
        if sel.exists():
            import pandas as pd
            d = pd.read_csv(sel)
            row = d[(d["tier"] == app_tier) & (d["model"] == app_algo)]
            if len(row):
                out["cal_mape"] = float(row["envirocar_mape_cal3"].iloc[0])
        elif gen.exists():                    # first block in the file is enviroCar: before, after
            m = re.search(rf"^\s*3\s+{app_tier}\s+course\s+{app_algo}\s+\d+\s+\d+\s+[\d.]+\s+([\d.]+)",
                          gen.read_text(encoding="utf-8"), re.M)
            if m:
                out["cal_mape"] = float(m.group(1))
    except Exception:
        pass
    return out


def go_auth(mode: str) -> None:
    st.session_state["auth_mode"] = mode
    st.switch_page("pages/sign_in.py")


st.html("""<style>
[data-testid="stHeader"] { background: #000000; }
[data-testid="stMainBlockContainer"] { padding-top: 0; max-width: 1180px; }
.st-key-mlhero { background: #000000; width: 100vw !important; max-width: 100vw; margin-left: calc(50% - 50vw);
  padding: 8px 16px 56px;
  border-bottom: 1px solid #17171B; position: relative; }
.st-key-mlhero::after { content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px;
  background: var(--ml-gradient); opacity: .55; }
.st-key-mlhero .ml-lockup { margin: 64px 0 12px 0; }
@media (max-width: 640px) { .st-key-mlhero .ml-lockup .ml-brandmark { width: 52px !important; height: 52px !important; }
  .st-key-mlhero .ml-wordmark { font-size: 38px !important; } }
@media (max-width: 640px) { .st-key-mlhero { padding-top: 40px; } }
.ml-hero-title { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: clamp(2rem, 4.6vw, 3.4rem);
  line-height: 1.08; letter-spacing: -0.03em; text-align: center; margin: 0; color: #F4F4F5; }
.ml-hero-sub { color: #A1A1AA; text-align: center; font-size: clamp(1rem, 1.6vw, 1.15rem); max-width: 720px;
  margin: 14px auto 0 auto; line-height: 1.55; }
.ml-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 8px 0 0 0; }
.ml-stat { background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: 12px; padding: 18px 20px; }
.ml-stat-value { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 2rem; letter-spacing: -0.02em; }
.ml-stat-label { color: #A1A1AA; font-size: .875rem; margin-top: 2px; }
.ml-section { margin: 56px 0 8px 0; }
.ml-eyebrow { color: #F26A10; font-weight: 600; font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; margin: 0; }
.ml-h2 { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: clamp(1.5rem, 2.6vw, 2rem);
  letter-spacing: -0.02em; margin: 6px 0 6px 0; color: #F4F4F5; }
.ml-lead { color: #A1A1AA; max-width: 720px; margin: 0 0 20px 0; line-height: 1.6; }
.ml-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
.ml-step, .ml-feature { background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: 12px;
  padding: 22px; box-shadow: var(--ml-shadow); transition: transform .15s ease, border-color .15s ease; }
.ml-step:hover, .ml-feature:hover { transform: translateY(-2px); border-color: rgba(242,106,16,.45); }
.ml-step-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.ml-step-n { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: #3A3A44; font-size: 1.6rem; }
.ml-ic { width: 44px; height: 44px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center;
  color: #F26A10; background: rgba(242,106,16,.10); border: 1px solid rgba(242,106,16,.25); }
.ml-step h4, .ml-feature h4 { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; margin: 0 0 6px 0; color: #F4F4F5; }
.ml-step p, .ml-feature p { color: #A1A1AA; font-size: .92rem; line-height: 1.55; margin: 0; }
.ml-features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.ml-feature .ml-ic { margin-bottom: 14px; }
.ml-ml-card { background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: 12px; padding: 22px 24px; }
.ml-ml-card h4 { font-family: 'Plus Jakarta Sans', sans-serif; margin: 16px 0 6px 0; font-size: 1rem; color: #F4F4F5; }
.ml-ml-card h4:first-child { margin-top: 0; }
.ml-ml-card p { color: #A1A1AA; line-height: 1.6; margin: 0; font-size: .93rem; }
.ml-ml-card b { color: #F4F4F5; font-weight: 600; }
.ml-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 0 0; }
.ml-chip { border-radius: 999px; padding: 3px 11px; font-size: .78rem; font-weight: 600; border: 1px solid var(--ml-border);
  background: var(--ml-raised); color: #F4F4F5; }
.ml-chip.course { border-color: rgba(242,106,16,.45); color: #F8A66B; }
.ml-footer { margin: 64px calc(50% - 50vw) -4rem; padding: 32px calc(50vw - 50% + 16px) 40px; background: #000000;
  border-top: 1px solid #17171B; color: #A1A1AA; font-size: .875rem; }
.ml-footer-grid { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px 32px; }
.ml-footer b { color: #F4F4F5; font-family: 'Plus Jakarta Sans', sans-serif; }
.ml-footer a { color: #A1A1AA; text-decoration: none; border-bottom: 1px solid #3A3A44; }
.ml-footer a:hover { color: #F26A10; border-color: #F26A10; }
.ml-footer-links { display: flex; flex-wrap: wrap; gap: 6px 18px; }
</style>""")

s = headline_stats()

# ---------------- 1. hero (black) ----------------
with st.container(key="mlhero", horizontal_alignment="center", gap="medium"):
    st.html(theme.brand_lockup(84))
    st.html('<h1 class="ml-hero-title">Know what every trip <span class="ml-grad-text">really costs</span></h1>'
            '<p class="ml-hero-sub">Motorlytics predicts your car\'s fuel use for each trip from its trip log, scores '
            'how you drive, and turns it into rupees, with tips to spend less.</p>')
    with st.container(horizontal=True, horizontal_alignment="center", gap="small"):
        if st.button("Get started", type="primary", icon=":material/arrow_forward:", icon_position="right",
                     key="hero_start"):
            go_auth("Create account")
        if st.button("Sign in", key="hero_signin"):
            go_auth("Sign in")

# ---------------- 2. stats strip (real results) ----------------
stats = [(f"{s['trips']:,}", "training trips (VED)"), (f"{s['vehicles']:,}", "vehicles"),
         (f"{s['test_mape']:.1f} %", "error on unseen cars (best course model)"),
         (f"{s['cal_mape']:.1f} %", "after 3-trip calibration on European cars")]
st.html('<div class="ml-section" style="margin-top:32px"><div class="ml-stats">' + "".join(
    f'<div class="ml-stat"><div class="ml-stat-value ml-grad-text">{E(v)}</div><div class="ml-stat-label">{E(k)}</div></div>'
    for v, k in stats) + "</div></div>")

# ---------------- 3. how it works ----------------
steps = [("upload", "Upload a trip log", "A CSV from Torque, Car Scanner, enviroCar or VED, or try a bundled sample or a "
          "real enviroCar trip with one click."),
         ("model", "The model reads your driving", "The ML model predicts L/100 km for the trip and k-Means places "
          "your driving style: eco, normal or aggressive."),
         ("coins", "Tips, cost and calibration", "Plain eco-driving tips, the trip's cost in PKR, and a correction "
          "factor that tunes every later prediction to your own car.")]
st.html('<div class="ml-section"><p class="ml-eyebrow">How it works</p><h2 class="ml-h2">From a trip log to a fuel bill '
        'in three steps</h2><div class="ml-steps">' + "".join(
            f'<div class="ml-step"><div class="ml-step-top"><span class="ml-ic">{theme.icon(ic)}</span>'
            f'<span class="ml-step-n">0{i}</span></div><h4>{E(t)}</h4><p>{E(d)}</p></div>'
            for i, (ic, t, d) in enumerate(steps, 1)) + "</div></div>")

# ---------------- 4. features ----------------
features = [("fuel", "Fuel prediction", "L/100 km and litres for every trip, measured from the MAF sensor when your "
             "log has one, predicted from speed, RPM and load when it does not."),
            ("gauge", "Driving-style score", f"A 0-100 eco-score from k-Means clustering of {s['trips']:,} real trips, "
             "so you see whether a trip was driven eco, normal or aggressive."),
            ("tips", "Eco-driving tips", "What to change, compared with eco drivers on the same kind of road, and how "
             "many litres per 100 km it would save."),
            ("calibrate", "Per-car calibration", "Three trips with MAF or one full-tank fill-up give your car its own "
             "correction factor, so later estimates fit your car."),
            ("pin", "Pakistani cars", "Suzuki, Toyota, Honda and more from the PakWheels fuel-average reference, used "
             "as a starting point until your car is calibrated."),
            ("calculator", "Quick estimate", "No log at hand? Enter the distance, road type and driving style and get "
             "fuel and cost for a planned trip.")]
st.html('<div class="ml-section"><p class="ml-eyebrow">Features</p><h2 class="ml-h2">Everything you need to drive '
        'cheaper</h2><div class="ml-features">' + "".join(
            f'<div class="ml-feature"><span class="ml-ic">{theme.icon(ic)}</span><h4>{E(t)}</h4><p>{E(d)}</p></div>'
            for ic, t, d in features) + "</div></div>")

# ---------------- 5. the ML behind it ----------------
chips_course = "".join(f'<span class="ml-chip course">{m}</span>' for m in
                       ["Decision Tree", "MLP (neural network)", "k-NN", "LWR"])
chips_ref = "".join(f'<span class="ml-chip">{m}</span>' for m in ["Ridge", "Random Forest", "LightGBM"])
st.html('<div class="ml-section"><p class="ml-eyebrow">The ML behind it</p>'
        '<h2 class="ml-h2">Course models, tested on cars they have never seen</h2></div>')
ml_text, ml_art = st.columns([1.25, 1], gap="large", vertical_alignment="center")
with ml_text:
    st.html(f"""<div class="ml-ml-card">
<h4>Course models</h4><p>Trained on {s['trips']:,} trips from {s['vehicles']} vehicles of the Vehicle Energy Dataset
(Ann Arbor, 2017-18). The test vehicles are held out completely, so every score is on cars the models never saw.</p>
<div class="ml-chips">{chips_course}</div>
<h4>Reference models</h4><p>Beyond the syllabus, for comparison only; the app never uses them.</p>
<div class="ml-chips">{chips_ref}</div>
<h4>How the app model was chosen</h4><p>Per input type (speed only, speed + RPM, speed + RPM + load), the course model
with the lowest error on 21 European enviroCar cars <b>after 3-trip calibration</b>, among those within 2 points of the
best course model on the held-out VED cars.</p>
<h4>Speed-glitch cleaning</h4><p>OBD loggers write 255 km/h when they have no reading, and single readings can spike.
These are removed and filled in over time before any feature is computed, in training and in the app.</p>
<p style="margin-top:14px"><b>About the model</b> has the full results after you sign in.</p></div>""")
with ml_art:
    st.image(theme.image_svg("hero_car"), width="stretch")

# ---------------- 6. footer ----------------
links = [("VED", "https://github.com/gsoh/VED"), ("enviroCar", "https://envirocar.org"),
         ("Open-Meteo", "https://open-meteo.com"), ("PakWheels", "https://www.pakwheels.com")]
st.html('<footer class="ml-footer"><div class="ml-footer-grid"><div><b>Motorlytics</b><br>NUML Machine Learning Lab'
        '</div><div><div style="margin-bottom:6px">Data sources</div><div class="ml-footer-links">' + "".join(
            f'<a href="{u}" target="_blank" rel="noopener">{E(n)}</a>' for n, u in links) + "</div></div></div></footer>")
