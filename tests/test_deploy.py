"""Deployment (Streamlit Community Cloud: Linux, fresh clone): every file the app reads at runtime is in git, paths
match the real folder names exactly (Linux is case-sensitive), requirements.txt pins the installed versions, and the
first start creates the database, the calibration store and the demo account (once)."""
import os
import re
import subprocess
from importlib import metadata

from _helpers import ROOT, Checks, setup

tmp = setup("deploy")
import motorlytics_features as mf  # noqa: E402
from core import app_core, db  # noqa: E402
from ui import theme  # noqa: E402

c = Checks("test_deploy")


def committable(path) -> bool:
    rel = os.path.relpath(path, ROOT)
    return subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode != 0


def exact_case(path) -> bool:
    """Every part of the path exists with exactly this spelling (what Linux needs)."""
    p = ROOT
    for part in os.path.relpath(path, ROOT).replace("\\", "/").split("/"):
        if part not in os.listdir(p):
            return False
        p = p / part
    return True


# ---- runtime files ----
meta = app_core._meta()
models = [app_core.model_path(meta[f"best_{t}"]) for t in app_core.TIERS if f"best_{t}" in meta]
models.append(mf.MODELS / "driving_style.joblib")
c.ok(all(p.exists() and committable(p) and exact_case(p) for p in models), "app models exist and are not gitignored",
     ", ".join(p.name for p in models))
c.ok(app_core.model_path("behaviour_only/LWR").as_posix().endswith("models/trip_behaviour_only_LWR.joblib")
     and "\\" in meta["models"]["behaviour_only/LWR"]["path"], "Windows paths in meta_trip.json resolve on Linux")
runtime = [mf.MODELS / "meta_trip.json", ROOT / theme.LOGO, ROOT / theme.ICON, theme.MARK_SVG, ROOT / "assets" / "pk_fuel_reference.csv",
           ROOT / "assets" / "quick_estimate_profiles.csv", ROOT / "assets" / "pk_cycles" / "city.csv",
           ROOT / "assets" / "pk_cycles" / "highway.csv", ROOT / "samples" / "index.csv",
           ROOT / "samples" / "corsa_d_2010_trip1.csv", mf.PROC / "results_trip" / "test_metrics.csv",
           mf.PROC / "results_trip" / "model_selection_table.csv", mf.PROC / "results_trip" / "generalisation.txt",
           ROOT / "tests" / "fixtures" / "SYNTHETIC_alto_lahore_city.csv", ROOT / ".streamlit" / "config.toml"]
bad = [os.path.relpath(p, ROOT) for p in runtime if not (p.exists() and committable(p) and exact_case(p))]
c.ok(not bad, "runtime data files are in git with the exact spelling", ", ".join(bad))
c.ok(all(not committable(ROOT / p) for p in ["models/calibration.json", "Data/motorlytics.db", "Data/uploads/x.jpg"]),
     "database, calibration store and uploads are never committed")

# ---- exact folder names ----
c.ok(mf.DATA.name == "Data" and mf.DATA.is_absolute() and mf.MODELS.is_absolute() and exact_case(mf.DATA),
     "data paths anchored to the project and spelled 'Data'")
src = (ROOT / "core" / "db.py").read_text(encoding="utf-8")
c.ok('ROOT / "Data" / "motorlytics.db"' in src, "database path uses 'Data' (the real folder name)")
code = " ".join(p.read_text(encoding="utf-8") for d in ["core", "ui", "pages"] for p in (ROOT / d).glob("*.py"))
code += (ROOT / "app.py").read_text(encoding="utf-8")
lits = set(re.findall(r'["\']((?:assets|samples|models|pages|Data)/[\w./-]+\.\w+)["\']', code))
missing = sorted(x for x in lits if not exact_case(ROOT / x))
c.ok(not missing, "every literal file path in the app matches the files exactly", ", ".join(missing))

# ---- requirements.txt pinned to the installed versions ----
req = dict(re.findall(r"^([\w-]+)==([\w.]+)", (ROOT / "requirements.txt").read_text(), re.M))
off = {k: (v, metadata.version(k)) for k, v in req.items() if metadata.version(k) != v}
c.ok(req.get("scikit-learn") == "1.9.1" and not off, "requirements.txt pins the installed versions", str(off))
c.ok("lightgbm" not in req and "matplotlib" not in req, "no training-only packages in the runtime requirements")

# ---- first start: database, calibration store, demo account (once) ----
from core import demo  # noqa: E402

c.ok(not db.DB_PATH.exists() and not app_core.APP_CALIB.exists(), "fresh start: no database, no calibration store")
created = demo.ensure_first_start()
user = db.get_user_by_email(demo.EMAIL)
c.ok(created and db.DB_PATH.exists() and app_core.APP_CALIB.exists() and user, "first start creates the database, "
     "the calibration store and the demo account")
trips = db.get_trips(user["id"])
c.ok(len(db.get_cars(user["id"])) == 3 and len(trips) >= 8 and any(t["calibrated"] for t in trips),
     "demo account has 3 cars and calibrated trips", f"{len(trips)} trips")
c.ok(not demo.ensure_first_start() and db.user_count() == 1, "second start does not add another demo account")
c.done()
