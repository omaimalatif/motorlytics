"""Create a filled demo account so you can click through the app (NOT a test; run_all.py skips it).
The app also creates it by itself on the first start when the database has no users (core/demo.py).

    .venv\Scripts\python.exe tests\demo_account.py                 -> the app's real database (Data/motorlytics.db)
    .venv\Scripts\python.exe tests\demo_account.py --db other.db   -> another database file

Sign in afterwards with demo@motorlytics.test / demo12345. Refuses to run if that account already exists.
What it adds: see core/demo.py.
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=None, help="database file (default: the app's Data/motorlytics.db)")
ap.add_argument("--calibration", default=None, help="calibration store (default: models/calibration.json)")
a = ap.parse_args()
if a.db:
    os.environ["MOTORLYTICS_DB"] = str(Path(a.db).resolve())
if a.calibration:
    os.environ["MOTORLYTICS_CALIB"] = str(Path(a.calibration).resolve())
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from core import db, demo  # noqa: E402

if demo.create_demo_account(log=print) is None:
    raise SystemExit(f"{demo.EMAIL} already exists in {db.DB_PATH}")
print(f"
Done. Sign in with {demo.EMAIL} / {demo.PASSWORD}")
