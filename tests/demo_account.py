"""Create a filled demo account so you can click through the app (NOT a test; run_all.py skips it).

    .venv\\Scripts\\python.exe tests\\demo_account.py                 -> the app's real database (data/motorlytics.db)
    .venv\\Scripts\\python.exe tests\\demo_account.py --db other.db   -> another database file

Sign in afterwards with demo@motorlytics.test / demo12345. Refuses to run if that account already exists.
Adds: Demo Corsa (manual entry; samples/gps_trip1-4 with MAF, a speed-only VED trip, one fill-up),
Family Cultus (from the catalogue, no trips) and a SYNTHETIC test Alto (the made-up Lahore trips in
tests/fixtures, PakWheels-adjusted). The petrol price 280 PKR/L is only a demo value.
"""
import argparse
import os
import sys
from io import StringIO
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=None, help="database file (default: the app's data/motorlytics.db)")
ap.add_argument("--calibration", default=None, help="calibration store (default: models/calibration.json)")
a = ap.parse_args()
if a.db:
    os.environ["MOTORLYTICS_DB"] = str(Path(a.db).resolve())
if a.calibration:
    os.environ["MOTORLYTICS_CALIB"] = str(Path(a.calibration).resolve())
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
SAMPLES, FIXTURES = ROOT / "samples", ROOT / "tests" / "fixtures"

import pandas as pd  # noqa: E402

from core import app_core, auth, car_photos, cars, db  # noqa: E402

EMAIL, PW = "demo@motorlytics.test", "demo12345"
db.init_db()
if db.get_user_by_email(EMAIL):
    raise SystemExit(f"{EMAIL} already exists in {db.DB_PATH}")
user, _ = auth.register("Demo Driver", EMAIL, PW, "Islamabad")
db.update_user(user["id"], petrol_pkr=280.0)
user = db.get_user(user["id"])
print("account:", EMAIL, "in", db.DB_PATH)


def show(res):
    for tid in res["trip_ids"]:
        t = db.get_trip(tid, user["id"])
        fuel = f"measured {t['measured_l100']:.2f}" if t["measured_l100"] is not None else "no MAF"
        print(f"  trip {tid}: {t['dist_km']:.1f} km, {t['road_type']}, {fuel}, model {t['pred_l100']:.2f} L/100 km, "
              f"eco-score {t['eco_score']:.0f}")
    print("  ", res["calibration"])


corsa = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4.0, nickname="Demo Corsa", colour="Red",
                   transmission="Manual", fuel_type="petrol", body_type="Hatchback", assembled="Imported",
                   tank_litres=45.0, odometer_km=98000.0, registration="LEA-19-1234")
print("Demo Corsa: 3 MAF trips, then a 4th (calibrated)")
show(app_core.ingest_logs(user, corsa, [(f"gps_trip{i}.csv", (SAMPLES / f"gps_trip{i}.csv").read_bytes())
                                        for i in (1, 2, 3)], temp=18.0))
show(app_core.ingest_logs(user, corsa, [("gps_trip4.csv", (SAMPLES / "gps_trip4.csv").read_bytes())], temp=18.0))
speed_only = pd.read_csv(SAMPLES / "ved_mixed_vehicle530.csv")[["time", "speed"]]
buf = StringIO()
speed_only.to_csv(buf, index=False)
print("Demo Corsa: a speed-only log (time + speed of a real VED trip, recorded by a different car)")
r = app_core.ingest_logs(user, corsa, [("speed_only.csv", buf.getvalue().encode())], temp=18.0)
show(r)
t5 = db.get_trip(r["trip_ids"][0], user["id"])
print("  ", app_core.log_fillup(corsa, [t5], litres=round(t5["dist_km"] * 7.5 / 100, 2), date="2026-11-26")["message"])

sug = cars.lookup("Suzuki", "Cultus", 2018)
cultus = db.add_car(user["id"], "Suzuki", "Cultus", 2018, sug.get("disp_L", 1.0), sug.get("cylinders", 3.0),
                    nickname="Family Cultus", transmission=sug.get("transmission"), fuel_type=sug.get("fuel_type"),
                    body_type=sug.get("body_type"), assembled=sug.get("assembled"), tank_litres=35.0)
alto = db.add_car(user["id"], "Suzuki", "Alto", 2019, 0.66, 3.0, nickname="SYNTHETIC test Alto",
                  transmission="Manual", fuel_type="petrol", body_type="Hatchback", assembled="Local", tank_litres=27.0)
print("SYNTHETIC test Alto: made-up Lahore trips (PakWheels-adjusted)")
show(app_core.ingest_logs(user, alto, [(p.name, p.read_bytes()) for p in sorted(FIXTURES.glob("SYNTHETIC_*.csv"))],
                          temp=20.0))
for c_ in (corsa, cultus, alto):
    car_photos.attach_example_photo(db.get_car(c_["id"]))
print(f"\nDone. Sign in with {EMAIL} / {PW}")
