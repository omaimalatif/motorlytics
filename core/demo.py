"""The demo account (demo@motorlytics.test / demo12345), so the app is never empty: created by app.py on the first
start when the database has no users (e.g. online, where the disk resets), or by tests/demo_account.py.

Adds: Demo Corsa (manual entry; samples/corsa_d_2010_trip1-4: 3 MAF trips, then a 4th predicted calibrated;
a speed-only VED trip and one fill-up), Family Cultus (from the catalogue, no trips) and a SYNTHETIC test Alto
(the made-up Lahore trips in tests/fixtures, PakWheels-adjusted). The petrol price 280 PKR/L is only a demo value.
"""
from io import StringIO

import pandas as pd

from core import app_core, auth, car_photos, cars, db

EMAIL, PASSWORD = "demo@motorlytics.test", "demo12345"
SAMPLES, FIXTURES = db.ROOT / "samples", db.ROOT / "tests" / "fixtures"


def create_demo_account(photos: bool = True, log=lambda *a: None) -> dict | None:
    """Create the demo account with its cars and trips; None if it already exists. log: print-like callback."""
    db.init_db()
    if db.get_user_by_email(EMAIL):
        return None
    user, _ = auth.register("Demo Driver", EMAIL, PASSWORD, "Islamabad")
    db.update_user(user["id"], petrol_pkr=280.0)
    user = db.get_user(user["id"])
    log("account:", EMAIL, "in", db.DB_PATH)

    def show(res):
        for tid in res["trip_ids"]:
            t = db.get_trip(tid, user["id"])
            fuel = f"measured {t['measured_l100']:.2f}" if t["measured_l100"] is not None else "no MAF"
            log(f"  trip {tid}: {t['dist_km']:.1f} km, {t['road_type']}, {fuel}, model {t['pred_l100']:.2f} L/100 km, "
                f"eco-score {t['eco_score']:.0f}")
        log("  ", res["calibration"])

    corsa = db.add_car(user["id"], "Opel", "Corsa", 2010, 1.2, 4, nickname="Demo Corsa", colour="Red",
                       transmission="Manual", fuel_type="petrol", body_type="Hatchback", assembled="Imported",
                       tank_litres=45.0, odometer_km=98000.0, registration="LEA-19-1234")
    log("Demo Corsa: 3 MAF trips, then a 4th (calibrated)")
    show(app_core.ingest_logs(user, corsa, [(f"corsa_d_2010_trip{i}.csv", (SAMPLES / f"corsa_d_2010_trip{i}.csv").read_bytes())
                                            for i in (1, 2, 3)], temp=18.0))
    show(app_core.ingest_logs(user, corsa, [("corsa_d_2010_trip4.csv", (SAMPLES / "corsa_d_2010_trip4.csv").read_bytes())],
                              temp=18.0))
    speed_only = pd.read_csv(SAMPLES / "ved_mixed_vehicle530.csv")[["time", "speed"]]
    buf = StringIO()
    speed_only.to_csv(buf, index=False)
    log("Demo Corsa: a speed-only log (time + speed of a real VED trip, recorded by a different car)")
    r = app_core.ingest_logs(user, corsa, [("speed_only.csv", buf.getvalue().encode())], temp=18.0)
    show(r)
    t5 = db.get_trip(r["trip_ids"][0], user["id"])
    log("  ", app_core.log_fillup(corsa, [t5], litres=round(t5["dist_km"] * 7.5 / 100, 2), date="2026-11-26")["message"])

    sug = cars.lookup("Suzuki", "Cultus", 2018)               # {} when the catalogue file is not there
    cultus = db.add_car(user["id"], "Suzuki", "Cultus", 2018, sug.get("disp_L", 1.0), sug.get("cylinders", 3),
                        nickname="Family Cultus", transmission=sug.get("transmission") or "Manual",
                        fuel_type=sug.get("fuel_type") or "petrol", body_type=sug.get("body_type") or "Hatchback",
                        assembled=sug.get("assembled") or "Local", tank_litres=35.0)
    alto = db.add_car(user["id"], "Suzuki", "Alto", 2019, 0.66, 3, nickname="SYNTHETIC test Alto",
                      transmission="Manual", fuel_type="petrol", body_type="Hatchback", assembled="Local", tank_litres=27.0)
    log("SYNTHETIC test Alto: made-up Lahore trips (PakWheels-adjusted)")
    show(app_core.ingest_logs(user, alto, [(p.name, p.read_bytes()) for p in sorted(FIXTURES.glob("SYNTHETIC_*.csv"))],
                              temp=20.0))
    if photos:
        for c_ in (corsa, cultus, alto):
            try:
                car_photos.attach_example_photo(db.get_car(c_["id"]))   # Wikipedia; a drawing if none / offline
            except Exception:
                pass
    return user


def ensure_first_start() -> bool:
    """Startup online: create the database and the calibration store if missing, and the demo account if there are
    no users yet. Returns True if the demo account was created now."""
    import motorlytics_features as mf
    db.init_db()
    if not app_core.APP_CALIB.exists():
        mf.save_calibration({"cars": {}}, app_core.APP_CALIB)
    if db.user_count() == 0:
        return create_demo_account() is not None
    return False
