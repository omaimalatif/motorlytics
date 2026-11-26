"""Every file in samples/ uploads through the app's upload code without errors."""
from _helpers import SAMPLES, Checks, setup, upload

setup("samples")
from core import auth, db  # noqa: E402

import pandas as pd  # noqa: E402

# car, engine and measured fuel of each sample (samples/index.csv, also used by the fetch fallback)
INDEX = pd.read_csv(SAMPLES / "index.csv").set_index("file")

c = Checks("test_samples")
db.init_db()
user, _ = auth.register("Samples Tester", "samples@example.com", "secret123", None)
db.update_user(user["id"], petrol_pkr=280.0)
user = db.get_user(user["id"])
files = sorted(p for p in SAMPLES.glob("*.csv") if p.name != "index.csv")
c.ok(len(files) >= 11, "samples present", f"{len(files)} files")
for f in files:
    c.ok(f.name in INDEX.index, f"{f.name} listed in samples/index.csv")
    if f.name not in INDEX.index:
        continue
    meta = INDEX.loc[f.name]
    car = db.add_car(user["id"], "Sample", f.stem, None, float(meta["disp_L"]), float(meta["cylinders"]))
    try:                                                                  # fresh car: no calibration carry-over
        res = upload(user, car, [f])
    except Exception as e:                                                # noqa: BLE001
        c.ok(False, f"{f.name} uploads", f"{type(e).__name__}: {e}")
        continue
    t = db.get_trip(res["trip_ids"][0], user["id"])
    c.ok(len(res["trip_ids"]) >= 1, f"{f.name} uploads", f"tier {t['tier']}, model {t['pred_l100']:.2f} L/100 km")
    c.ok(abs(t["measured_l100"] - float(meta["measured_l100"])) < 0.01, f"{f.name} measured fuel as listed",
         f"{t['measured_l100']:.2f} vs {meta['measured_l100']}")
    if f.name.startswith("ved_raw"):
        c.ok(t["tier"] == "full", "raw VED file uses the full model tier", t["tier"])
c.done()
