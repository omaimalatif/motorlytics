"""PakWheels starting adjustment: factors, mixed-trip clamp, own calibration overrides it, pages render."""
from _helpers import FIXTURES, ROOT, SAMPLES, Checks, setup, upload

setup("pakwheels")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import app_core, auth, db, pk_reference  # noqa: E402

c = Checks("test_pakwheels")
db.init_db()
user, _ = auth.register("PK Tester", "pk@example.com", "secret123", "Lahore")
db.update_user(user["id"], petrol_pkr=280.0)
user = db.get_user(user["id"])
alto = db.add_car(user["id"], "Suzuki", "Alto", 2019, 0.66, 3.0, transmission="Manual", fuel_type="petrol")

ref = pk_reference.match(alto)
c.ok(ref and ref["city_kmpl"] == 18 and ref["highway_kmpl"] == 22, "Alto matched to PakWheels 18 / 22 km/L")
c.ok(pk_reference.match({"make": "Opel", "model": "Corsa", "disp_L": 1.2}) is None, "unknown model -> no match")

# clamp unit test: a mixed trip whose blended factor would beat the highway figure is held at highway km/L
p = pk_reference.prior(alto)
raw = 100 / 13.8                                  # the model's raw mixed prediction for the synthetic Alto trip
pred, k = pk_reference.adjust(p, raw, 42.2)
c.ok(18 - 1e-9 <= 100 / pred <= 22 + 1e-9, "mixed trip clamped between city and highway km/L", f"{100 / pred:.1f}")
pred_c, _ = pk_reference.adjust(p, 100 / 9.8, 19.3)
c.ok(abs(pred_c - 100 / 9.8 * p["k_city"]) < 1e-9, "city trip: city factor, no clamp")
pred_h, _ = pk_reference.adjust(p, 100 / 15.2, 88.6)
c.ok(abs(pred_h - 100 / 15.2 * p["k_highway"]) < 1e-9, "highway trip: highway factor, no clamp")

# end to end on the synthetic Alto trips (labelled SYNTHETIC: not real data)
logs = sorted(FIXTURES.glob("SYNTHETIC_alto_*.csv"))
res = upload(user, alto, logs, temp=16.7)
trips = {t["road_type"]: t for t in (db.get_trip(i, user["id"]) for i in res["trip_ids"])}
c.ok(all(t["tips"]["reference_adjusted"] for t in trips.values()), "trips marked PakWheels-adjusted")
kmpl = {k: 100 / t["pred_l100"] for k, t in trips.items()}
print("  km/L:", {k: round(v, 1) for k, v in kmpl.items()})
c.ok(kmpl["city"] <= kmpl["mixed"] <= kmpl["highway"] + 1e-9, "city <= mixed <= highway after the clamp")
c.ok(18 <= kmpl["mixed"] <= 22, "mixed between PakWheels city and highway")

# own calibration wins
upload(user, alto, [SAMPLES / "gps_trip1.csv", SAMPLES / "gps_trip2.csv"], temp=16.7)
r = upload(user, alto, [logs[0]], temp=16.7)
t = db.get_trip(r["trip_ids"][0], user["id"])
c.ok(t["calibrated"] and not t["tips"]["reference_adjusted"], "own calibration replaces the PakWheels factor")

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.session_state["trip_id"] = res["trip_ids"][0]
for page in ["trip_detail", "my_car", "calibration", "overview"]:
    at.switch_page(f"app_pages/{page}.py").run()
    c.page(at, page)
fresh = db.add_car(user["id"], "Suzuki", "Alto", 2020, 0.66, 3.0, transmission="Manual")
# AppTest drops the sidebar after switch_page, so check it on a fresh run with the car already selected
at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at2.session_state["user_id"] = user["id"]
at2.session_state["car_id"] = fresh["id"]
at2.run()
sidebar_text = [str(getattr(e, "value", "")) for e in at2.sidebar.children.values()]
c.ok(any("PakWheels-adjusted" in s for s in sidebar_text), "sidebar shows PakWheels-adjusted")
c.done()
