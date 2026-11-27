"""Quick estimate: PakWheels level under standard conditions, style / road ordering, factor sources,
claimed km/L, nothing saved until the Save button, page and trip detail render."""
from _helpers import ROOT, SAMPLES, Checks, setup, upload

setup("quick")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, db, quick_estimate as q  # noqa: E402

c = Checks("test_quick_estimate")
db.init_db()
user, _ = auth.register("Quick Tester", "quick@example.com", "secret123", None)
db.update_user(user["id"], petrol_pkr=280.0)
user = db.get_user(user["id"])
alto = db.add_car(user["id"], "Suzuki", "Alto", 2019, 0.66, 3.0, transmission="Manual", fuel_type="petrol")
corsa = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4.0)

# PakWheels sets the level: normal driving under standard conditions = PakWheels figure
for road, ref in [("city", 18.0), ("highway", 22.0)]:
    e = q.estimate(alto, q.STANDARD[road], road, "normal", temp_c=25.0, petrol_pkr=280, when=q.STANDARD_WHEN)
    c.ok(abs(e["kmpl"] - ref) < 0.05 and e["factor_source"] == "pakwheels",
         f"Alto {road}, normal, standard conditions = PakWheels {ref:.0f} km/L", f"{e['kmpl']:.2f}")
e = q.estimate(alto, 15, "mixed", "normal", temp_c=25.0, petrol_pkr=280, when=q.STANDARD_WHEN)
c.ok(18 <= e["kmpl"] <= 22, "mixed between city and highway", f"{e['kmpl']:.1f}")

# the model sets the differences: eco > normal > aggressive km/L on every road type
for road in q.ROAD_TYPES:
    e = q.estimate(alto, 20, road, "normal", temp_c=25.0, petrol_pkr=280)
    k = {s: v["kmpl"] for s, v in e["by_style"].items()}
    c.ok(k["eco"] > k["normal"] > k["aggressive"], f"{road}: eco > normal > aggressive",
         ", ".join(f"{s} {v:.1f}" for s, v in k.items()))
    c.ok(abs(e["cost_pkr"] - e["litres"] * 280) < 1e-6 and e["eco_saving_litres"] >= 0, f"{road}: cost and eco saving")

# factor sources
e = q.estimate(corsa, 20, "city", "normal", temp_c=25.0)
c.ok(e["factor_source"] == "none" and abs(e["l100"] - e["raw_l100"]) < 1e-9 and e["cost_pkr"] is None,
     "car not in PakWheels and uncalibrated: uncorrected, no cost without a price")
upload(user, alto, [SAMPLES / "gps_trip1.csv", SAMPLES / "gps_trip2.csv"])
e = q.estimate(db.get_car(alto["id"]), 20, "city", "normal", temp_c=25.0, petrol_pkr=280, claimed_kmpl=15)
c.ok(e["factor_source"] == "calibration" and abs(e["l100"] - e["raw_l100"] * e["k"]) < 1e-9,
     "own calibration used once the car has it", f"k {e['k']:.3f}")
c.ok(abs(e["claimed"]["difference_pct"] - (e["kmpl"] / 15 - 1) * 100) < 1e-9 and abs(e["claimed"]["litres"] - 20 / 15) < 1e-9,
     "claimed km/L compared")
try:
    q.estimate(alto, 0, "city", "normal")
    c.ok(False, "distance 0 rejected")
except ValueError:
    c.ok(True, "distance 0 rejected")

# page: estimate shows results, nothing saved until Save
n0 = len(db.get_trips(user["id"]))
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.session_state["car_id"] = corsa["id"]
at.run()
at.switch_page("pages/quick_estimate.py").run()
c.page(at, "quick estimate form")
[n for n in at.number_input if n.label == "Distance (km)"][0].set_value(25.0)
[s for s in at.selectbox if s.label == "Road type"][0].set_value("highway")
[s for s in at.selectbox if s.label == "Driving style"][0].set_value("eco")
[b for b in at.button if b.label == "Estimate"][0].click().run()
c.page(at, "estimate shown")
labels = [m.label for m in at.metric]
c.ok("Fuel economy (km/L)" in labels and "Cost (PKR)" in labels, "result tiles shown")
c.ok(len(db.get_trips(user["id"])) == n0, "nothing saved before clicking Save")
[b for b in at.button if b.label == "Save as trip"][0].click().run()
trips = db.get_trips(user["id"])
c.ok(len(trips) == n0 + 1 and trips[0]["tips"].get("quick_estimate") and trips[0]["route"] is None,
     "Save stores one quick-estimate trip")
at.session_state["trip_id"] = trips[0]["id"]
at.switch_page("pages/trip_detail.py").run()
c.page(at, "trip detail of a quick estimate")
c.ok(any("Quick estimate, not a recorded trip" in i.value for i in at.info), "trip detail says it is an estimate")
c.done()
