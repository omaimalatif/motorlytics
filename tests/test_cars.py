"""Car details: catalogue suggestions, saving, masking, editing, CNG warning, car profile numbers, delete."""
import pandas as pd

from _helpers import ROOT, SAMPLES, Checks, setup, upload

setup("cars")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, cars, db  # noqa: E402

c = Checks("test_cars")
db.init_db()
user, _ = auth.register("Test Driver", "cars@example.com", "secret123", "Lahore")
db.update_user(user["id"], petrol_pkr=280.0)
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.run()

# catalogue suggestions for a Suzuki Cultus 2018
at.switch_page("pages/my_car.py").run()
at.selectbox[0].set_value("Suzuki").run()
at.selectbox[1].set_value("Cultus").run()
at.selectbox[2].set_value(2018).run()
c.page(at, "Cultus picked")
sfx = "Suzuki_Cultus_2018"
c.ok(at.selectbox(key=f"add_disp_{sfx}").value == 1.0, "engine size suggested (1.0 L)")
c.ok(at.selectbox(key=f"add_{sfx}_trans").value == "Manual", "transmission suggested")
c.ok(at.selectbox(key=f"add_{sfx}_fuel").value == "petrol", "fuel type suggested")
at.text_input(key=f"add_{sfx}_nick").set_value("Silver Cultus")
at.text_input(key=f"add_{sfx}_reg").set_value("lea-19-1234")
at.number_input(key=f"add_{sfx}_tank").set_value(35.0)
at.run()
[b for b in at.button if b.label == "Add car"][0].click().run()
c.page(at, "car added")
car = db.get_cars(user["id"])[0]
c.ok(car["nickname"] == "Silver Cultus" and car["registration"] == "LEA-19-1234", "details saved")
c.ok(not any("LEA191234" in m.value.replace("-", "") for m in at.markdown), "full registration never shown")
c.ok(any("•" in m.value for m in at.markdown), "masked registration shown")

# edit: fuel type CNG -> warning on upload page
cid = car["id"]
at.selectbox(key=f"edit{cid}_fuel").set_value("CNG")
at.number_input(key=f"edit{cid}_odo").set_value(85000.0)
[b for b in at.button if b.label == "Save changes"][0].click().run()
c.ok(db.get_car(cid)["fuel_type"] == "CNG" and db.get_car(cid)["car_key"] == car["car_key"],
     "edit saved, calibration key unchanged")
at.switch_page("pages/upload_trip.py").run()
c.ok(any("CNG car" in w.value for w in at.warning), "CNG warning on upload page")
db.update_car(cid, user["id"], fuel_type="petrol")

# profile numbers = direct computation
upload(db.get_user(user["id"]), db.get_car(cid), [SAMPLES / f"gps_trip{i}.csv" for i in (1, 3, 4)], temp=20.0)
prof = cars.car_profile(cid)
t = pd.DataFrame(db.get_trips(user["id"], cid))
t["l100"] = t["measured_l100"].fillna(t["pred_l100"])
lit = float((t["l100"] * t["dist_km"] / 100).sum())
c.ok(prof["trips"] == 3 and abs(prof["km"] - t["dist_km"].sum()) < 1e-6, "profile trips and km")
c.ok(abs(prof["litres"] - lit) < 1e-6 and abs(prof["avg_l100"] - lit / t["dist_km"].sum() * 100) < 1e-6,
     "profile litres and average")
c.ok(abs(prof["range_km"] - 35 / prof["avg_l100"] * 100) < 1e-6, "full-tank range")
at.switch_page("pages/my_car.py").run()
c.page(at, "my car with trips")

# second car, delete needs confirmation
at.toggle[0].set_value(True).run()
fields = {ti.label: ti for ti in at.text_input}
fields["Make"].set_value("Honda")
fields["Model"].set_value("City")
[n for n in at.number_input if n.label == "Engine size (L)"][0].set_value(1.3)
at.run()
[b for b in at.button if b.label == "Add car"][0].click().run()
second = db.get_cars(user["id"])[-1]
c.ok([b for b in at.button if b.label == "Delete car"][0].disabled, "delete disabled before confirming")
at.checkbox(key=f"sure_{second['id']}").check().run()
[b for b in at.button if b.label == "Delete car"][0].click().run()
c.ok(len(db.get_cars(user["id"])) == 1, "car deleted after confirming")

# migration: init_db on an existing database is safe and idempotent
db.init_db()
db.init_db()
c.ok(db.get_car(cid) is not None, "init_db twice keeps data")
c.done()
