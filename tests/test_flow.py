"""Full app flow: register -> add car -> account -> upload -> trip detail -> history -> every page -> fill-up -> logout."""
from _helpers import SAMPLES, Checks, setup, upload

setup("flow")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import app_core, db  # noqa: E402
from _helpers import ROOT  # noqa: E402

c = Checks("test_flow")
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.run()
c.page(at, "welcome")
c.ok(any(b.label == "Get started" for b in at.button) and any(b.label == "Sign in" for b in at.button),
     "welcome page: Get started and Sign in")
[b for b in at.button if b.label == "Get started"][0].click().run()
c.page(at, "sign in page (from Get started)")
c.ok(len(at.text_input) == 7 and any(t.label == "Repeat password" for t in at.text_input), "sign-in and create-account forms")

# register through the form
at.text_input[2].set_value("Test Driver")
at.text_input(key="reg_email").set_value("test@example.com")
at.text_input[4].set_value("Islamabad")
at.text_input(key="reg_pw").set_value("secret123")
at.text_input[6].set_value("secret123")
[b for b in at.button if b.label == "Create account"][0].click().run()
c.page(at, "overview after register")
uid = at.session_state["user_id"]
c.ok(db.get_user(uid)["password_hash"].startswith("pbkdf2_sha256$"), "password stored as a salted hash")

# add a car manually
at.switch_page("app_pages/my_car.py").run()
at.toggle[0].set_value(True).run()
fields = {t.label: t for t in at.text_input}
fields["Make"].set_value("Opel")
fields["Model"].set_value("Corsa")
[n for n in at.number_input if n.label == "Year"][0].set_value(2011)
[n for n in at.number_input if n.label == "Engine size (L)"][0].set_value(1.2)
at.run()
[b for b in at.button if b.label == "Add car"][0].click().run()
c.page(at, "car added")
car = db.get_cars(uid)[0]
c.ok(car["disp_L"] == 1.2 and car["car_key"], "car saved with engine size and calibration key")

# petrol price
at.switch_page("app_pages/account.py").run()
at.number_input[0].set_value(280.0)
[b for b in at.button if b.label == "Save"][0].click().run()
c.ok(db.get_user(uid)["petrol_pkr"] == 280.0, "petrol price saved")

# upload 3 MAF trips (same call as the Upload trip page)
at.switch_page("app_pages/upload_trip.py").run()
c.page(at, "upload page")
res = upload(db.get_user(uid), car, [SAMPLES / f"corsa_d_2010_trip{i}.csv" for i in (1, 2, 3)], temp=18.0)
c.ok(len(res["trip_ids"]) == 3, "3 trips saved", str(res["trip_ids"]))
c.ok(app_core.calibration_status(car["car_key"]).get("behaviour_only") is not None, "MAF trips calibrated the car")
at.session_state["last_upload"] = res["trip_ids"]
at.session_state["trip_id"] = res["trip_ids"][0]

for page in ["trip_detail", "trip_history", "overview", "analytics", "calibration", "about_model", "my_car",
             "account", "quick_estimate"]:
    at.switch_page(f"app_pages/{page}.py").run()
    c.page(at, page)
    if page == "trip_detail":
        metrics = dict((m.label, m.value) for m in at.metric)
        c.ok("Measured (L/100 km)" in metrics and "Model predicted (L/100 km)" in metrics,
             "trip detail shows measured and predicted fuel",
             f"{metrics.get('Measured (L/100 km)')} / {metrics.get('Model predicted (L/100 km)')}")

# fill-up through the form
at.switch_page("app_pages/calibration.py").run()
at.number_input[0].set_value(0.4)
at.multiselect[0].set_value(res["trip_ids"])
[b for b in at.button if b.label == "Save fill-up"][0].click().run()
c.page(at, "fill-up saved")
c.ok(len(db.get_fillups(car["id"])) == 1, "fill-up stored")

# a 4th trip is predicted with the car's calibration
r2 = upload(db.get_user(uid), car, [SAMPLES / "corsa_d_2010_trip4.csv"], temp=18.0)
t4 = db.get_trip(r2["trip_ids"][0], uid)
c.ok(bool(t4["calibrated"]), "4th trip uses the car's calibration", r2["calibration"][:60])
c.ok(len((t4["route"] or {}).get("path") or []) > 10, "route stored")

at.switch_page("app_pages/account.py").run()
[b for b in at.button if b.label == "Sign out"][0].click().run()
c.ok("user_id" not in at.session_state, "logged out")
c.page(at, "public page after logout")
c.ok(any(b.label == "Get started" for b in at.button) or len(at.text_input) == 7, "back on the public pages")
c.done()
