"""My car: best / least efficient trip cards only with 2+ trips of at least 3 km; one such trip shows a single
'Your trip' card; none shows a short note."""
from _helpers import ROOT, SAMPLES, Checks, setup, upload

setup("my_car_trips")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, cars, db  # noqa: E402

c = Checks("test_my_car_trips")
db.init_db()
user, _ = auth.register("Trips Tester", "trips@example.com", "secret123", None)
user = db.get_user(user["id"])
car = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4.0)
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.run()


def cards():
    at.switch_page("app_pages/my_car.py").run()
    c.page(at, f"my car with {cars.car_profile(car['id'])['long_trips']} long trip(s)")
    md = [e.value for e in at.markdown]
    caps = [e.value for e in at.caption]
    return md, caps


upload(user, car, [SAMPLES / "gps_trip3.csv"])                    # 2.83 km: not a long trip
md, caps = cards()
c.ok(not any("Best trip" in m or "Least efficient" in m or "Your trip" in m for m in md)
     and any("No trips of 3 km or more yet" in x for x in caps), "no trip of 3+ km: short note only")

upload(user, car, [SAMPLES / "gps_trip1.csv"])                    # 6.53 km: one long trip
md, caps = cards()
c.ok(any("Your trip" in m for m in md) and not any("Best trip" in m or "Least efficient" in m for m in md),
     "one trip of 3+ km: a single 'Your trip' card, not best and worst")
c.ok(any("Upload more trips to compare your best and least efficient" in x for x in caps), "asks for more trips")
c.ok(sum(b.label == "Open trip" for b in at.button) == 1, "one Open trip button")

upload(user, car, [SAMPLES / "envirocar_mixed_opel_corsa_d_2013.csv"])   # 11.8 km: two long trips
md, caps = cards()
prof = cars.car_profile(car["id"])
c.ok(any("Best trip" in m for m in md) and any("Least efficient" in m for m in md) and not any("Your trip" in m for m in md),
     "two trips of 3+ km: best and least efficient cards")
c.ok(prof["best_trip"]["id"] != prof["worst_trip"]["id"], "best and least efficient are different trips")
c.done()
