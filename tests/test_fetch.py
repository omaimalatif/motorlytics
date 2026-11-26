"""'Fetch a real trip': unseen enviroCar track with real MAF that uploads (internet), cached ids skipped,
fallback to a bundled sample when the fetch fails, and the Upload trip page analyses a fetched log."""
import pandas as pd

from _helpers import ROOT, SAMPLES, Checks, setup

setup("fetch")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import app_core, auth, db, envirocar_fetch as ef  # noqa: E402

c = Checks("test_fetch")
db.init_db()
user, _ = auth.register("Fetch Tester", "fetch@example.com", "secret123", None)
user = db.get_user(user["id"])
car = db.add_car(user["id"], "Toyota", "Corolla", 2015, 1.6, 4.0)

# live fetch (needs internet; reported as NOTE when offline)
try:
    first = ef.fetch_unseen_track(seed=11)
    d = pd.read_csv(pd.io.common.BytesIO(first["bytes"]))
    c.ok(list(d.columns) == ["time", "lat", "lon", "speed", "rpm", "maf", "load"], "fetched log in upload format")
    c.ok(first["maf_share"] >= 0.8 and first["dist_km"] >= 1, "real MAF and at least 1 km",
         f"{first['car']}, {first['dist_km']} km, MAF {first['maf_share']:.0%}")
    c.ok(first["track_id"] in ef.seen_ids(), "fetched track remembered")
    res = app_core.ingest_logs(user, car, [(first["filename"], first["bytes"])])
    t = db.get_trip(res["trip_ids"][0], user["id"])
    c.ok(t["measured_l100"] is not None, "fetched trip uploads with measured fuel", f"{t['measured_l100']:.2f} L/100 km")
    second = ef.fetch_unseen_track(seed=11)                     # same random choices: must skip the cached id
    c.ok(second["track_id"] != first["track_id"], "a second fetch skips the cached track")
except ef.FetchError as e:
    print(f"  NOTE  live fetch skipped: {e}")

# validation ids are never fetched again
val = ef._validation_ids()
if val:
    c.ok(val <= ef.seen_ids(), "validation-set track ids count as seen", f"{len(val)} ids")

# fallback when enviroCar cannot be reached
try:
    ef.fetch_unseen_track(base="https://envirocar.invalid/api", timeout=5)
    c.ok(False, "unreachable server raises FetchError")
except ef.FetchError:
    c.ok(True, "unreachable server raises FetchError")
fb = ef.fallback_sample(seed=0)
c.ok((SAMPLES / fb["filename"]).exists() and fb["bytes"] and fb["disp_L"], "fallback is a bundled sample with its car")
fb2 = ef.fallback_sample(exclude=[fb["filename"]], seed=0)
c.ok(fb2["filename"] != fb["filename"], "fallback avoids samples already used")

# the Upload trip page analyses a fetched log as a separate demo car (bundled sample here: no network needed)
from core import app_core as ac  # noqa: E402

mine = db.add_car(user["id"], "Suzuki", "Cultus", 2018, 1.0, 3.0, nickname="My Cultus")
ac.ingest_logs(user, mine, [("gps_trip1.csv", (SAMPLES / "gps_trip1.csv").read_bytes())])   # mine gets its own k
k_before = ac.calibration_status(mine["car_key"])
n_mine_before = len(db.get_trips(user["id"], mine["id"]))
n0 = len(db.get_trips(user["id"]))
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.session_state["car_id"] = mine["id"]
at.session_state["fetched_log"] = fb
at.run()
at.switch_page("pages/upload_trip.py").run()
c.page(at, "upload page with a fetched log")
c.ok(any(b.label == "Analyse this trip" for b in at.button), "fetched log offered for analysis")
[b for b in at.button if b.label == "Analyse this trip"][0].click().run()
c.page(at, "fetched log analysed")
c.ok(len(db.get_trips(user["id"])) > n0, "fetched log saved as trip(s)")
c.ok(ac.calibration_status(mine["car_key"]) == k_before, "the user's car calibration (k) is unchanged",
     str({t: round(v["k"], 3) for t, v in k_before.items()}))
c.ok(len(db.get_trips(user["id"], mine["id"])) == n_mine_before, "nothing saved under the user's car")
demo = [x for x in db.get_cars(user["id"]) if x.get("is_demo")]
c.ok(len(demo) == 1 and demo[0]["nickname"] == "enviroCar demo, not your car" and demo[0]["disp_L"] == fb["disp_L"],
     "trip saved under a demo car built from the track's vehicle", f"{demo[0]['make']} {demo[0]['model']}" if demo else "")
c.ok(not ac.calibration_status(demo[0]["car_key"]), "the demo car gets no calibration either")
t_demo = db.get_trips(user["id"], demo[0]["id"])[0]
c.ok(t_demo["tips"].get("demo") and not t_demo["calibrated"], "trip flagged as a demo trip")
c.ok(at.session_state["car_id"] == mine["id"], "the selected car stays active")
from ui import session as sess  # noqa: E402

c.ok(all(not x.get("is_demo") for x in sess.user_cars(user)), "demo car not offered as one of the user's cars")
at.session_state["trip_id"] = t_demo["id"]
at.switch_page("pages/trip_detail.py").run()
c.ok(any("not your car" in str(getattr(e, "value", "")) for e in list(at.caption) + list(at.markdown)),
     "trip detail says it is not the user's car")
c.done()
