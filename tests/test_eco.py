"""Eco-score card: the 'eco driver' estimate is capped at a plausible saving (max 25 % below the trip, never
below the eco 25th percentile for the road type in model units), cleaned samples score sensibly, and the card
names what the saving comes from."""
import pandas as pd

from _helpers import ROOT, SAMPLES, Checks, setup, upload

setup("eco")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import app_core, auth, db, texts  # noqa: E402

c = Checks("test_eco")
idx = pd.read_csv(SAMPLES / "index.csv")
for _, s in idx[idx["uploads"] != "no"].iterrows():
    res = app_core.predict_logs([pd.read_csv(SAMPLES / s["file"])], {"disp_L": s["disp_L"], "cylinders": s["cylinders"]})
    for r in res["trips"]:
        main = r.get("measured_l_per_100km_from_maf", r["pred_l_per_100km"])
        eco = r["eco_l_per_100km"]
        c.ok(0.75 * main - 0.01 <= eco <= main + 0.01 and 0 <= r["possible_saving_pct"] <= 25,
             f"{s['file']}: eco estimate within 25 % of the trip", f"trip {main:.2f}, eco {eco:.2f}, -{r['possible_saving_pct']:.0f}%")

# gps_trip4 used to score 0 / aggressive with 'eco driver 2.6 L/100 km' because of the 255 km/h glitches
r4 = app_core.predict_logs([pd.read_csv(SAMPLES / "gps_trip4.csv")], {"disp_L": 1.2, "cylinders": 4})["trips"][0]
c.ok(r4["eco_score"] >= 70 and r4["style"] != "aggressive", "gps_trip4 is no longer scored 0 / aggressive",
     f"eco-score {r4['eco_score']:.0f}, {r4['style']}")
c.ok(r4["eco_l_per_100km"] >= 0.75 * r4["measured_l_per_100km_from_maf"], "gps_trip4 eco estimate is plausible",
     f"{r4['eco_l_per_100km']:.2f} vs measured {r4['measured_l_per_100km_from_maf']:.2f}")

# wording names the source of the saving
s1 = texts.eco_summary(100, "eco", [{"feature": "idle_frac", "saving_l_per_100km": 1.2}], 8.0, 20.6)
c.ok(s1 == "Smooth driving. Less idling could bring this to about 8.0 L/100 km (up to 21% less).", "wording with a cause", s1)
s2 = texts.eco_summary(30, "aggressive", [{"feature": "hard_accel_per_km", "saving_l_per_100km": 0.4},
                                          {"feature": "rpa", "saving_l_per_100km": 0.6}], 9.1, 12)
c.ok(s2.startswith("Frequent hard acceleration or braking. A steadier pace and gentler acceleration"), "two causes", s2)
c.ok(texts.eco_summary(100, "eco", [], 7.29, 0).endswith("already close to eco driving for its road type."), "no saving")

# the card on Trip detail
db.init_db()
user, _ = auth.register("Eco Tester", "eco@example.com", "secret123", None)
user = db.get_user(user["id"])
car = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4.0)
tid = upload(user, car, [SAMPLES / "gps_trip3.csv"])["trip_ids"][0]
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.session_state["trip_id"] = tid
at.run()
at.switch_page("app_pages/trip_detail.py").run()
c.page(at, "trip detail")
caps = [e.value for e in at.caption]
c.ok(any("could bring this to about" in x or "already close to eco driving" in x for x in caps),
     "eco-score card explains the saving", next((x for x in caps if "could bring" in x or "already close" in x), ""))
c.ok(not any("Driven like an eco driver" in x for x in caps), "old wording gone")
c.done()
