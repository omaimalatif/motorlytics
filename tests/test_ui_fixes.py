"""Fixes from manual testing: measured vs predicted (2), hidden Trip detail (3), names as typed (4),
no internal names and correct plurals in the UI (7, 10), quick-estimate calculation line (8), analytics with few trips (9)."""
import io
import json
import re

import pandas as pd

from _helpers import ROOT, SAMPLES, Checks, setup, upload

setup("ui_fixes")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, db  # noqa: E402

c = Checks("test_ui_fixes")
db.init_db()
user, _ = auth.register("UI Tester", "ui@example.com", "secret123", None)
user = db.get_user(user["id"])
car = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4.0, nickname="My Black Beauty", colour="Midnight Blue")

# trip A: MAF (uncalibrated); trip B: speed only; trip C: MAF after the car is calibrated
a = upload(user, car, [SAMPLES / "gps_trip1.csv"])["trip_ids"][0]
speed_only = pd.read_csv(SAMPLES / "ved_mixed_vehicle530.csv")[["time", "speed"]]
buf = io.StringIO()
speed_only.to_csv(buf, index=False)
from core import app_core  # noqa: E402

b = app_core.ingest_logs(user, car, [("speed_only.csv", buf.getvalue().encode())])["trip_ids"][0]
c_trip = upload(user, car, [SAMPLES / "gps_trip3.csv"])["trip_ids"][0]

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.run()

INTERNAL = [r"user\d+_car\d+", r"behaviour_only", r"full_no_load", r"\bspeed_std\b", r"\bidle_frac\b",
            r"stops_per_km", r"hard_(accel|brake)_per_km", r"frac_rpm", r"rpm_p90", r"frac_speed", r"l_per_100km",
            r"\bmodel: ", r"\btier\b", r"\b1 trips\b", r"uncalibrated estimate \("]
texts_seen = {}


def page_texts(at_):
    out = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "subheader", "title", "header"):
        out += [getattr(e, "value", "") for e in getattr(at_, kind)]
    for m in at_.metric:
        out += [m.label, str(m.value), str(getattr(m.proto, "help", ""))]
    for t in at_.tabs:
        out.append(t.label)
    return [str(x) for x in out]


UNIT_IN_VALUE = re.compile(r"L/100|km/L|km|PKR|min|km/h|/ 100")


def scan(at_, page):
    vals = [str(mt.value) for mt in at_.metric if UNIT_IN_VALUE.search(str(mt.value))]
    c.ok(not vals, f"tiles on {page}: unit in the label, number only in the value", ", ".join(vals))
    texts = page_texts(at_)
    texts_seen[page] = texts
    bad = sorted({p for p in INTERNAL for t in texts if re.search(p, t)})
    c.ok(not bad, f"no internal names / bad plurals on {page}", ", ".join(bad))


def metrics(at_):
    return {m.label: m.value for m in at_.metric}


# (2) measured vs predicted on Trip detail
at.session_state["trip_id"] = a
at.switch_page("pages/trip_detail.py").run()
c.page(at, "trip detail (MAF, uncalibrated)")
m = metrics(at)
c.ok("Measured (L/100 km)" in m and re.fullmatch(r"[\d.]+", m["Measured (L/100 km)"]), "MAF trip: Measured tile",
     m.get("Measured (L/100 km)"))
c.ok("Model predicted (L/100 km)" in m and "Model error" in m and m["Model error"].endswith("%"),
     "MAF trip: Model predicted and error % side by side", f"{m.get('Model predicted (L/100 km)')} / {m.get('Model error')}")
scan(at, "trip detail (MAF)")
c.ok(any("Not calibrated yet: this trip's MAF data will calibrate your car" in t for t in texts_seen["trip detail (MAF)"]),
     "friendly calibration line")
c.ok(any("Speed-only model (LWR)" in t for t in texts_seen["trip detail (MAF)"]), "model badge uses the friendly name")
at.session_state["trip_id"] = b
at.switch_page("pages/trip_detail.py").run()
m = metrics(at)
c.ok("Measured (L/100 km)" not in m and "Model error" not in m and any(k.startswith("Model predicted") for k in m),
     "speed-only trip: only the prediction")
scan(at, "trip detail (speed only)")
at.session_state["trip_id"] = c_trip
at.switch_page("pages/trip_detail.py").run()
m = metrics(at)
c.ok("Model predicted after calibration (L/100 km)" in m, "calibrated trip: 'after calibration' label", ", ".join(m))
scan(at, "trip detail (calibrated)")

# (2) Trip history has measured and predicted columns
at.switch_page("pages/trip_history.py").run()
c.page(at, "trip history")
cols = list(at.dataframe[0].value.columns)
c.ok("measured_l100" in cols and "pred_l100" in cols, "history shows measured and predicted columns")
scan(at, "trip history")

# (3) Trip detail is hidden from the sidebar, still reachable; empty state without a selected trip
c.ok('visibility="hidden"' in (ROOT / "app.py").read_text(encoding="utf-8"), "Trip detail hidden in navigation")
at.session_state["trip_id"] = None
at.switch_page("pages/trip_detail.py").run()
c.page(at, "trip detail without a trip")
c.ok(any("No trip selected" in i.value for i in at.info) and any(bt.label == "Go to Trip history" for bt in at.button),
     "empty state with a button to Trip history")

# (4) nickname and colour exactly as typed
at.switch_page("pages/my_car.py").run()
c.page(at, "my car")
md = [e.value for e in at.markdown] + [e.value for e in at.subheader]
c.ok(any("My Black Beauty" in t for t in md) and not any("MY BLACK BEAUTY" in t for t in md), "nickname as typed")
c.ok(any("Midnight Blue" in t for t in md) and not any("MIDNIGHT BLUE" in t for t in md), "colour as typed")
scan(at, "my car")
fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
fresh.session_state["user_id"] = user["id"]
fresh.run()
c.ok(any("My Black Beauty" in str(getattr(e, "value", "")) for e in fresh.sidebar.children.values()),
     "sidebar shows the nickname as typed")

for page in ["overview", "calibration", "about_model", "upload_trip", "analytics"]:
    at.switch_page(f"pages/{page}.py").run()
    c.page(at, page)
    scan(at, page)

# (8) quick estimate shows the model value and the factor
alto = db.add_car(user["id"], "Suzuki", "Alto", 2019, 0.66, 3.0, transmission="Manual")
for chosen, expect in [(alto, "PakWheels factor"), (car, None)]:
    at.session_state["car_id"] = chosen["id"]
    if "quick_est" in at.session_state:
        del at.session_state["quick_est"]
    at.switch_page("pages/quick_estimate.py").run()
    [n for n in at.number_input if n.label == "Distance (km)"][0].set_value(20.0)
    [bt for bt in at.button if bt.label == "Estimate"][0].click().run()
    line = [e.value for e in at.markdown if e.value.startswith("**Model:")]
    if expect:
        c.ok(line and re.search(r"Model: [\d.]+ km/L × PakWheels factor [\d.]+ = [\d.]+ km/L", line[0]),
             "quick estimate: model x PakWheels factor = result", line[0] if line else "")
    else:
        c.ok(line and "no correction applied" in line[0] or (line and "calibration factor" in line[0]),
             "quick estimate: model line without PakWheels", line[0] if line else "")
    scan(at, f"quick estimate ({chosen['make']})")

# (9) analytics with few trips: separate user with 1 trip
u2, _ = auth.register("Few Trips", "few@example.com", "secret123", None)
u2 = db.get_user(u2["id"])
car2 = db.add_car(u2["id"], "Opel", "Corsa", 2011, 1.2, 4.0)
upload(u2, car2, [SAMPLES / "gps_trip1.csv"])
at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at2.session_state["user_id"] = u2["id"]
at2.run()
at2.switch_page("pages/analytics.py").run()
c.page(at2, "analytics with 1 trip")
c.ok(any("Charts fill in as you upload more trips" in e.value for e in at2.caption), "few-trips note")
specs = {json.loads(e.proto.spec)["layout"].get("meta"): json.loads(e.proto.spec) for e in at2.get("plotly_chart")}
infos = [i.value for i in at2.info]
c.ok("fuel_over_time" not in specs and any("at least 2 trips" in i for i in infos),
     "fuel over time: empty state with 1 trip")
c.ok("eco_distribution" not in specs and any("at least 3 trips" in i for i in infos),
     "eco-score distribution: empty state with 1 trip")
bar_text = [x for tr in specs.get("style_road", {}).get("data", []) for x in tr.get("text", [])]
c.ok(bar_text and all(re.fullmatch(r"1 trip|\d+ trips", x) for x in bar_text), "style bars labelled '1 trip' / 'n trips'",
     ", ".join(map(str, bar_text)))
mv = specs.get("measured_vs_predicted", {"data": [], "layout": {}})
names = {tr.get("name") for tr in mv["data"]}
c.ok("coloraxis" not in mv["layout"] and "perfect prediction" in names and names & {"no", "yes"},
     "measured vs predicted: categorical calibrated colours, labelled diagonal", ", ".join(sorted(n for n in names if n)))
c.ok(mv["layout"].get("xaxis", {}).get("range") == mv["layout"].get("yaxis", {}).get("range"), "measured vs predicted: equal axes")
cards = [e.value for e in at2.markdown if e.value.startswith("**")]
c.ok(all(any(title in x for x in cards) for title in ["Fuel use over time", "Fuel use by road type and driving style",
                                                     "Measured vs predicted", "Cost by month (PKR)", "Eco-score distribution"]),
     "five chart cards with titles")
c.done()
