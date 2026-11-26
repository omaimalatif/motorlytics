"""Time-based charts never use sub-day ticks (11), Overview charts are day / hour bar charts (11), and the trip's
style uses the same names everywhere: eco / normal / aggressive (13)."""
import json
import re

from _helpers import ROOT, SAMPLES, Checks, setup, upload

setup("charts")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, db  # noqa: E402

c = Checks("test_charts")
db.init_db()
user, _ = auth.register("Chart Tester", "charts@example.com", "secret123", None)
db.update_user(user["id"], petrol_pkr=280.0)
user = db.get_user(user["id"])
car = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4.0)
# trips on several days years apart (2016 enviroCar, 2018 VED) and within one day
for f in ["gps_trip1.csv", "gps_trip3.csv", "ved_city_vehicle575.csv", "ved_mixed_vehicle530.csv",
          "envirocar_highway_bmw_1_2013.csv"]:
    upload(user, car, [SAMPLES / f])

DATE_STR = re.compile(r"^\d{4}-\d{2}-\d{2}")
SUBDAY_FMT = re.compile(r"%[HIMSLfp]")


def check_specs(at, page):
    specs = [json.loads(e.proto.spec) for e in at.get("plotly_chart")]
    problems = []
    for n, sp in enumerate(specs):
        lay = sp.get("layout", {})
        axes = {k: v for k, v in lay.items() if re.fullmatch(r"[xy]axis\d*", k)}
        for tr in sp.get("data", []):
            for ax_letter in ("x", "y"):
                vals = tr.get(ax_letter) or []
                if isinstance(vals, dict):                      # plotly binary arrays: numeric, not dates
                    continue
                if any(isinstance(x, str) and DATE_STR.match(x) for x in vals[:5]):
                    ax = axes.get(f"{ax_letter}axis" + str(tr.get(f"{ax_letter}axis", ax_letter))[1:], {})
                    if ax.get("type") != "date":
                        problems.append(f"chart {n}: date values on a non-date axis")
        for name, ax in axes.items():
            if ax.get("type") == "date":
                if not ax.get("tickformat") or SUBDAY_FMT.search(ax["tickformat"]):
                    problems.append(f"chart {n} {name}: tickformat {ax.get('tickformat')!r} is not day-level")
                if not ax.get("dtick") or float(ax["dtick"]) < 86_400_000:
                    problems.append(f"chart {n} {name}: dtick {ax.get('dtick')!r} below one day")
    c.ok(not problems, f"{page}: no chart axis with sub-day ticks", "; ".join(problems))
    return specs


at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.run()
for page in ["overview", "analytics", "trip_history", "calibration"]:
    at.switch_page(f"pages/{page}.py").run()
    c.page(at, page)
    specs = check_specs(at, page)
    if page == "overview":
        day, hour = specs[0], specs[1]
        labels = [x for tr in day["data"] for x in tr["x"]]
        c.ok(all(t["type"] == "bar" for t in day["data"]) and day["layout"]["xaxis"]["type"] == "category"
             and all(re.fullmatch(r"\d{2} [A-Z][a-z]{2} \d{4}", x) for x in labels),
             "fuel by day: bars with day labels like '20 May 2018'", ", ".join(labels[:3]))
        xs = hour["data"][0]["x"]
        xs = xs if isinstance(xs, list) else list(range(24))
        c.ok(hour["data"][0]["type"] == "bar" and len(hour["data"][0]["y"] if isinstance(hour["data"][0]["y"], list)
                                                       else range(24)) == 24 and list(xs) == list(range(24)),
             "driving by hour: 24 bars, hours 0-23")

# (13) style names: the trip header and the eco-score badge use the same eco / normal / aggressive name
problems = []
for t in db.get_trips(user["id"]):
    at.session_state["trip_id"] = t["id"]
    at.switch_page("pages/trip_detail.py").run()
    header = " ".join(e.value for e in at.caption)
    badges = [e.value for e in at.markdown if "badge[" in e.value]
    if f"{t['style']} style" not in header or not any(f"{t['style']} style" in b for b in badges):
        problems.append(f"trip {t['id']} ({t['style']})")
    if any("moderate" in str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.caption)):
        problems.append(f"trip {t['id']}: 'moderate' shown")
c.ok(not problems, "trip header and eco-score badge show the same style name", ", ".join(problems))
c.done()
