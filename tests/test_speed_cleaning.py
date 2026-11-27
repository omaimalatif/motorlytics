"""Speed glitches (255 km/h 'no data' readings, single-sample spikes) are removed before features are computed,
in the shared feature code used by the app, the CLI and the enviroCar fetch."""
import io

import numpy as np
import pandas as pd

from _helpers import SAMPLES, Checks, setup, upload

setup("speed_cleaning")
import motorlytics_features as mf  # noqa: E402
from core import app_core, auth, db  # noqa: E402

c = Checks("test_speed_cleaning")

# unit: 255 in the middle (interpolated), spike up and back (interpolated), 255 at the end (dropped)
t = np.arange(0, 20, 1.0)
v = np.full(20, 50.0)
v[5], v[10], v[19] = 255.0, 90.0, 255.0
out = mf.clean_speed(pd.DataFrame({"trip_key": "a", "t": t, "speed": v}))
c.ok(out["speed"].max() < 250 and len(out) == 19, "255 readings and a spike removed", str(out.attrs["speed_cleaning"]))
c.ok(abs(out.loc[out["t"] == 5, "speed"].iloc[0] - 50) < 1e-9 and abs(out.loc[out["t"] == 10, "speed"].iloc[0] - 50) < 1e-9,
     "interior glitches interpolated over time")
c.ok(out.attrs["speed_cleaning"] == {"fixed": 2, "dropped": 1}, "counts reported")

# real driving is kept: hard braking over several readings (same direction), steady acceleration
brake = pd.DataFrame({"trip_key": "b", "t": np.arange(6.0), "speed": [60, 60, 40, 20, 0, 0.0]})
c.ok(mf.clean_speed(brake)["speed"].tolist() == [60, 60, 40, 20, 0, 0], "hard braking (5.6 m/s², 3 readings) kept")
two = pd.DataFrame({"trip_key": ["a"] * 3 + ["b"] * 3, "t": [0, 1, 2, 0, 1, 2.0], "speed": [10, 10, 10, 80, 80, 80.0]})
c.ok(mf.clean_speed(two)["speed"].tolist() == [10, 10, 10, 80, 80, 80], "jumps between trips are not spikes")
burst = pd.DataFrame({"trip_key": "e", "t": np.arange(0, 45, 5.0), "speed": [12, 12, 0, 0, 0, 223, 255, 255, 255.0]})
c.ok(mf.clean_speed(burst)["speed"].tolist() == [12, 12, 0, 0, 0], "glitch burst 0 -> 223 -> 255 removed (incl. the 223)")
rounding = pd.DataFrame({"trip_key": "r", "t": np.arange(0, 0.5, 0.1), "speed": [20, 22, 20, 22, 24.0]})
c.ok(mf.clean_speed(rounding)["speed"].tolist() == [20, 22, 20, 22, 24], "10 Hz whole-km/h rounding steps kept")

# a real log with a 255 spike injected: same trip as without it (distance, fuel, max speed)
base = pd.read_csv(SAMPLES / "ved_mixed_vehicle530.csv")
spiked = base.copy()
mid = len(spiked) // 2
spiked.loc[mid, "speed"] = 255.0                                 # OBD 'no data' byte in the middle
spiked.loc[mid + 200, "speed"] = spiked.loc[mid + 200, "speed"] + 60   # single-sample spike


def trip_of(df):
    d = mf.add_row_features(mf.add_time_and_trips(mf.standardize(df)))
    return mf.aggregate_trips(d).iloc[0], d.attrs["speed_cleaning"]


t0, _ = trip_of(base)
t1, info = trip_of(spiked)
c.ok(t1["max_speed"] < 250, "injected 255 never reaches the features", f"max {t1['max_speed']:.0f} km/h")
c.ok(abs(t1["dist_km"] - t0["dist_km"]) / t0["dist_km"] < 0.005, "distance unchanged by the spikes",
     f"{t0['dist_km']:.3f} vs {t1['dist_km']:.3f} km")
c.ok(abs(t1["l_per_100km"] - t0["l_per_100km"]) < 0.05, "measured fuel unchanged by the spikes",
     f"{t0['l_per_100km']:.2f} vs {t1['l_per_100km']:.2f}")
c.ok(info["fixed"] >= 2, "both glitches fixed", str(info))

# the app's upload path and the CLI path both clean: the sample that ends in 255 readings
db.init_db()
user, _ = auth.register("Clean Tester", "clean@example.com", "secret123", None)
user = db.get_user(user["id"])
car = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4.0)
raw = pd.read_csv(SAMPLES / "gps_trip1.csv")
c.ok(raw["speed"].max() >= 250, "gps_trip1 sample really contains 255 km/h readings")
tid = upload(user, car, [SAMPLES / "gps_trip1.csv"])["trip_ids"][0]
trip = db.get_trip(tid, user["id"])
c.ok(abs(trip["dist_km"] - 6.53) < 0.01 and abs(trip["measured_l100"] - 6.31) < 0.01,
     "upload: distance and measured fuel without the 255 readings", f"{trip['dist_km']} km, {trip['measured_l100']:.2f}")
speeds = [s for _, s in (trip["route"] or {}).get("speed", [])]
c.ok(speeds and max(speeds) < 250, "stored speed trace has no 255 readings")
buf = io.StringIO()
spiked.to_csv(buf, index=False)
res = app_core.predict_logs([pd.read_csv(io.StringIO(buf.getvalue()))], {"disp_L": 2.5, "cylinders": 4})
c.ok(max(s for _, s in res["trips"][0]["speed_trace"]) < 250, "CLI/core path (predict_logs) cleans too")
# a log with nothing usable left after cleaning gets the normal message, not a crash
try:
    app_core.predict_logs([pd.DataFrame({"time": ["2026-01-01 10:00:00", "2026-01-01 10:00:05"], "speed": [255, 255]})],
                          {"disp_L": 1.2, "cylinders": 4})
    c.ok(False, "all-glitch log rejected with a clear message")
except ValueError as e:
    c.ok("1 km" in str(e), "all-glitch log rejected with a clear message", str(e))
c.done()
