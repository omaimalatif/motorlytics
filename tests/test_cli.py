"""05_predict_trip.py runs on the samples: tiers, measured figure, calibration and fill-up via the command line."""
import json
import subprocess
import sys

from _helpers import ROOT, SAMPLES, Checks, setup

tmp = setup("cli")
c = Checks("test_cli")
cal = tmp / "cli_calibration.json"


def run(*args):
    p = subprocess.run([sys.executable, "05_predict_trip.py", *map(str, args), "--calibration", str(cal),
                        "--out", str(tmp / "out.json")], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


code, out = run(SAMPLES / "gps_trip1.csv", "--disp", 1.2, "--cyl", 4, "--petrol", 280)
c.ok(code == 0 and "model tier: behaviour_only" in out and "measured 6.31" in out, "phone-style log, speed-only tier")
code, out = run(SAMPLES / "ved_raw_vehicle157_highway.csv", "--disp", 3.5, "--cyl", 6)
c.ok(code == 0 and "model tier: full" in out, "raw VED log uses the full tier")
code, out = run(SAMPLES / "gps_trip1.csv", SAMPLES / "gps_trip2.csv", "--disp", 1.2, "--cyl", 4, "--car-id", "cli_car")
c.ok(code == 0 and "calibration store updated for car 'cli_car'" in out, "MAF trips calibrate a car")
code, out = run(SAMPLES / "gps_trip3.csv", "--disp", 1.2, "--cyl", 4, "--car-id", "cli_car", "--no-update")
c.ok(code == 0 and "calibration: calibrated" in out, "next log is predicted calibrated")
code, out = run(SAMPLES / "gps_trip4.csv", "--disp", 1.2, "--cyl", 4, "--car-id", "fill_car", "--fillup-litres", 0.2)
store = json.loads(cal.read_text())
entry = store["cars"]["fill_car"]["entries"][-1]
c.ok(code == 0 and "calibration store updated for car 'fill_car': 1 entry added" in out
     and entry["kind"] == "fillup" and entry["litres"] == 0.2, "fill-up calibration stored")
c.ok(isinstance(json.loads((tmp / "out.json").read_text()), list), "prediction JSON written")
code, out = run(SAMPLES / "gps_trip1.csv", "--disp", 1.2, "--fillup-litres", 5)
c.ok(code != 0 and "needs --car-id" in out, "fill-up without a car id is refused")
c.done()
