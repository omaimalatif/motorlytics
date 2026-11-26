"""Generate the SYNTHETIC Suzuki Alto trip logs in this folder (not real data).

    python tests/fixtures/make_synthetic_alto.py      (already run; the CSVs are committed)

Three made-up Lahore trips at 1 Hz with speed and GPS, like a phone app log: city stop-and-go (Gulberg),
mixed (Canal Road) and motorway (M-2). Used by test_pakwheels.py and demo_account.py to show what the app does
with a small Pakistani car. They contain no measured fuel, so they cannot show whether a prediction is right.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(7)


def drive(segments, start):
    v, out = 0.0, []
    for target, cruise, stop in segments:
        while v < target - 0.5:
            v = min(target, v + rng.uniform(3.6, 6.5) * max(0.3, 1 - v / (target + 10)))
            out.append(v)
        for _ in range(int(cruise)):
            v = float(np.clip(v + rng.normal(0, 0.8), target - 6, target + 6))
            out.append(v)
        if stop is not None:
            while v > 0.5:
                v = max(0.0, v - rng.uniform(5.4, 9.0))
                out.append(v)
            out += [0.0] * int(stop)
    speed = np.array(out)
    return pd.Timestamp(start) + pd.to_timedelta(np.arange(len(speed)), unit="s"), speed


def route(points, speed):
    pts = np.array(points, float)
    seg = np.sqrt(((np.diff(pts, axis=0) * [111_000, 95_000]) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    d = np.cumsum(speed / 3.6)
    d = d / d[-1] * cum[-1]
    return (np.interp(d, cum, pts[:, 0]) + rng.normal(0, 0.00002, len(d)),
            np.interp(d, cum, pts[:, 1]) + rng.normal(0, 0.00002, len(d)))


TRIPS = {
    "SYNTHETIC_alto_lahore_city": (
        [(rng.uniform(28, 45), rng.uniform(25, 80), rng.uniform(15, 70)) for _ in range(18)], "2026-11-24 08:40",
        [(31.5106, 74.3432), (31.5163, 74.3492), (31.5040, 74.3310), (31.4880, 74.3180), (31.4730, 74.3080)]),
    "SYNTHETIC_alto_lahore_mixed": (
        [(rng.uniform(45, 60), rng.uniform(80, 200), rng.uniform(10, 40)) for _ in range(8)], "2026-11-24 17:55",
        [(31.4730, 74.3080), (31.4900, 74.3000), (31.5100, 74.3200), (31.5300, 74.3350), (31.5500, 74.3420)]),
    "SYNTHETIC_alto_m2_motorway": (
        [(35, 40, None), (75, 60, None), (rng.uniform(88, 98), 1100, None), (rng.uniform(85, 95), 400, 5)],
        "2026-11-25 10:10", [(31.6100, 74.2500), (31.6600, 74.1700), (31.7300, 74.0700), (31.7700, 74.0000)]),
}
if __name__ == "__main__":
    for name, (segs, start, pts) in TRIPS.items():
        t, v = drive(segs, start)
        lat, lon = route(pts, v)
        pd.DataFrame({"time": t.strftime("%Y-%m-%d %H:%M:%S"), "lat": lat.round(6), "lon": lon.round(6),
                      "speed": v.round(1)}).to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(v)} s, {v.sum() / 3600:.1f} km, average {v.mean():.1f} km/h")
