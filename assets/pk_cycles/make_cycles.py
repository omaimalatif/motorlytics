"""Generate the two fixed standard drive cycles used to turn PakWheels km/L figures into a per-car factor.

    python assets/pk_cycles/make_cycles.py      (already run; the CSVs are committed)

These are SYNTHETIC speed profiles (1 Hz, speed only), like the standard cycles used in official fuel tests:
  city.csv     Lahore-style stop-and-go: signals every few hundred metres, 25-45 km/h between stops (~20 km/h average)
  highway.csv  motorway (M-2 style): steady 85-100 km/h after a short ramp (~88 km/h average)
They are not anyone's real trip; they only fix "typical city / highway driving" for the comparison.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(2026)


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
    t = pd.Timestamp(start) + pd.to_timedelta(np.arange(len(out)), unit="s")
    return pd.DataFrame({"time": t.strftime("%Y-%m-%d %H:%M:%S"), "speed": np.round(out, 1)})


city = drive([(rng.uniform(25, 45), rng.uniform(25, 80), rng.uniform(15, 60)) for _ in range(20)], "2026-03-10 09:00")
highway = drive([(40, 40, None), (75, 60, None), (rng.uniform(88, 98), 1200, None), (rng.uniform(85, 95), 400, 5)],
                "2026-03-10 11:00")
for name, df in [("city", city), ("highway", highway)]:
    df.to_csv(OUT / f"{name}.csv", index=False)
    v = df["speed"].to_numpy()
    print(f"{name}: {len(df)} s, {v.sum() / 3600:.1f} km, average {v.mean():.1f} km/h")
