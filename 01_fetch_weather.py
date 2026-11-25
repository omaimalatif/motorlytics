"""
Live API source #1: hourly weather from Open-Meteo (free, no API key).

Default: Ann Arbor, Michigan (where VED was recorded), UTC, Oct 2017 to Dec 2018
-> about 11,000 hourly records.
For the app you can fetch a Pakistani city, e.g.
    python 01_fetch_weather.py --lat 33.6844 --lon 73.0479 --tz Asia/Karachi \
        --start 2025-01-01 --end 2025-12-31 --out Data/weather_islamabad.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, default=42.2808)
ap.add_argument("--lon", type=float, default=-83.7430)
ap.add_argument("--tz", default="UTC")   # VED timestamps are UTC, so the weather join is in UTC
ap.add_argument("--start", default="2017-10-01")
ap.add_argument("--end", default="2018-12-31")
ap.add_argument("--out", default="Data/weather_annarbor_api.csv")
a = ap.parse_args()

url = "https://archive-api.open-meteo.com/v1/archive"
params = {"latitude": a.lat, "longitude": a.lon, "start_date": a.start, "end_date": a.end,
          "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
          "timezone": a.tz}
r = requests.get(url, params=params, timeout=60)
r.raise_for_status()
h = r.json()["hourly"]
df = pd.DataFrame(h)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"saved {len(df):,} hourly rows -> {a.out}")
print(df.describe().round(1).to_string())