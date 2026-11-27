"""Route map view: zoom fitted to the route's bounding box with padding (max zoom 15), so every sample route
stays inside the map on Trip detail and the Overview, even in a narrow (420 px) map."""
import json
import math

import pandas as pd

from _helpers import SAMPLES, Checks, setup

setup("map")
import motorlytics_features as mf  # noqa: E402
from ui.components import route_deck  # noqa: E402

c = Checks("test_map")


def px(lat, lon, z):
    s = 256 * 2 ** z
    return (lon + 180) / 360 * s, (1 - math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) / math.pi) / 2 * s


idx = pd.read_csv(SAMPLES / "index.csv")
for f in idx[idx["uploads"] != "no"]["file"]:
    d = mf.add_row_features(mf.add_time_and_trips(mf.standardize(pd.read_csv(SAMPLES / f))))
    if not {"lat", "lon"} <= set(d.columns):
        continue
    path = d[["lat", "lon"]].dropna().values.tolist()
    for page, w, h in [("trip detail", 480, 340), ("overview", 420, 300)]:
        spec = json.loads(route_deck(path, height=h, width_px=w).to_json())
        v = spec["initialViewState"]
        cx, cy = px(v["latitude"], v["longitude"], v["zoom"])
        pts = [px(a, b, v["zoom"]) for a, b in path]
        half_w, half_h = max(abs(x - cx) for x, _ in pts), max(abs(y - cy) for _, y in pts)
        ok = v["zoom"] <= 15 and half_w <= min(w, 420) / 2 * 0.95 and half_h <= h / 2 * 0.95
        c.ok(ok, f"{f} fits the {page} map", f"zoom {v['zoom']:.1f}, {half_w:.0f}/{w // 2} px wide, {half_h:.0f}/{h // 2} px high")
    layer = spec["layers"][0]
    c.ok(layer["widthUnits"] == "@@=pixels" and layer["getWidth"] == 4 and spec["mapProvider"] == "carto",
         f"{f}: 4 px line on the CARTO basemap")
c.done()
