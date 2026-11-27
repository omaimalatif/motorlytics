"""Validation IDs: the committed Data/processed/envirocar_validation_ids.csv lists the enviroCar tracks used in the
external validation, _validation_ids() reads it (also without the big Data/envirocar_rows.csv, as online), and
"Fetch a real trip" never returns or downloads one of those tracks (enviroCar API mocked, no internet needed)."""
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from _helpers import ROOT, Checks, setup

setup("validation_ids")
import motorlytics_features as mf  # noqa: E402
from core import envirocar_fetch as ef  # noqa: E402

c = Checks("test_validation_ids")
f = ef.VALIDATION_IDS
c.ok(f == mf.PROC / "envirocar_validation_ids.csv" and f.exists(), "validation id list exists", str(f.name))
committable = subprocess.run(["git", "check-ignore", "-q", os.path.relpath(f, ROOT)], cwd=ROOT).returncode != 0
c.ok(committable, "validation id list is not gitignored (deployed with the app)")
ids = pd.read_csv(f, dtype=str)["track_id"]
c.ok(len(ids) == 1500 and ids.is_unique and ids.str.fullmatch(r"[0-9a-f]{24}").all(), "1,500 unique enviroCar track ids",
     f"{len(ids)} ids")
rows = mf.DATA / "envirocar_rows.csv"
if rows.exists():
    full = set(pd.read_csv(rows, usecols=["trip"])["trip"].astype(str))
    c.ok(set(ids) == full, "the list matches the validation download exactly")

# online there is no Data/envirocar_rows.csv: the committed list alone must be enough
real_data = mf.DATA
mf.DATA = Path(tempfile.mkdtemp())
ef._validation_ids.cache_clear()
val = ef._validation_ids()
mf.DATA = real_data
c.ok(val == frozenset(ids), "read from the committed list without the big download", f"{len(val)} ids")
c.ok(val <= ef.seen_ids(), "validation track ids count as seen")

# mocked enviroCar: a page with 5 validation tracks and 1 new one -> only the new one is downloaded and returned
new_id = "f" * 24
listed = list(ids[:5]) + [new_id]
gasoline = {"properties": {"fuelType": "gasoline", "manufacturer": "Test", "model": "Car", "constructionYear": 2012,
                           "engineDisplacement": 1400}}
t0 = pd.Timestamp("2016-01-01T10:00:00Z")
features = [{"properties": {"time": (t0 + pd.Timedelta(seconds=i)).isoformat(),
                            "phenomenons": {"Speed": {"value": 40.0}, "MAF": {"value": 8.0}, "Rpm": {"value": 1800}}},
             "geometry": {"coordinates": [7.6 + i * 1e-4, 51.9]}} for i in range(300)]
downloaded = []


def fake_get(s, url, timeout, **params):
    if url.endswith("/tracks"):
        return {"tracks": [{"id": i, "sensor": gasoline, "begin": "2016-01-01"} for i in listed]}
    tid = url.rsplit("/", 1)[-1]
    downloaded.append(tid)
    return {"features": features}


real_get, ef._get = ef._get, fake_get
try:
    got = ef.fetch_unseen_track(seed=1)
finally:
    ef._get = real_get
c.ok(got["track_id"] == new_id, "fetch returns the unseen track", got["track_id"])
c.ok(downloaded == [new_id], "validation tracks are never downloaded", ", ".join(downloaded))
c.done()
