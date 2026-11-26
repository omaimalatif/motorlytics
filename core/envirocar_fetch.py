"""'Fetch a real trip': download one enviroCar track that the app has not used before (no Streamlit code).

enviroCar (https://envirocar.org) publishes OBD-II tracks as open data (ODbL). fetch_unseen_track():
  - lists a random page of older tracks (2014-2019, where tracks have a real MAF sensor rather than a
    calculated one), keeps petrol cars only,
  - skips track ids already used: the validation download (Data/envirocar_rows.csv) and every track this
    function fetched before (data/uploads/fetched_tracks.json),
  - downloads candidates until one has real MAF on >= 80 % of rows and covers >= 1 km,
  - returns it as a CSV in the app's upload format (time, lat, lon, speed, rpm, maf, load) plus the car it
    was recorded with.
If that fails (offline, API down, nothing suitable), fallback_sample() returns a bundled log from samples/.
Nothing personal is sent: only anonymous public API requests.
"""
import io
import json
import random
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

import motorlytics_features as mf
from core import db

BASE = "https://envirocar.org/api/stable"
PAGES = (200, 262)                       # pages of 100 tracks with real MAF (see 01b_fetch_envirocar.py)
MAX_DOWNLOADS = 12
PHEN = {"Speed": "speed", "GPS Speed": "gps_speed", "Rpm": "rpm", "MAF": "maf", "Engine Load": "load"}
SAMPLES = db.ROOT / "samples"


class FetchError(Exception):
    pass


def _cache_path():
    return db.DB_PATH.parent / "uploads" / "fetched_tracks.json"


@lru_cache(maxsize=1)
def _validation_ids() -> frozenset:
    p = mf.DATA / "envirocar_rows.csv"
    if not p.exists():
        return frozenset()
    return frozenset(pd.read_csv(p, usecols=["trip"])["trip"].astype(str).unique())


def seen_ids() -> set:
    p = _cache_path()
    fetched = set(json.loads(p.read_text())) if p.exists() else set()
    return set(_validation_ids()) | fetched


def _remember(track_id: str) -> None:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    ids = set(json.loads(p.read_text())) if p.exists() else set()
    ids.add(track_id)
    p.write_text(json.dumps(sorted(ids)))


def _get(s, url, timeout, **params):
    r = s.get(url, params=params or None, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _track_df(tr: dict) -> pd.DataFrame:
    rows = []
    for f in tr.get("features", []):
        p = f.get("properties", {})
        ph = p.get("phenomenons", {})
        xy = (f.get("geometry") or {}).get("coordinates", [None, None])
        rec = {"time": p.get("time"), "lon": xy[0], "lat": xy[1]}
        for k, v in PHEN.items():
            rec[v] = (ph.get(k) or {}).get("value")
        rows.append(rec)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    for c in ["speed", "gps_speed", "rpm", "maf", "load", "lat", "lon"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["speed"] = d["speed"].fillna(d["gps_speed"])
    return d[["time", "lat", "lon", "speed", "rpm", "maf", "load"]]


def fetch_unseen_track(max_downloads: int = MAX_DOWNLOADS, timeout: int = 20, base: str = BASE, seed=None) -> dict:
    """{'filename', 'bytes', 'track_id', 'car', 'disp_L', 'cylinders', 'dist_km', 'maf_share', 'begin', 'source'}.
    Raises FetchError if no suitable unseen track is found."""
    s = requests.Session()
    s.headers["Accept"] = "application/json"
    rng = random.Random(seed if seed is not None else time.time())
    seen = seen_ids()
    downloads = 0
    try:
        for _ in range(3):                                     # up to 3 random pages
            lst = _get(s, f"{base}/tracks", timeout, limit=100, page=rng.randint(*PAGES))
            cands = [t for t in lst.get("tracks", [])
                     if str((t.get("sensor") or {}).get("properties", {}).get("fuelType", "")).lower() == "gasoline"
                     and t.get("id") not in seen]
            rng.shuffle(cands)
            for t in cands:
                if downloads >= max_downloads:
                    raise FetchError(f"no suitable track among {downloads} downloads")
                downloads += 1
                d = _track_df(_get(s, f"{base}/tracks/{t['id']}", timeout))
                if d.empty or d["maf"].notna().mean() < 0.8:
                    continue
                ts = pd.to_datetime(d["time"], errors="coerce", utc=True, format="mixed")
                chk = mf.clean_speed(d.assign(trip_key="track", t=(ts - ts.min()).dt.total_seconds()))  # no 255 glitches
                dt = chk["t"].diff()
                km = float((chk["speed"] / 3.6 * dt.where(dt.between(0, 5), 0)).sum() / 1000)
                if km < 1:
                    continue
                sp = t["sensor"]["properties"]
                disp = pd.to_numeric(sp.get("engineDisplacement"), errors="coerce") / 1000
                buf = io.StringIO()
                d.to_csv(buf, index=False)
                _remember(t["id"])
                return {"filename": f"envirocar_{t['id']}.csv", "bytes": buf.getvalue().encode(), "track_id": t["id"],
                        "car": f"{sp.get('manufacturer', '?')} {sp.get('model', '?')} {sp.get('constructionYear', '')}".strip(),
                        "make": str(sp.get("manufacturer") or "Unknown").strip(),
                        "model": str(sp.get("model") or "car").strip(),
                        "year": int(sp["constructionYear"]) if str(sp.get("constructionYear", "")).isdigit() else None,
                        "disp_L": None if np.isnan(disp) else round(float(disp), 2),
                        "cylinders": mf.estimate_cylinders(disp), "dist_km": round(km, 2),
                        "maf_share": round(float(d["maf"].notna().mean()), 3), "begin": t.get("begin"),
                        "source": "enviroCar (open data, ODbL)"}
    except FetchError:
        raise
    except (requests.RequestException, ValueError, KeyError) as e:
        raise FetchError(f"enviroCar is not reachable ({type(e).__name__})") from e
    raise FetchError(f"no suitable unseen track found ({downloads} downloads)")


def fallback_sample(exclude=(), seed=None) -> dict:
    """A bundled real trip from samples/ (enviroCar ones first), in the same shape as fetch_unseen_track()."""
    idx = pd.read_csv(SAMPLES / "index.csv")
    if "uploads" in idx:                                     # skip samples the app rejects (e.g. < 1 km)
        idx = idx[idx["uploads"].fillna("yes") != "no"]
    idx = idx[~idx["file"].isin(set(exclude)) & ~idx["file"].str.startswith("ved_raw")]
    if idx.empty:
        idx = pd.read_csv(SAMPLES / "index.csv")
    pref = idx[idx["source"].str.startswith("enviroCar")]
    row = (pref if len(pref) else idx).sample(1, random_state=seed).iloc[0]
    words = str(row["car"]).split()
    year = int(words[-1]) if words and words[-1].isdigit() else None
    return {"filename": row["file"], "bytes": (SAMPLES / row["file"]).read_bytes(), "track_id": None,
            "car": row["car"], "make": words[0] if words else "Unknown",
            "model": " ".join(words[1:-1] if year else words[1:]) or "car", "year": year, "disp_L": float(row["disp_L"]), "cylinders": float(row["cylinders"]),
            "dist_km": float(row["dist_km"]), "maf_share": None, "begin": None,
            "source": f"bundled sample ({row['source']})"}
