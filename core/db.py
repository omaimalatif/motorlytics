"""SQLite storage for the app (data/motorlytics.db, gitignored).

Set MOTORLYTICS_DB to use another file (e.g. for tests).
Tables: users, cars, trips, fillups.
"""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("MOTORLYTICS_DB", ROOT / "data" / "motorlytics.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    city TEXT,
    petrol_pkr REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    make TEXT, model TEXT, year INTEGER,
    disp_L REAL, cylinders INTEGER,
    car_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    car_id INTEGER REFERENCES cars(id) ON DELETE SET NULL,
    started_at TEXT, dist_km REAL, dur_min REAL, avg_speed REAL, road_type TEXT,
    measured_l100 REAL, pred_l100 REAL, tier TEXT, model TEXT, calibrated INTEGER,
    eco_score REAL, style TEXT, cost_pkr REAL,
    tips_json TEXT, route_json TEXT, raw_file TEXT
);
CREATE TABLE IF NOT EXISTS fillups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    car_id INTEGER NOT NULL REFERENCES cars(id) ON DELETE CASCADE,
    litres REAL NOT NULL, km REAL, date TEXT, trip_ids_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_trips_user ON trips(user_id, started_at);
"""


def connect(path=None) -> sqlite3.Connection:
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


# columns added after the first release: added to existing databases by init_db (ALTER TABLE ADD COLUMN)
CAR_EXTRA_COLUMNS = {"nickname": "TEXT", "colour": "TEXT", "transmission": "TEXT", "fuel_type": "TEXT",
                     "body_type": "TEXT", "assembled": "TEXT", "tank_litres": "REAL", "odometer_km": "REAL",
                     "registration": "TEXT",
                     # picture: photo_source = 'user' | 'wikipedia' | 'none' (looked up, nothing found) | NULL
                     "photo_path": "TEXT", "photo_source": "TEXT", "photo_credit": "TEXT",
                     # 1 = demo car built from a fetched enviroCar track / bundled sample (not the user's car)
                     "is_demo": "INTEGER"}
CAR_FIELDS = ["make", "model", "year", "disp_L", "cylinders", *CAR_EXTRA_COLUMNS]


def init_db(path=None) -> None:
    """Create missing tables and add missing columns (safe to run on every start)."""
    with connect(path) as con:
        con.executescript(SCHEMA)
        have = {r["name"] for r in con.execute("PRAGMA table_info(cars)")}
        for col, typ in CAR_EXTRA_COLUMNS.items():
            if col not in have:
                con.execute(f"ALTER TABLE cars ADD COLUMN {col} {typ}")
        # cylinders are whole numbers (older rows were stored as 3.0)
        con.execute("UPDATE cars SET cylinders = CAST(ROUND(cylinders) AS INTEGER) "
                    "WHERE cylinders IS NOT NULL AND typeof(cylinders) != 'integer'")


def _int_cyl(v):
    return None if v is None or v != v else int(round(float(v)))


def _car(r) -> dict:
    """Car row as a dict with cylinders as an int (a REAL column in older databases)."""
    d = dict(r)
    if "cylinders" in d:
        d["cylinders"] = _int_cyl(d["cylinders"])
    return d


def _rows(cur) -> list:
    return [dict(r) for r in cur.fetchall()]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------- users ----------------
def create_user(name, email, password_hash, city=None, petrol_pkr=None) -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO users(name,email,password_hash,city,petrol_pkr,created_at) VALUES (?,?,?,?,?,?)",
                          (name, email.strip().lower(), password_hash, city, petrol_pkr, now()))
        return cur.lastrowid


def get_user(user_id) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(r) if r else None


def get_user_by_email(email) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    return dict(r) if r else None


def update_user(user_id, **fields) -> None:
    allowed = {k: v for k, v in fields.items() if k in ("name", "city", "petrol_pkr", "password_hash")}
    if not allowed:
        return
    sets = ", ".join(f"{k}=?" for k in allowed)
    with connect() as con:
        con.execute(f"UPDATE users SET {sets} WHERE id=?", (*allowed.values(), user_id))


# ---------------- cars ----------------
def add_car(user_id, make, model, year, disp_L, cylinders, **extra) -> dict:
    """extra: any of CAR_EXTRA_COLUMNS (nickname, colour, transmission, fuel_type, body_type, assembled,
    tank_litres, odometer_km, registration)."""
    fields = {"make": make, "model": model, "year": year, "disp_L": disp_L, "cylinders": _int_cyl(cylinders),
              **{k: v for k, v in extra.items() if k in CAR_EXTRA_COLUMNS}}
    with connect() as con:
        cur = con.execute(f"INSERT INTO cars(user_id,{','.join(fields)}) VALUES (?{',?' * len(fields)})",
                          (user_id, *fields.values()))
        car_id = cur.lastrowid
        car_key = f"user{user_id}_car{car_id}"          # key in the calibration store
        con.execute("UPDATE cars SET car_key=? WHERE id=?", (car_key, car_id))
    return get_car(car_id)


def get_car(car_id) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM cars WHERE id=?", (car_id,)).fetchone()
    return _car(r) if r else None


def get_cars(user_id, include_demo: bool = True) -> list:
    q = "SELECT * FROM cars WHERE user_id=?" + ("" if include_demo else " AND COALESCE(is_demo, 0) = 0")
    with connect() as con:
        return [_car(r) for r in con.execute(q + " ORDER BY id", (user_id,)).fetchall()]


def update_car(car_id, user_id, **fields) -> dict | None:
    """Change any of CAR_FIELDS for the user's own car; car_key never changes (calibration stays)."""
    upd = {k: v for k, v in fields.items() if k in CAR_FIELDS}
    if "cylinders" in upd:
        upd["cylinders"] = _int_cyl(upd["cylinders"])
    if upd:
        with connect() as con:
            con.execute(f"UPDATE cars SET {', '.join(f'{k}=?' for k in upd)} WHERE id=? AND user_id=?",
                        (*upd.values(), car_id, user_id))
    return get_car(car_id)


def delete_car(car_id, user_id) -> None:
    with connect() as con:
        con.execute("DELETE FROM cars WHERE id=? AND user_id=?", (car_id, user_id))


# ---------------- trips ----------------
TRIP_COLS = ["user_id", "car_id", "started_at", "dist_km", "dur_min", "avg_speed", "road_type", "measured_l100",
             "pred_l100", "tier", "model", "calibrated", "eco_score", "style", "cost_pkr", "tips_json",
             "route_json", "raw_file"]


def add_trip(**t) -> int:
    for k in ("tips_json", "route_json"):
        if k in t and not isinstance(t[k], str):
            t[k] = json.dumps(t[k])
    vals = [t.get(c) for c in TRIP_COLS]
    with connect() as con:
        cur = con.execute(f"INSERT INTO trips({','.join(TRIP_COLS)}) VALUES ({','.join('?' * len(TRIP_COLS))})", vals)
        return cur.lastrowid


def get_trip(trip_id, user_id) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM trips WHERE id=? AND user_id=?", (trip_id, user_id)).fetchone()
    return _decode(dict(r)) if r else None


def get_trips(user_id, car_id=None) -> list:
    q, args = "SELECT * FROM trips WHERE user_id=?", [user_id]
    if car_id is not None:
        q += " AND car_id=?"
        args.append(car_id)
    with connect() as con:
        return [_decode(r) for r in _rows(con.execute(q + " ORDER BY started_at DESC, id DESC", args))]


def delete_trip(trip_id, user_id) -> None:
    with connect() as con:
        con.execute("DELETE FROM trips WHERE id=? AND user_id=?", (trip_id, user_id))


def _decode(t: dict) -> dict:
    for k in ("tips_json", "route_json"):
        try:
            t[k[:-5]] = json.loads(t[k]) if t.get(k) else None
        except (TypeError, ValueError):
            t[k[:-5]] = None
    return t


# ---------------- fill-ups ----------------
def add_fillup(car_id, litres, km, date, trip_ids) -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO fillups(car_id,litres,km,date,trip_ids_json) VALUES (?,?,?,?,?)",
                          (car_id, litres, km, date, json.dumps(list(trip_ids))))
        return cur.lastrowid


def get_fillups(car_id) -> list:
    with connect() as con:
        rows = _rows(con.execute("SELECT * FROM fillups WHERE car_id=? ORDER BY date DESC, id DESC", (car_id,)))
    for r in rows:
        r["trip_ids"] = json.loads(r["trip_ids_json"] or "[]")
    return rows
