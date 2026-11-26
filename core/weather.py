"""Current temperature for a city from Open-Meteo (free, no API key), cached for 30 minutes.

Only the city name is sent (to Open-Meteo's geocoding API), then its coordinates (to the forecast API).
"""
import time

import requests

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
TTL_S = 30 * 60
_cache: dict = {}


def _get(url, **params):
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def geocode(city: str) -> dict | None:
    """{'name', 'country', 'lat', 'lon', 'timezone'} for the best match, or None."""
    if not city:
        return None
    key = ("geo", city.strip().lower())
    if key in _cache:
        return _cache[key][1]
    res = (_get(GEOCODE, name=city.strip(), count=1, language="en", format="json").get("results") or [None])[0]
    out = None if not res else {"name": res.get("name"), "country": res.get("country"),
                                "lat": res["latitude"], "lon": res["longitude"], "timezone": res.get("timezone")}
    _cache[key] = (time.time(), out)                               # place names do not change: no expiry
    return out


def current(city: str) -> dict | None:
    """{'city', 'country', 'temp_c', 'wind_kmh', 'time', 'cached'} or None if the city is unknown or offline."""
    key = ("now", (city or "").strip().lower())
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL_S:
        return {**hit[1], "cached": True}
    try:
        g = geocode(city)
        if not g:
            return None
        w = _get(FORECAST, latitude=g["lat"], longitude=g["lon"], current="temperature_2m,wind_speed_10m",
                 timezone="auto").get("current", {})
        out = {"city": g["name"], "country": g["country"], "temp_c": w.get("temperature_2m"),
               "wind_kmh": w.get("wind_speed_10m"), "time": w.get("time")}
    except (requests.RequestException, ValueError, KeyError):
        return None
    _cache[key] = (time.time(), out)
    return {**out, "cached": False}
