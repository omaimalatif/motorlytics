"""Eco-score 0-100 from the k-Means driving-style model (models/driving_style.joblib).

A trip's behaviour features (rpa, hard accelerations / brakes per km, mean positive acceleration) are
clipped and standardised exactly as for clustering, then projected onto the line from the eco
cluster centre to the aggressive cluster centre: at the eco centre (or beyond) the score is 100,
at the aggressive centre (or beyond) it is 0, linear in between.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import motorlytics_features as mf


@lru_cache(maxsize=1)
def style_bundle() -> dict:
    return joblib.load(mf.MODELS / "driving_style.joblib")


def _centres(b: dict):
    inv = {v: k for k, v in b["label_map"].items()}
    c = b["kmeans"].cluster_centers_
    return c[inv["eco"]], c[inv["aggressive"]]


def eco_scores(trips: pd.DataFrame, bundle: dict = None) -> np.ndarray:
    """Eco-score (0-100, higher = calmer driving) for each row of a trips table."""
    b = bundle or style_bundle()
    feats = b["features"]
    X = pd.DataFrame({c: trips[c] if c in trips else np.nan for c in feats}, index=trips.index).astype(float)
    X = X.clip(pd.Series(b["clip_lo"]), pd.Series(b["clip_hi"]), axis=1)
    X = X.fillna(pd.Series(b["scaler"].mean_, index=feats))          # missing -> average driver
    Z = b["scaler"].transform(X)
    eco, agg = _centres(b)
    axis = agg - eco
    t = (Z - eco) @ axis / float(axis @ axis)                        # 0 at eco centre, 1 at aggressive centre
    return np.round(100 * (1 - np.clip(t, 0, 1)), 1)


def eco_score(trip: dict | pd.Series, bundle: dict = None) -> float:
    row = pd.DataFrame([dict(trip)])
    return float(eco_scores(row, bundle)[0])


def score_label(score) -> str:
    if score is None or pd.isna(score):
        return "n/a"
    return "eco" if score >= 70 else "normal" if score >= 40 else "aggressive"   # same names as the k-Means styles
