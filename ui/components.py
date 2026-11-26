"""Small reusable UI pieces (phase 1: plain st.* calls; colours and CSS come in phase 2)."""
import math
from contextlib import contextmanager

import pandas as pd
import pydeck as pdk
import streamlit as st

BADGE_COLORS = {"good": "green", "info": "blue", "warn": "orange", "bad": "red", "neutral": "gray"}


def page_header(title: str, subtitle: str = None, icon: str = None) -> None:
    st.title(f"{icon} {title}" if icon else title)
    if subtitle:
        st.caption(subtitle)


def metric_tile(label: str, value, help: str = None, delta=None, delta_color: str = "normal") -> None:
    with st.container(border=True):
        st.metric(label, value, delta=delta, delta_color=delta_color, help=help)


@contextmanager
def card(title: str = None, caption: str = None, height: int = None):
    """with card("Title"): ...  -> a bordered container with an optional heading."""
    with st.container(border=True, height=height if height else "content"):
        if title:
            st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        yield


def badge(text: str, kind: str = "neutral", icon: str = None) -> None:
    st.badge(text, color=BADGE_COLORS.get(kind, kind), icon=icon)


def badges(items) -> None:
    """items: [(text, kind)] shown side by side."""
    cols = st.columns(len(items) or 1)
    for c, (text, kind) in zip(cols, items):
        with c:
            badge(text, kind)


def tip_card(tip: dict) -> None:
    with st.container(border=True):
        st.markdown(tip["tip"])
        st.caption(f"Possible saving about {tip['saving_l_per_100km']} L/100 km "
                   f"(your {tip['feature'].replace('_', ' ')}: {tip['your_value']}, eco drivers: {tip['eco_value']})")


def empty_state(text: str, button: str = None, page: str = None) -> None:
    st.info(text)
    if button and page and st.button(button):
        st.switch_page(page)


def route_map(path, height: int = 320, key: str = None) -> None:
    """GPS route as a line with start (green) and end (red) points. path: [[lat, lon], ...]."""
    if not path or len(path) < 2:
        st.info("This log has no GPS route.")
        return
    df = pd.DataFrame(path, columns=["lat", "lon"])
    line = [[lon, lat] for lat, lon in path]
    ends = pd.DataFrame({"lon": [line[0][0], line[-1][0]], "lat": [line[0][1], line[-1][1]],
                         "color": [[30, 160, 90], [220, 60, 50]]})
    span = max(df["lat"].max() - df["lat"].min(), df["lon"].max() - df["lon"].min(), 1e-3)
    zoom = float(max(3, min(16, 9 - math.log2(span))))      # ~city level for a few km of route
    deck = pdk.Deck(
        layers=[pdk.Layer("PathLayer", data=[{"path": line}], get_path="path", get_width=4, width_units="pixels",
                          get_color=[40, 110, 220]),
                pdk.Layer("ScatterplotLayer", data=ends, get_position="[lon, lat]", get_fill_color="color",
                          get_radius=7, radius_units="pixels")],
        initial_view_state=pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()),
                                         zoom=zoom),
        map_style=None)
    st.pydeck_chart(deck, height=height, key=key)
