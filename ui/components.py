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
    """One eco-driving tip: the advice, a plain comparison with eco drivers and the rounded saving."""
    from core import texts
    with st.container(border=True):
        st.markdown(tip["tip"])
        st.caption(texts.tip_comparison(tip) + f" Possible saving {texts.saving_text(tip['saving_l_per_100km'])}.")


def empty_state(text: str, button: str = None, page: str = None) -> None:
    st.info(text)
    if button and page and st.button(button):
        st.switch_page(page)


def car_photo(car: dict, width="stretch", caption: bool = True) -> None:
    """The car's picture: the user's photo, else a credited example photo, else a body-type drawing."""
    from pathlib import Path

    from core import car_photos
    pic = car_photos.car_picture(car)
    if pic["kind"] == "drawing":
        st.image(Path(pic["path"]).read_text(encoding="utf-8"), width=width)   # SVG markup
    else:
        st.image(pic["path"], width=width)
    if caption:
        text = pic["caption"]
        if pic.get("credit_url"):
            text += f" ([source]({pic['credit_url']}))"
        st.caption(text)


def fit_view(lats, lons, width_px: int = 700, height_px: int = 320, padding: float = 1.25,
             max_zoom: float = 16.0) -> dict:
    """Centre and zoom (Web Mercator) so the bounding box fits the map with some padding."""
    lat0, lat1, lon0, lon1 = min(lats), max(lats), min(lons), max(lons)

    def merc_y(lat):
        lat = max(min(lat, 85.0), -85.0)
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    lon_span = max((lon1 - lon0) * padding, 1e-5)
    y_span = max((merc_y(lat1) - merc_y(lat0)) * padding, 1e-7)
    zoom_x = math.log2(width_px * 360.0 / (256.0 * lon_span))
    zoom_y = math.log2(height_px * 2 * math.pi / (256.0 * y_span))
    return {"latitude": (lat0 + lat1) / 2, "longitude": (lon0 + lon1) / 2,
            "zoom": float(max(1.0, min(max_zoom, zoom_x, zoom_y)))}


def route_deck(path, height: int = 320, width_px: int = 700) -> pdk.Deck:
    """pydeck map of a GPS route: blue line (4 px) with start (green) and end (red) dots, free CARTO light
    basemap, view fitted to the route. path: [[lat, lon], ...]."""
    line = [[float(lon), float(lat)] for lat, lon in path]
    ends = [{"position": line[0], "color": [30, 160, 90]}, {"position": line[-1], "color": [220, 60, 50]}]
    view = fit_view([p[1] for p in line], [p[0] for p in line], width_px=width_px, height_px=height)
    return pdk.Deck(
        layers=[pdk.Layer("PathLayer", data=[{"path": line}], get_path="path", get_color=[40, 110, 220],
                          get_width=4, width_units="pixels", width_min_pixels=3, width_max_pixels=6,
                          cap_rounded=True, joint_rounded=True),
                pdk.Layer("ScatterplotLayer", data=ends, get_position="position", get_fill_color="color",
                          get_radius=6, radius_units="pixels", radius_min_pixels=4, radius_max_pixels=8,
                          stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1)],
        initial_view_state=pdk.ViewState(**view, pitch=0, bearing=0),
        map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT)


def route_map(path, height: int = 320, key: str = None) -> None:
    """GPS route on a map (see route_deck)."""
    if not path or len(path) < 2:
        st.info("This log has no GPS route.")
        return
    st.pydeck_chart(route_deck(path, height=height), height=height, key=key)
