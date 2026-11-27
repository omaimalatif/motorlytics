"""Small reusable UI pieces, styled by ui/theme.py (cards are keyed containers, tiles are st.metric)."""
import math
from contextlib import contextmanager

import pandas as pd
import pydeck as pdk
import streamlit as st

from ui import theme

BADGE_COLORS = {k: v[0] for k, v in theme.BADGE.items()}


def page_header(title: str, subtitle: str = None, icon: str = None, actions: bool = False):
    """The same header on every page: title, one-line subtitle, thin brand accent line. actions=True returns a
    right-hand column for buttons (with ...)."""
    if actions:
        left, right = st.columns([3, 1], vertical_alignment="bottom")
    else:
        left, right = st.container(), None
    with left:
        st.title(f"{icon} {title}" if icon else title)
        if subtitle:
            st.caption(subtitle)
    st.html('<div class="ml-accent-line"></div>')
    return right


def metric_tile(label: str, value, help: str = None, delta=None, delta_color: str = "normal", accent: bool = False,
                chart_data=None, chart_type: str = "area", delta_description: str = None) -> None:
    """One KPI tile (st.metric, styled as a raised card). accent: orange value and a gradient top line;
    chart_data: a small sparkline under the value."""
    with st.container(key=theme.next_key("mlkpi-accent" if accent else "mlkpi")):
        st.metric(label, value, delta=delta, delta_color=delta_color, help=help, chart_data=chart_data,
                  chart_type=chart_type, delta_description=delta_description)


@contextmanager
def card(title: str = None, caption: str = None, height: int = None, kind: str = None):
    """with card("Title"): ...  -> a styled card (keyed container mlcard-...). kind: 'raised', 'hero' or
    'click' (lifts on hover)."""
    key = theme.next_key("mlcard" + (f"-{kind}" if kind else ""))
    with st.container(key=key, height=height if height else "content"):
        if title:
            st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        yield


def chart_card(title: str, takeaway: str = None):
    """Card for a chart: title and a one-line takeaway under it."""
    return card(title, takeaway)


def badge(text: str, kind: str = "neutral", icon: str = None) -> None:
    """Status pill: colour + icon + text (never colour alone)."""
    color, default_icon = theme.BADGE.get(kind, (kind, None))
    st.badge(text, color=color, icon=icon or default_icon)


def style_badge(style: str, suffix: str = " style") -> None:
    """Driving-style pill in the style colour family, always with the name as text."""
    color, ic = theme.STYLE_BADGE.get(style, ("gray", None))
    st.badge(f"{style}{suffix}", color=color, icon=ic)


def badges(items) -> None:
    """items: [(text, kind)] shown side by side."""
    with st.container(horizontal=True, gap="small"):
        for text, kind in items:
            badge(text, kind)


def tip_card(tip: dict) -> None:
    """One eco-driving tip: the advice, a plain comparison with eco drivers and the saving in bold."""
    from core import texts
    with card(kind="raised"):
        st.markdown(f":material/lightbulb: {tip['tip']}")
        st.caption(texts.tip_comparison(tip))
        st.markdown(f"Possible saving **{texts.saving_text(tip['saving_l_per_100km'])}**")


def empty_state(text: str, button: str = None, page: str = None) -> None:
    st.info(text, icon=":material/info:")
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


def fit_view(lats, lons, width_px: int = 480, height_px: int = 320, padding: float = 1.35,
             max_zoom: float = 15.0) -> dict:
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


def route_deck(path, height: int = 320, width_px: int = 480) -> pdk.Deck:
    """pydeck map of a GPS route: orange line (4 px) with start (green) and end (red) dots on the free CARTO
    dark-matter basemap, view fitted to the route. path: [[lat, lon], ...]."""
    line = [[float(lon), float(lat)] for lat, lon in path]
    ends = [{"position": line[0], "color": theme.MAP_START}, {"position": line[-1], "color": theme.MAP_END}]
    view = fit_view([p[1] for p in line], [p[0] for p in line], width_px=width_px, height_px=height)
    return pdk.Deck(
        layers=[pdk.Layer("PathLayer", data=[{"path": line}], get_path="path", get_color=theme.MAP_ROUTE,
                          get_width=4, width_units="pixels", width_min_pixels=3, width_max_pixels=6,
                          cap_rounded=True, joint_rounded=True),
                pdk.Layer("ScatterplotLayer", data=ends, get_position="position", get_fill_color="color",
                          get_radius=6, radius_units="pixels", radius_min_pixels=4, radius_max_pixels=8,
                          stroked=True, get_line_color=[11, 11, 13], line_width_min_pixels=1)],
        initial_view_state=pdk.ViewState(**view, pitch=0, bearing=0),
        map_provider="carto", map_style=pdk.map_styles.CARTO_DARK)


def route_map(path, height: int = 320, key: str = None, width_px: int = 480) -> None:
    """GPS route on a map (see route_deck). width_px: a conservative guess of the map's width on that page, so
    the fitted zoom keeps the whole route visible (a wider map just shows more margin)."""
    if not path or len(path) < 2:
        st.info("This log has no GPS route.")
        return
    st.pydeck_chart(route_deck(path, height=height, width_px=width_px), height=height, key=key)
