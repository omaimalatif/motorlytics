"""Motorlytics design system: the single source of truth for colours, type, spacing, the injected CSS, the
Plotly template and the map colours. .streamlit/config.toml repeats the base theme colours (Streamlit reads
them before any Python runs); tests/test_theme.py checks that both agree.

Look: white, sky blue and black; flat (no gradients, glows or hover jumps), thin borders, generous spacing.

    from ui import theme
    theme.setup()                 # once per run (app.py does it for every page)
    theme.plot(fig, height=300)   # every Plotly chart
"""
import html
from pathlib import Path

import streamlit as st          # Plotly is imported only when a chart is drawn (keeps the first page fast)

ROOT = Path(__file__).resolve().parents[1]
# brand: the name only, "Motorlytics", with the first "o" drawn as a gauge (no separate icon)
LOGO = "assets/brand/motorlytics_wordmark.png"      # the wordmark as an image, dark text: public top bar
ICON = "assets/brand/gauge_o.png"                   # the gauge "o" alone: browser tab only (it needs an image)
GAUGE_O_SVG = ROOT / "assets" / "brand" / "gauge_o.svg"   # the gauge "o" inside the live wordmark on the pages
ICONS = ROOT / "assets" / "icons"

# ---------------- colours: white, sky blue, black ----------------
INK = "#0B1220"            # near-black: text, primary buttons, footer
BG = "#FFFFFF"             # page background
SURFACE = "#FFFFFF"        # cards (with a 1px border)
RAISED = "#F5F9FC"         # tiles, inputs, table headers
BORDER = "#E3EAF0"
TEXT = INK
MUTED = "#3E4C5E"          # secondary text: dark enough to read easily (about 9:1 on white)
GRID = "#D5DEE8"            # chart gridlines
SKY = "#D4ECF8"            # sky blue bands: Welcome hero, Sign-in panel
SKY_SOFT = "#EEF7FC"       # soft sky: hover, highlighted cards, upload area
ACCENT = "#1F8FD6"         # sky blue accent: charts, map route, the "lytics" of the name
ACCENT_TEXT = "#14689E"    # sky blue for text on white (readable contrast)
ORANGE = "#EA580C"         # the warm pop colour: active nav item, key numbers, highlights, sparklines
ORANGE_TEXT = "#C2410C"    # orange for small text on white (readable contrast)
ORANGE_SOFT = "#FDBA74"    # orange numbers on the dark tiles
SIDEBAR = INK              # dark sidebar for contrast
SIDE_TEXT = "#E2E8F0"
BLACK = INK                # the black elements (sidebar, footer, primary buttons, highlight tiles)
RED = "#D92D20"            # destructive actions
GRADIENT = INK             # kept name for older callers: a flat colour now (no gradients)

SUCCESS, WARNING, ERROR = "#15803D", "#B45309", "#D92D20"    # always with an icon and text

# charts: dark, saturated colours only (readable on white); style names are always shown as text as well
CHART_BLUE = "#0369A1"     # deep sky blue: single-series charts, predictions
CHART_ORANGE = "#C2410C"   # deep orange: highlights (today, 7-trip average, speed line)
CHART_TEXT = "#1E293B"     # axis tick labels
STYLE_COLORS = {"eco": "#0F766E", "normal": "#A16207", "aggressive": "#B91C1C"}
CATEGORICAL = [CHART_BLUE, INK, CHART_ORANGE]                 # fixed order
MEASURED, PREDICTED, SINGLE = INK, CHART_BLUE, CHART_BLUE

# badge kinds -> Streamlit palette names (their colours are set in config.toml) and a default icon
BADGE = {"good": ("green", ":material/check_circle:"), "info": ("blue", ":material/info:"),
         "warn": ("orange", ":material/warning:"), "bad": ("red", ":material/error:"),
         "neutral": ("gray", None)}
STYLE_BADGE = {"eco": ("green", ":material/eco:"), "normal": ("yellow", ":material/speed:"),
               "aggressive": ("red", ":material/bolt:")}

# ---------------- type, spacing, radius ----------------
FONT_HEAD = "'Plus Jakarta Sans', 'Inter', system-ui, sans-serif"
FONT_BODY = "'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"
SPACE = (4, 8, 12, 16, 24, 32, 48)
R_CARD, R_INPUT, R_PILL = 12, 8, 999

# ---------------- maps (pydeck, RGB) ----------------
MAP_STYLE = "light"                                 # CARTO positron (pdk.map_styles.CARTO_LIGHT)
MAP_ROUTE = [3, 105, 161]                           # CHART_BLUE (dark)
MAP_START = [21, 128, 61]                           # dark green
MAP_END = [185, 28, 28]                             # dark red


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


# ---------------- Plotly template (registered with the first chart) ----------------
def template():
    """Register the 'motorlytics' Plotly template (once), make it the default, and return it."""
    import plotly.graph_objects as go
    import plotly.io as pio
    if "motorlytics" in pio.templates:
        return pio.templates["motorlytics"]
    axis = dict(gridcolor=GRID, linecolor="#94A3B8", zerolinecolor="#94A3B8", tickcolor="rgba(0,0,0,0)",
                showline=False, zeroline=False, tickfont=dict(color=CHART_TEXT, size=12),
                title=dict(font=dict(color=INK, size=13)), automargin=True)
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=CATEGORICAL,
        font=dict(family=FONT_BODY.replace("'", ""), color=CHART_TEXT, size=12),
        xaxis={**axis, "showgrid": False}, yaxis={**axis, "showgrid": True},
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font=dict(color=TEXT, family="Inter, sans-serif", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT, size=12), title=dict(text="")),
        barcornerradius=6, bargap=0.28, bargroupgap=0.08, margin=dict(l=8, r=8, t=28, b=8),
        hovermode="closest", dragmode=False,
    )
    t.data.scatter = [go.Scatter(line=dict(width=2.5), marker=dict(size=9, line=dict(width=0)))]
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)), textfont=dict(color=INK, size=12))]
    t.data.histogram = [go.Histogram(marker=dict(color=SINGLE, line=dict(width=0)))]
    t.data.box = [go.Box(line=dict(width=1.5))]
    pio.templates["motorlytics"] = t
    pio.templates.default = "motorlytics"
    return t


PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False}


def plot(fig, height: int = 300, key: str = None, margin: dict = None) -> None:
    """Show a Plotly figure with the motorlytics template (theme=None, so Streamlit's chart theme does not
    override it)."""
    template()
    fig.update_layout(template="motorlytics", height=height, margin=margin or dict(l=8, r=8, t=28, b=8))
    st.plotly_chart(fig, theme=None, width="stretch", config=PLOT_CONFIG, key=key)


# ---------------- icons and small HTML pieces ----------------
def _data_uri(svg: str) -> str:
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon_css(color: str = INK) -> str:
    """One CSS class per assets/icons/*.svg: .ml-icon-<name> (st.html strips inline <svg> and <img>, so icons
    are background images of the injected style block)."""
    rules = []
    for p in sorted(ICONS.glob("*.svg")):
        svg = p.read_text(encoding="utf-8").strip().replace("currentColor", color)
        rules.append(f'.ml-icon-{p.stem} {{ background-image: url("{_data_uri(svg)}"); }}')
    if GAUGE_O_SVG.exists():
        rules.append(f'.ml-gauge-o {{ background-image: url("{_data_uri(GAUGE_O_SVG.read_text(encoding="utf-8"))}"); }}')
    return "\n".join(rules)


def brand_lockup(size: int = 64) -> str:
    """The logo as HTML: the word 'Motorlytics' in the heading font, the first 'o' a gauge, 'lytics' in sky blue.
    size: font size in px."""
    return (f'<div class="ml-lockup"><span class="ml-wordmark" style="font-size:{size}px" aria-label="Motorlytics">'
            f'M<span class="ml-gauge-o" aria-hidden="true"></span>tor<span class="ml-grad-text">lytics</span></span></div>')


def icon(name: str, size: int = 22) -> str:
    """<span> showing assets/icons/<name>.svg (original line icon, ink); '' if missing."""
    if not (ICONS / f"{name}.svg").exists():
        return ""
    return f'<span class="ml-icon ml-icon-{name}" style="width:{size}px;height:{size}px" aria-hidden="true"></span>'


def esc(text) -> str:
    return html.escape(str(text))


# ---------------- CSS ----------------
CSS = f"""

:root {{
  --ml-ink: {INK}; --ml-bg: {BG}; --ml-surface: {SURFACE}; --ml-raised: {RAISED}; --ml-border: {BORDER};
  --ml-text: {TEXT}; --ml-muted: {MUTED}; --ml-sky: {SKY}; --ml-sky-soft: {SKY_SOFT}; --ml-accent: {ACCENT};
  --ml-accent-text: {ACCENT_TEXT}; --ml-red: {RED}; --ml-r-card: {R_CARD}px; --ml-r-input: {R_INPUT}px;
  --ml-orange: {ORANGE}; --ml-orange-text: {ORANGE_TEXT}; --ml-orange-soft: {ORANGE_SOFT};
}}

/* ---------- base ---------- */
.stApp {{ background: var(--ml-bg); color: var(--ml-text); }}
html, body, .stApp, .stMarkdown, button, input, textarea, select {{ font-family: {FONT_BODY}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: {FONT_HEAD}; font-weight: 700; letter-spacing: -0.015em;
  color: var(--ml-text); }}
.stApp h1 {{ font-size: 2rem; }}
[data-testid="stMain"] {{ overflow-x: hidden; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1280px; }}
[data-testid="stHeader"] {{ background: #FFFFFF; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .stCaption {{ color: var(--ml-muted) !important;
  font-size: .9rem; opacity: 1 !important; }}   /* Streamlit fades captions to 60 %: too light to read */
a {{ color: var(--ml-accent-text); }}
hr {{ border-color: var(--ml-border) !important; }}
.ml-num, [data-testid="stMetricValue"], .ml-stat-value {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }}

/* ---------- page header: a thin divider under the title ---------- */
.ml-accent-line {{ height: 1px; width: 100%; background: var(--ml-border); margin: 14px 0 8px 0; position: relative; }}
.ml-accent-line::before {{ content: ""; position: absolute; left: 0; top: -1px; width: 56px; height: 3px;
  border-radius: 2px; background: var(--ml-orange); }}

/* ---------- cards: st.container(key="mlcard-...") ---------- */
[class*="st-key-mlcard"] {{
  background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: var(--ml-r-card);
  padding: 18px 20px; gap: 12px;
}}
[class*="st-key-mlcard-raised"] {{ background: var(--ml-raised); }}
[class*="st-key-mlcard-hero"] {{ background: var(--ml-sky-soft); border-color: #D6E8F3; }}
[class*="st-key-mlcard-click"] {{ transition: border-color .15s ease; }}
[class*="st-key-mlcard-click"]:hover {{ border-color: var(--ml-orange); }}
[class*="st-key-mlcard"] [data-testid="stMarkdownContainer"] strong {{ color: var(--ml-ink); }}
.ml-takeaway {{ color: var(--ml-muted); font-size: .85rem; margin: 0; }}

/* formula strip (Quick estimate): Model x factor = result */
.st-key-mlformula {{ background: var(--ml-raised); border: 1px solid var(--ml-border); border-left: 3px solid var(--ml-orange);
  border-radius: var(--ml-r-card); padding: 12px 16px; }}
.st-key-mlformula p {{ font-variant-numeric: tabular-nums; margin: 0; }}

/* ---------- metric tiles ---------- */
[data-testid="stMetric"] {{
  background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: var(--ml-r-card);
  padding: 14px 16px; height: 100%;
}}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {{ color: var(--ml-muted) !important; font-size: .82rem; font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-family: {FONT_HEAD}; color: var(--ml-text); font-size: 1.75rem; font-weight: 700; }}
/* never cut labels off with "...": wrap them */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] *,
[data-testid="stMetricDeltaDescription"] {{ white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; }}
[data-testid^="stBaseButton"] p, [data-testid="stPopover"] button p {{ white-space: normal !important;
  overflow: visible !important; text-overflow: clip !important; line-height: 1.25; }}
/* the key number of a page: a dark tile with an orange value */
[class*="st-key-mlkpi-accent"] [data-testid="stMetric"] {{ background: var(--ml-ink); border-color: var(--ml-ink); }}
[class*="st-key-mlkpi-accent"] [data-testid="stMetricLabel"] p, [class*="st-key-mlkpi-accent"] [data-testid="stMetricLabel"] {{
  color: #CBD5E1 !important; }}
[class*="st-key-mlkpi-accent"] [data-testid="stMetricValue"] {{ color: var(--ml-orange-soft); }}
[class*="st-key-mlkpi-accent"] [data-testid="stMetricDelta"] div, [class*="st-key-mlkpi-accent"] [data-testid="stMetricDeltaDescription"] {{
  color: #E2E8F0 !important; }}
[class*="st-key-mlkpi-accent"] [data-testid="stTooltipIcon"] svg, [class*="st-key-mlkpi-accent"] [data-testid="stTooltipHoverTarget"] svg {{
  color: #CBD5E1; stroke: #CBD5E1; }}

/* ---------- buttons: primary solid black, secondary white with a border ---------- */
[data-testid^="stBaseButton-primary"] {{
  background: var(--ml-ink) !important; border: 1px solid var(--ml-ink) !important; color: #FFFFFF !important;
  font-weight: 600; border-radius: var(--ml-r-input); transition: background .15s;
}}
[data-testid^="stBaseButton-primary"]:hover {{ background: #1E293B !important; border-color: #1E293B !important; }}
[data-testid^="stBaseButton-primary"]:disabled {{ background: var(--ml-raised) !important; color: #94A3B8 !important;
  border: 1px solid var(--ml-border) !important; }}
[data-testid^="stBaseButton-secondary"] {{
  background: #FFFFFF !important; border: 1px solid var(--ml-border) !important; color: var(--ml-text) !important;
  border-radius: var(--ml-r-input); font-weight: 500; transition: border-color .15s, background .15s;
}}
[data-testid^="stBaseButton-secondary"]:hover {{ border-color: var(--ml-ink) !important; background: var(--ml-raised) !important; }}
[class*="st-key-mldanger"] [data-testid^="stBaseButton"] {{ background: #FFFFFF !important;
  border: 1px solid {_hex_rgba(RED, .5)} !important; color: {RED} !important; }}
[class*="st-key-mldanger"] [data-testid^="stBaseButton"]:hover {{ background: {_hex_rgba(RED, .06)} !important; }}
[data-testid="stPopover"] button {{ border-radius: var(--ml-r-input); }}

/* ---------- inputs and selects ---------- */
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div, [data-testid="stDateInput"] > div > div, .stTextArea textarea {{
  background: #FFFFFF !important; border-color: var(--ml-border) !important; border-radius: var(--ml-r-input) !important;
}}
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within {{
  border-color: var(--ml-accent) !important; }}
[data-testid="stWidgetLabel"] p {{ color: var(--ml-text); font-weight: 500; font-size: .875rem; }}
[data-testid="stForm"] {{ border: 0; padding: 0; }}

/* ---------- tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {{ gap: 6px; border-bottom: 1px solid var(--ml-border); }}
[data-testid="stTab"] {{ color: var(--ml-muted); font-weight: 600; padding: 8px 14px; }}
[data-testid="stTab"]:hover {{ color: var(--ml-text); }}
[data-testid="stTab"][aria-selected="true"] {{ color: var(--ml-text); box-shadow: inset 0 -3px 0 var(--ml-orange); }}

/* ---------- expanders, dataframes, uploader, alerts ---------- */
[data-testid="stExpander"] details {{ background: var(--ml-surface); border: 1px solid var(--ml-border) !important;
  border-radius: var(--ml-r-card); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--ml-border); border-radius: var(--ml-r-card);
  overflow: hidden; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--ml-sky-soft); border: 1.5px dashed #A9CFE6;
  border-radius: var(--ml-r-card); padding: 28px 20px; transition: border-color .15s; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--ml-accent); }}
[data-testid="stAlertContainer"] {{ border-radius: var(--ml-r-card); }}

/* ---------- sidebar ---------- */
/* dark sidebar: light text, orange bar on the active page */
[data-testid="stSidebar"] {{ background: {SIDEBAR}; border-right: 1px solid {SIDEBAR}; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ color: {SIDE_TEXT}; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
  color: #AEBBCB !important; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 8px; margin: 1px 0; border-left: 3px solid transparent; transition: background .15s; }}
[data-testid="stSidebarNavLink"] span, [data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {{ color: #CBD5E1 !important; }}
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,.06); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(249,115,22,.16); border-left-color: var(--ml-orange); }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: #FFFFFF !important; font-weight: 600; }}
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{ color: var(--ml-orange) !important; }}
[data-testid="stNavSectionHeader"] {{ color: #7DD3FC !important; text-transform: uppercase; letter-spacing: .1em; font-size: .7rem; font-weight: 700; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {{ background: #1E293B !important; border-color: #334155 !important; }}
[data-testid="stSidebar"] hr {{ border-color: #1E293B !important; }}
[data-testid="stSidebarLogo"], [data-testid="stSidebarHeader"] [data-testid="stLogo"] {{ height: 2.4rem !important; max-width: 100%; }}
[data-testid="stSidebarHeader"] {{ height: auto; padding-top: 1rem; }}
[data-testid="stTopNav"] [data-testid="stLogo"], [data-testid="stHeader"] [data-testid="stLogo"] {{ height: 2rem !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ box-shadow: inset 0 -3px 0 var(--ml-orange); }}

/* ---------- badges, pills, brand ---------- */
[data-testid="stMarkdownBadge"], .ml-pill {{ border-radius: {R_PILL}px !important; padding: 2px 10px; font-weight: 600; }}
.ml-pill {{ display: inline-flex; align-items: center; gap: 6px; font-size: .78rem; border: 1px solid var(--ml-border);
  background: var(--ml-raised); color: var(--ml-text); }}
.ml-icon {{ display: inline-block; background-repeat: no-repeat; background-position: center; background-size: contain; }}
.ml-gauge-o {{ display: inline-block; width: .6em; height: .6em; margin: 0 .025em; vertical-align: -0.02em;
  background-size: contain; background-repeat: no-repeat; background-position: center; }}
.ml-lockup {{ display: flex; align-items: center; justify-content: center; flex-wrap: wrap; }}
.ml-wordmark {{ font-family: {FONT_HEAD}; font-weight: 800; letter-spacing: -0.03em; color: var(--ml-ink); line-height: 1;
  white-space: nowrap; }}
.ml-grad-text {{ color: var(--ml-accent); }}

/* ---------- phones (iPhone / Android, up to 640 px) ---------- */
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding: 4.5rem 1rem 3rem 1rem !important; }}   /* clear of the top bar and menu button */
  .stApp h1 {{ font-size: 1.6rem; }}
  .stApp h2 {{ font-size: 1.3rem; }}
  .stApp h3 {{ font-size: 1.15rem; }}
  /* 16 px text in fields: iPhones zoom into anything smaller when you tap it */
  input, textarea, select, [data-testid="stTextInputRootElement"] input, [data-testid="stNumberInputField"] {{
    font-size: 16px !important; }}
  [class*="st-key-mlcard"] {{ padding: 14px 14px; }}
  [data-testid="stMetric"] {{ padding: 10px 12px; }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
  [data-testid="stMetricLabel"] p {{ font-size: .75rem; }}
  /* KPI rows: two tiles per row instead of one long column */
  [data-testid="stHorizontalBlock"]:has([class*="st-key-mlkpi"]):not(:has([class*="st-key-mlcard"])) {{
    flex-direction: row !important; flex-wrap: wrap !important; gap: 8px !important; }}
  [data-testid="stHorizontalBlock"]:has([class*="st-key-mlkpi"]):not(:has([class*="st-key-mlcard"])) > [data-testid="stColumn"] {{
    flex: 1 1 calc(50% - 8px) !important; width: calc(50% - 8px) !important; min-width: calc(50% - 8px) !important; }}
  [data-testid="stTabs"] [role="tablist"] {{ overflow-x: auto; flex-wrap: nowrap; }}
  [data-testid="stTab"] {{ padding: 8px 10px; white-space: nowrap; }}
}}
"""


def setup() -> None:
    """Inject the CSS (one block, every page, icon classes included) and reset the per-run card counter."""
    st.session_state["_ml_card_n"] = 0
    st.html(f"<style>{CSS}\n{icon_css()}</style>")


def next_key(prefix: str) -> str:
    """Unique, stable-per-run container key: 'mlcard-3'."""
    n = st.session_state.get("_ml_card_n", 0) + 1
    st.session_state["_ml_card_n"] = n
    return f"{prefix}-{n}"
