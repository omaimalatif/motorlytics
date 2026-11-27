"""Motorlytics design system: the single source of truth for colours, type, spacing, the injected CSS, the
Plotly template and the map colours. .streamlit/config.toml repeats the base theme colours (Streamlit reads
them before any Python runs); tests/test_theme.py checks that both agree.

    from ui import theme
    theme.setup()                 # once per run (app.py does it for every page)
    theme.plot(fig, height=300)   # every Plotly chart
"""
import html
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# brand: a minimal gauge mark in the brand gradient (original); the first logo stays in assets/brand, unused
LOGO = "assets/brand/motorlytics_lockup.png"        # mark + wordmark, transparent: sidebar (st.logo)
ICON = "assets/brand/motorlytics_mark.png"          # mark only: browser tab, collapsed sidebar
MARK_SVG = ROOT / "assets" / "brand" / "motorlytics_mark.svg"   # mark for the HTML lockup on public pages
ICONS = ROOT / "assets" / "icons"
IMAGES = ROOT / "assets" / "images"

# ---------------- colours ----------------
BLACK = "#000000"          # Welcome hero, Sign-in brand panel, sidebar
BG = "#0B0B0D"             # page background
SURFACE = "#141417"        # cards
RAISED = "#1B1B20"         # raised cards, tiles, inputs
BORDER = "#2A2A31"
TEXT = "#F4F4F5"
MUTED = "#A1A1AA"
GRID = "#26262C"

CRIMSON = "#80031F"        # subtle glows, gradient start
RED = "#DF3511"            # accents, destructive actions
ORANGE = "#F26A10"         # primary buttons, active nav item, key numbers
GRADIENT = "linear-gradient(135deg, #80031F 0%, #DF3511 55%, #F26A10 100%)"

SUCCESS, WARNING, ERROR = "#22C55E", "#F59E0B", "#DF3511"    # always with an icon and text

# charts (colour-blind checked on the dark background); style names are always shown as text as well
STYLE_COLORS = {"eco": "#0E9384", "normal": "#C98500", "aggressive": "#E5484D"}
CATEGORICAL = ["#E8650A", "#0D9488", "#8B5CF6"]               # fixed order
MEASURED, PREDICTED, SINGLE = "#0D9488", "#E8650A", "#E8650A"

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
R_CARD, R_INPUT, R_PILL = 12, 10, 999

# ---------------- maps (pydeck, RGB) ----------------
MAP_STYLE = "dark"                                  # CARTO dark-matter (pdk.map_styles.CARTO_DARK)
MAP_ROUTE = [242, 106, 16]                          # ORANGE
MAP_START = [34, 197, 94]                           # SUCCESS green
MAP_END = [223, 53, 17]                             # RED


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


# ---------------- Plotly template ----------------
def _template() -> go.layout.Template:
    axis = dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickcolor="rgba(0,0,0,0)", showline=False,
                zeroline=False, tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=MUTED, size=12)),
                automargin=True)
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=CATEGORICAL,
        font=dict(family=FONT_BODY.replace("'", ""), color=MUTED, size=12),
        xaxis={**axis, "showgrid": False}, yaxis={**axis, "showgrid": True},
        hoverlabel=dict(bgcolor=RAISED, bordercolor=ORANGE, font=dict(color=TEXT, family="Inter, sans-serif", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT, size=12), title=dict(text="")),
        barcornerradius=6, bargap=0.28, bargroupgap=0.08, margin=dict(l=8, r=8, t=28, b=8),
        hovermode="closest", dragmode=False, uniformtext=dict(minsize=10, mode="hide"),
    )
    t.data.scatter = [go.Scatter(line=dict(width=2.5), marker=dict(size=8, line=dict(width=0)))]
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)), textfont=dict(color=TEXT, size=11))]
    t.data.histogram = [go.Histogram(marker=dict(color=SINGLE, line=dict(width=0)))]
    t.data.box = [go.Box(line=dict(width=1.5))]
    return t


pio.templates["motorlytics"] = _template()
pio.templates.default = "motorlytics"
PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False}


def plot(fig, height: int = 300, key: str = None, margin: dict = None) -> None:
    """Show a Plotly figure with the motorlytics template (theme=None, so Streamlit's chart theme does not
    override it)."""
    fig.update_layout(template="motorlytics", height=height, margin=margin or dict(l=8, r=8, t=28, b=8))
    st.plotly_chart(fig, theme=None, width="stretch", config=PLOT_CONFIG, key=key)


# ---------------- icons and small HTML pieces ----------------
def _data_uri(svg: str) -> str:
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon_css(color: str = ORANGE) -> str:
    """One CSS class per assets/icons/*.svg: .ml-icon-<name> (st.html strips inline <svg> and <img>, so icons
    are background images of the injected style block)."""
    rules = []
    for p in sorted(ICONS.glob("*.svg")):
        svg = p.read_text(encoding="utf-8").strip().replace("currentColor", color)
        rules.append(f'.ml-icon-{p.stem} {{ background-image: url("{_data_uri(svg)}"); }}')
    if MARK_SVG.exists():
        rules.append(f'.ml-brandmark {{ background-image: url("{_data_uri(MARK_SVG.read_text(encoding="utf-8"))}"); }}')
    return "\n".join(rules)


def brand_lockup(size: int = 64) -> str:
    """The logo as HTML: gauge mark + 'Motorlytics' wordmark in the heading font ('lytics' in the brand gradient)."""
    return (f'<div class="ml-lockup" style="gap:{size * 0.28:.0f}px"><span class="ml-brandmark" '
            f'style="width:{size}px;height:{size}px" aria-hidden="true"></span><span class="ml-wordmark" '
            f'style="font-size:{size * 0.72:.0f}px">Motor<span class="ml-grad-text">lytics</span></span></div>')


def icon(name: str, size: int = 22) -> str:
    """<span> showing assets/icons/<name>.svg (original line icon, brand orange); '' if missing."""
    if not (ICONS / f"{name}.svg").exists():
        return ""
    return f'<span class="ml-icon ml-icon-{name}" style="width:{size}px;height:{size}px" aria-hidden="true"></span>'


def image_svg(name: str) -> str:
    """SVG markup of assets/images/<name>.svg, for st.image."""
    p = IMAGES / f"{name}.svg"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def esc(text) -> str:
    return html.escape(str(text))


# ---------------- CSS ----------------
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {{
  --ml-black: {BLACK}; --ml-bg: {BG}; --ml-surface: {SURFACE}; --ml-raised: {RAISED}; --ml-border: {BORDER};
  --ml-text: {TEXT}; --ml-muted: {MUTED}; --ml-orange: {ORANGE}; --ml-red: {RED}; --ml-crimson: {CRIMSON};
  --ml-gradient: {GRADIENT}; --ml-success: {SUCCESS}; --ml-warning: {WARNING}; --ml-error: {ERROR};
  --ml-r-card: {R_CARD}px; --ml-r-input: {R_INPUT}px;
  --ml-shadow: 0 1px 2px rgba(0,0,0,.45), 0 8px 24px rgba(0,0,0,.28);
}}

/* ---------- base ---------- */
.stApp {{ background: var(--ml-bg); color: var(--ml-text); }}
html, body, .stApp, .stMarkdown, button, input, textarea, select {{ font-family: {FONT_BODY}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: {FONT_HEAD}; font-weight: 700; letter-spacing: -0.015em; }}
.stApp h1 {{ font-size: 2rem; }}
[data-testid="stMain"] {{ overflow-x: hidden; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1280px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--ml-muted); }}
a {{ color: var(--ml-orange); }}
hr {{ border-color: var(--ml-border) !important; }}
.ml-num, [data-testid="stMetricValue"], .ml-stat-value {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }}

/* ---------- page header ---------- */
.ml-page-title {{ font-family: {FONT_HEAD}; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--ml-text); margin: 0; line-height: 1.2; }}
.ml-page-sub {{ color: var(--ml-muted); font-size: .95rem; margin: 6px 0 0 0; }}
.ml-accent-line {{ height: 2px; width: 56px; border-radius: 2px; background: var(--ml-gradient); margin: 14px 0 8px 0; }}

/* ---------- cards: st.container(key="mlcard-...") ---------- */
[class*="st-key-mlcard"] {{
  background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: var(--ml-r-card);
  box-shadow: var(--ml-shadow); padding: 18px 20px; gap: 12px;
}}
[class*="st-key-mlcard-raised"] {{ background: var(--ml-raised); }}
[class*="st-key-mlcard-hero"] {{ background:
  radial-gradient(120% 140% at 100% 0%, {_hex_rgba(CRIMSON, .28)} 0%, rgba(0,0,0,0) 55%), var(--ml-surface); }}
[class*="st-key-mlcard-click"] {{ transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease; }}
[class*="st-key-mlcard-click"]:hover {{ transform: translateY(-2px); border-color: {_hex_rgba(ORANGE, .55)};
  box-shadow: 0 10px 28px rgba(0,0,0,.4), 0 0 0 1px {_hex_rgba(ORANGE, .15)}; }}
.ml-card-title {{ font-family: {FONT_HEAD}; font-weight: 700; font-size: 1.02rem; color: var(--ml-text); margin: 0; }}
.ml-card-sub {{ color: var(--ml-muted); font-size: .85rem; margin: 2px 0 0 0; }}
.ml-takeaway {{ color: var(--ml-muted); font-size: .85rem; margin: 0; }}
.ml-takeaway b {{ color: var(--ml-text); }}

/* formula strip (Quick estimate): Model x factor = result */
.st-key-mlformula {{ background: var(--ml-raised); border: 1px solid var(--ml-border); border-left: 3px solid var(--ml-orange);
  border-radius: var(--ml-r-card); padding: 12px 16px; }}
.st-key-mlformula p {{ font-variant-numeric: tabular-nums; margin: 0; }}

/* ---------- metric tiles ---------- */
[data-testid="stMetric"] {{
  background: var(--ml-raised); border: 1px solid var(--ml-border); border-radius: var(--ml-r-card);
  padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.35); height: 100%;
}}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {{ color: var(--ml-muted) !important; font-size: .8rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-family: {FONT_HEAD}; color: var(--ml-text); font-size: 1.75rem; font-weight: 700; }}
[class*="st-key-mlkpi-accent"] [data-testid="stMetricValue"] {{ color: var(--ml-orange); }}
[class*="st-key-mlkpi"] [data-testid="stMetric"] {{ position: relative; overflow: hidden; }}
[class*="st-key-mlkpi"] [data-testid="stMetric"]::before {{ content: ""; position: absolute; left: 0; right: 0; top: 0;
  height: 2px; background: var(--ml-gradient); opacity: .9; }}

/* ---------- buttons ---------- */
[data-testid^="stBaseButton-primary"] {{
  background: var(--ml-gradient) !important; border: 0 !important; color: #fff !important; font-weight: 600;
  border-radius: var(--ml-r-input); box-shadow: 0 6px 18px {_hex_rgba(RED, .28)}; transition: filter .15s, transform .15s;
}}
[data-testid^="stBaseButton-primary"]:hover {{ filter: brightness(1.08); transform: translateY(-1px); }}
[data-testid^="stBaseButton-primary"]:disabled {{ background: var(--ml-raised) !important; color: #71717A !important;
  border: 1px solid var(--ml-border) !important; box-shadow: none; transform: none; filter: none; }}
[data-testid^="stBaseButton-secondary"] {{
  background: transparent !important; border: 1px solid var(--ml-border) !important; color: var(--ml-text) !important;
  border-radius: var(--ml-r-input); font-weight: 500; transition: border-color .15s, color .15s, background .15s;
}}
[data-testid^="stBaseButton-secondary"]:hover {{ border-color: var(--ml-orange) !important; color: var(--ml-orange) !important;
  background: {_hex_rgba(ORANGE, .06)} !important; }}
[class*="st-key-mldanger"] [data-testid^="stBaseButton"] {{ background: transparent !important;
  border: 1px solid {_hex_rgba(RED, .6)} !important; color: {RED} !important; box-shadow: none; }}
[class*="st-key-mldanger"] [data-testid^="stBaseButton"]:hover {{ background: {_hex_rgba(RED, .12)} !important; }}
[data-testid="stPopover"] button {{ border-radius: var(--ml-r-input); }}

/* ---------- inputs and selects ---------- */
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div, [data-testid="stDateInput"] > div > div, .stTextArea textarea {{
  background: var(--ml-raised) !important; border-color: var(--ml-border) !important; border-radius: var(--ml-r-input) !important;
}}
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within {{
  border-color: var(--ml-orange) !important; box-shadow: 0 0 0 3px {_hex_rgba(ORANGE, .15)}; }}
[data-testid="stWidgetLabel"] p {{ color: var(--ml-text); font-weight: 500; font-size: .875rem; }}
[data-testid="stForm"] {{ border: 0; padding: 0; }}

/* ---------- tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {{ gap: 6px; border-bottom: 1px solid var(--ml-border); }}
[data-testid="stTab"] {{ color: var(--ml-muted); font-weight: 600; padding: 8px 14px; border-radius: 8px 8px 0 0; }}
[data-testid="stTab"]:hover {{ color: var(--ml-text); background: {_hex_rgba(ORANGE, .05)}; }}
[data-testid="stTab"][aria-selected="true"] {{ color: var(--ml-text); box-shadow: inset 0 -2px 0 var(--ml-orange); }}

/* ---------- expanders, dataframes, uploader, alerts ---------- */
[data-testid="stExpander"] details {{ background: var(--ml-surface); border: 1px solid var(--ml-border) !important;
  border-radius: var(--ml-r-card); }}
[data-testid="stExpander"] summary:hover {{ color: var(--ml-orange); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--ml-border); border-radius: var(--ml-r-card);
  overflow: hidden; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--ml-raised); border: 1.5px dashed #3A3A44;
  border-radius: var(--ml-r-card); padding: 28px 20px; transition: border-color .15s, background .15s; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--ml-orange); background: {_hex_rgba(ORANGE, .05)}; }}
[data-testid="stAlertContainer"] {{ border-radius: var(--ml-r-card); }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: var(--ml-black); border-right: 1px solid #17171B; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 10px; border-left: 3px solid transparent; margin: 1px 0;
  transition: background .15s; }}
[data-testid="stSidebarNavLink"]:hover {{ background: {_hex_rgba(ORANGE, .06)}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {_hex_rgba(ORANGE, .12)};
  border-left-color: var(--ml-orange); }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--ml-text) !important; font-weight: 600; }}
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{ color: var(--ml-orange) !important; }}
[data-testid="stNavSectionHeader"] {{ color: #71717A; text-transform: uppercase; letter-spacing: .08em; font-size: .7rem; }}
[data-testid="stSidebarLogo"], [data-testid="stSidebarHeader"] [data-testid="stLogo"] {{ height: 4.5rem !important; max-width: 100%; }}
[data-testid="stSidebarHeader"] {{ height: auto; padding-top: 1rem; }}
[data-testid="stTopNav"] [data-testid="stLogo"], [data-testid="stHeader"] [data-testid="stLogo"] {{ height: 2.6rem !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ box-shadow: inset 0 -2px 0 var(--ml-orange); }}

/* ---------- badges / pills ---------- */
[data-testid="stMarkdownBadge"], .ml-pill {{ border-radius: {R_PILL}px !important; padding: 2px 10px; font-weight: 600; }}
.ml-pill {{ display: inline-flex; align-items: center; gap: 6px; font-size: .78rem; border: 1px solid var(--ml-border);
  background: var(--ml-raised); color: var(--ml-text); }}
.ml-icon {{ display: inline-block; background-repeat: no-repeat; background-position: center; background-size: contain; }}
.ml-brandmark {{ display: inline-block; flex: none; background-size: contain; background-repeat: no-repeat;
  filter: drop-shadow(0 8px 24px {_hex_rgba(RED, .35)}); }}
.ml-lockup {{ display: flex; align-items: center; justify-content: center; flex-wrap: wrap; }}
.ml-wordmark {{ font-family: {FONT_HEAD}; font-weight: 800; letter-spacing: -0.03em; color: var(--ml-text); line-height: 1; }}
.ml-grad-text {{ background: var(--ml-gradient); -webkit-background-clip: text; background-clip: text; color: transparent; }}
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
