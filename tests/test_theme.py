"""Design system: config.toml agrees with ui/theme.py, the Plotly template and map colours are the brand ones,
the CSS covers the styled widgets, and every page injects it and renders on the dark theme."""
import hashlib
import json
import tomllib

from _helpers import ROOT, SAMPLES, Checks, setup, upload

setup("theme")
import plotly.express as px  # noqa: E402
import plotly.io as pio  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, db  # noqa: E402
from ui import theme  # noqa: E402
from ui.components import route_deck  # noqa: E402

c = Checks("test_theme")
cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
th = cfg["theme"]
c.ok(th["base"] == "light" and th["primaryColor"] == theme.ORANGE and th["backgroundColor"] == theme.BG
     and th["secondaryBackgroundColor"] == theme.RAISED and th["textColor"] == theme.TEXT,
     "config.toml base colours match ui/theme.py")
c.ok(cfg["client"]["toolbarMode"] == "minimal", "Deploy button and main menu hidden")
c.ok(cfg["theme"]["sidebar"]["backgroundColor"] == theme.SIDEBAR, "sidebar colour matches ui/theme.py")
c.ok("gradient" not in theme.CSS and "box-shadow: 0 " not in theme.CSS and "translateY" not in theme.CSS,
     "flat look: no gradients, glows or hover jumps")
c.ok(not (ROOT / "pages").exists() and (ROOT / "app_pages").is_dir(),
     "page files in app_pages/ (no automatic page list flashes on start)")
c.ok(th["chartCategoricalColors"] == theme.CATEGORICAL, "categorical chart colours in the fixed order")

t = theme.template().layout
c.ok(pio.templates.default == "motorlytics" and t.paper_bgcolor == "rgba(0,0,0,0)" and t.yaxis.gridcolor == theme.GRID
     and t.hoverlabel.bordercolor == theme.BORDER and t.barcornerradius == 6, "Plotly template registered and default")
fig = px.bar(x=[1, 2], y=[3, 4])
c.ok(fig.layout.template.layout.barcornerradius == 6, "px charts pick up the template")
c.ok(theme.STYLE_COLORS == {"eco": "#0E9384", "normal": "#C98500", "aggressive": "#E5484D"}
     and (theme.MEASURED, theme.PREDICTED) == (theme.INK, theme.ACCENT), "style and measured / predicted colours")

spec = json.loads(route_deck([[42.28, -83.74], [42.29, -83.73]]).to_json())
c.ok("positron" in spec["mapStyle"] and spec["layers"][0]["getColor"] == theme.MAP_ROUTE
     and spec["layers"][1]["data"][0]["color"] == theme.MAP_START, "map: CARTO light, sky-blue route, green start")

for sel in ['st-key-mlcard', 'stMetric', 'stBaseButton-primary', 'stBaseButton-secondary', 'stTextInputRootElement',
            'stTab', 'stExpander', 'stDataFrame', 'stFileUploaderDropzone', 'stSidebarNavLink', 'aria-current',
            'stMarkdownBadge', 'Plus+Jakarta+Sans', 'tabular-nums']:
    if sel not in theme.CSS:
        c.ok(False, f"CSS styles {sel}")
c.ok(all(s in theme.CSS for s in ['st-key-mlcard', 'stSidebarNavLink', 'stFileUploaderDropzone']), "CSS covers the widgets")
c.ok(all((ROOT / p).exists() for p in [theme.LOGO, theme.ICON]) and theme.GAUGE_O_SVG.exists()
     and ".ml-gauge-o" in theme.icon_css() and 'M<span class="ml-gauge-o"' in theme.brand_lockup(),
     "logo: the wordmark with a gauge 'o' (no separate icon)")
c.ok(hashlib.sha256((ROOT / "assets/brand/motorlytics_logo.png").read_bytes()).hexdigest().startswith("89d72e8742b39654"),
     "the first logo file is kept unchanged")

# every page renders with the CSS block injected
db.init_db()
user, _ = auth.register("Theme Tester", "theme@example.com", "secret123", None)
db.update_user(user["id"], petrol_pkr=280.0)
user = db.get_user(user["id"])
car = db.add_car(user["id"], "Opel", "Corsa", 2011, 1.2, 4)
tid = upload(user, car, [SAMPLES / "corsa_d_2010_trip1.csv"])["trip_ids"][0]
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.session_state["trip_id"] = tid
at.run()
c.ok(any("st-key-mlcard" in e.proto.body for e in at.get("html")), "theme CSS injected (app.py runs theme.setup() "
     "before every page)")
src = (ROOT / "app.py").read_text(encoding="utf-8")
c.ok("page_icon=theme.ICON" in src and "st.logo(image=theme.LOGO" in src and theme.LOGO.endswith("wordmark.png"),
     "logo file used for the sidebar and the browser tab")
svgs = list((ROOT / "assets" / "icons").glob("*.svg")) + list((ROOT / "assets" / "images").glob("*.svg"))
c.ok(len(svgs) >= 10 and all(p.read_text(encoding="utf-8").lstrip().startswith("<svg") and "<image" not in
                              p.read_text(encoding="utf-8") for p in svgs), "icons and illustration are plain SVG drawings",
     f"{len(svgs)} files")
c.ok("ml-icon-upload" in theme.icon("upload") and theme.icon("missing") == ""
     and ".ml-icon-upload { background-image: url(\"data:image/svg+xml;base64," in theme.icon_css(),
     "icons: CSS classes with the SVG as data URI (st.html strips inline SVG and img)")
c.ok(src.index("theme.setup()") < src.index(".run()"), "setup() runs before the page")
for page in ["overview", "upload_trip", "quick_estimate", "trip_detail", "trip_history", "analytics", "my_car",
             "calibration", "about_model", "account"]:
    at.switch_page(f"app_pages/{page}.py").run()
    c.page(at, page)
    charts = [json.loads(e.proto.spec) for e in at.get("plotly_chart")]
    if any(e.proto.theme for e in at.get("plotly_chart")) or any(
            sp["layout"].get("template", {}).get("layout", {}).get("barcornerradius") != 6 for sp in charts):
        c.ok(False, f"{page}: charts use the motorlytics template without Streamlit's chart theme")
c.ok(True, "every page renders; every chart uses the motorlytics template")
c.done()
