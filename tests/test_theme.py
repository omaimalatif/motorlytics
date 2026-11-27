"""Design system: config.toml agrees with ui/theme.py, the Plotly template and map colours are the brand ones,
the CSS covers the styled widgets, and every page injects it and renders on the dark theme."""
import hashlib
from pathlib import Path
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
c.ok("Plus Jakarta Sans:https://fonts.googleapis.com" in th["headingFont"] and "@import" not in theme.CSS,
     "fonts loaded once (config.toml), no duplicate @import")
c.ok(cfg["server"]["enableWebsocketCompression"] is True, "websocket compression on (faster on mobile data)")

t = theme.template().layout
c.ok(pio.templates.default == "motorlytics" and t.paper_bgcolor == "rgba(0,0,0,0)" and t.yaxis.gridcolor == theme.GRID
     and t.hoverlabel.bordercolor == theme.BORDER and t.barcornerradius == 6, "Plotly template registered and default")
fig = px.bar(x=[1, 2], y=[3, 4])
c.ok(fig.layout.template.layout.barcornerradius == 6, "px charts pick up the template")
def lum(h):
    r, g, b = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
data_colours = list(theme.STYLE_COLORS.values()) + theme.CATEGORICAL + [theme.MEASURED, theme.PREDICTED, theme.SINGLE]
c.ok(all(lum(x) < 0.45 for x in data_colours) and (theme.MEASURED, theme.PREDICTED) == (theme.INK, theme.CHART_BLUE),
     "chart colours are all dark (readable on white)", ", ".join(f"{x}:{lum(x):.2f}" for x in data_colours))

spec = json.loads(route_deck([[42.28, -83.74], [42.29, -83.73]]).to_json())
c.ok("positron" in spec["mapStyle"] and spec["layers"][0]["getColor"] == theme.MAP_ROUTE
     and spec["layers"][1]["data"][0]["color"] == theme.MAP_START, "map: CARTO light, sky-blue route, green start")

for sel in ['st-key-mlcard', 'stMetric', 'stBaseButton-primary', 'stBaseButton-secondary', 'stTextInputRootElement',
            'stTab', 'stExpander', 'stDataFrame', 'stFileUploaderDropzone', 'stSidebarNavLink', 'aria-current',
            'stMarkdownBadge', 'tabular-nums', 'max-width: 640px', 'font-size: 16px']:
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
    n_img = len([e for e in at.main if type(e).__name__ == "Image"])
    allowed = {"overview": 1, "my_car": 1}.get(page, 0)        # the car's own picture; About shows result figures
    if page != "about_model" and n_img > allowed:
        c.ok(False, f"{page}: only the car's own picture", f"{n_img} images")
    if page in ("overview", "my_car") and n_img != 1:
        c.ok(False, f"{page}: the Opel Corsa's picture is shown", f"{n_img} images")
c.ok(True, "every page renders; every chart uses the motorlytics template; car picture only on Overview / My car")
code = " ".join(p_.read_text(encoding="utf-8") for d in ["core", "ui", "app_pages"] for p_ in (ROOT / d).glob("*.py"))
code += (ROOT / "app.py").read_text(encoding="utf-8")
c.ok("wikipedia" not in code.lower() and "car_photo" not in code and "st.logo(image=theme.LOGO_LIGHT" not in code,
     "no internet car photos, no wordmark in the signed-in sidebar")
from PIL import Image  # noqa: E402

from core import cars  # noqa: E402

expect = {("Suzuki", "Alto"): "alto_2022", ("Opel", "Corsa"): "corsa_d_2010", ("BMW", "1 Series"): "bmw_1_2013",
          ("VED", "vehicle 530"): "ved_530", ("VED", "vehicle 157"): "ved_157", ("Suzuki", "Cultus"): None}
got = {k: (cars.car_image({"make": k[0], "model": k[1]}) or Path("none")).stem for k in expect}
c.ok(all(got[k] == (v or "none") for k, v in expect.items()), "car pictures matched by make / model", str(got))
pics = sorted(cars.CAR_IMAGES.glob("*.png"))
ok = []
for p_ in pics:
    im = Image.open(p_).convert("RGBA")
    ok.append(im.getpixel((0, 0))[3] == 0 and im.getpixel((im.width // 2, im.height // 2))[3] >= 250
              and p_.stat().st_size < 150_000)
c.ok(len(pics) == 5 and all(ok), "5 transparent, light car pictures (background removed, under 150 KB)")
c.ok("drop-shadow" in theme.CSS, "soft CSS drop shadow under the car pictures")
c.done()
