"""Public pages: signed out, only Welcome and Sign in exist; Welcome shows the real results; Sign in shows inline
errors, keeps the PBKDF2 login, and lands on Overview; logout returns to Welcome."""
from _helpers import ROOT, Checks, setup

setup("public")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, db  # noqa: E402

c = Checks("test_public")
db.init_db()
auth.register("Public Tester", "public@example.com", "secret123", "Lahore")

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.run()
c.page(at, "welcome")
html = " ".join(e.proto.body for e in at.get("html"))
c.ok("Know what every trip" in html and "really costs" in html, "hero headline")
for text in ["12,793", "196", "training trips", "error on unseen cars", "after 3-trip calibration"]:
    if text not in html:
        c.ok(False, f"stats strip shows {text}")
c.ok(all(t in html for t in ["12,793", "196 ", "8.7 %", "10.3 %"]), "stats strip: 12,793 trips, 196 vehicles, 8.7 %, 10.3 %")
c.ok(html.count('class="ml-step"') == 3 and html.count('class="ml-feature"') == 6, "3 steps and 6 feature cards")
c.ok(all(m in html for m in ["Decision Tree", "MLP", "k-NN", "LWR", "Ridge", "Random Forest", "LightGBM",
                             "About the model", "255 km/h"]), "ML section: course and reference models, cleaning")
c.ok(all(u in html for u in ["github.com/gsoh/VED", "envirocar.org", "open-meteo.com", "pakwheels.com",
                             "NUML Machine Learning Lab"]), "footer with data sources")
c.ok(not [e for e in at.sidebar if type(e).__name__ != "SpecialBlock"], "nothing in the sidebar when signed out")
src = (ROOT / "app.py").read_text(encoding="utf-8")
public = src[src.index("if not user:"):src.index("st.stop()")]
c.ok(public.count("st.Page(") == 2 and "welcome.py" in public and "sign_in.py" in public,
     "signed out: navigation has only Welcome and Sign in")
imgs = [e for e in at.main if type(e).__name__ == "Image"]
c.ok(len(imgs) == 0, "no car pictures on the Welcome page")
c.ok('class="ml-lockup"' in html and "ml-gauge-o" in html and "brandmark" not in html,
     "logo: the name only, with the gauge 'o', in the hero")

# Sign in button -> Sign in page with the two tabs; wrong password -> inline error
[b for b in at.button if b.label == "Sign in"][0].click().run()
c.page(at, "sign in")
c.ok([t.label for t in at.tabs] == ["Sign in", "Create account"], "tabs Sign in / Create account")
c.ok(any(p.proto.label == "Back to Welcome" for p in at.get("page_link")), "link back to Welcome")
fields = at.text_input
fields[0].set_value("public@example.com")
fields[1].set_value("wrong-password")
[b for b in at.button if b.label == "Sign in"][-1].click().run()
c.ok(any("not correct" in e.value for e in at.error), "wrong password: inline error")
c.ok("user_id" not in at.session_state, "not signed in")
# create account with mismatched passwords
at.text_input[2].set_value("X")
at.text_input(key="reg_email").set_value("x@example.com")
at.text_input(key="reg_pw").set_value("secret123")
at.text_input[6].set_value("different1")
[b for b in at.button if b.label == "Create account"][0].click().run()
c.ok(any("do not match" in e.value for e in at.error), "create account: passwords do not match")

at.text_input[0].set_value("public@example.com")
at.text_input[1].set_value("secret123")
[b for b in at.button if b.label == "Sign in"][-1].click().run()
c.page(at, "after sign in")
c.ok(at.session_state["user_id"] and db.get_user(at.session_state["user_id"])["password_hash"].startswith("pbkdf2_sha256$"),
     "signed in (PBKDF2 hash unchanged)")
c.ok(at.title and at.title[0].value.startswith("Good") and "goto" not in at.session_state, "lands on Overview",
     at.title[0].value if at.title else "")

at.switch_page("app_pages/account.py").run()
[b for b in at.button if b.label == "Sign out"][0].click().run()
c.page(at, "after logout")
c.ok("user_id" not in at.session_state and any(b.label == "Get started" for b in at.button), "logout returns to Welcome")
# fast first page: app.py and the public pages do not import pandas, Plotly or the models before sign-in
import subprocess  # noqa: E402
import sys  # noqa: E402

probe = subprocess.run([sys.executable, "-c", (
    "import sys; sys.path.insert(0, '.')\n"
    "from streamlit.testing.v1 import AppTest\n"
    "before = set(sys.modules)\n"                      # AppTest itself may load Plotly: count only what the app adds
    "at = AppTest.from_file('app.py', default_timeout=120); at.run()\n"
    "heavy = [m for m in ('pandas', 'plotly', 'sklearn', 'pydeck', 'core.app_core')\n"
    "         if m in sys.modules and m not in before]\n"
    "print('HEAVY', heavy, 'BUTTONS', [b.label for b in at.button])")],
    cwd=ROOT, capture_output=True, text=True, env={**__import__('os').environ})
line = next((x for x in probe.stdout.splitlines() if x.startswith("HEAVY")), probe.stderr[-300:])
c.ok("HEAVY []" in line and "Get started" in line, "signed-out start loads no pandas / Plotly / models", line)
c.done()
