"""Sign in / Create account (public): brand panel on black on the left, a card with the two forms on the right.
Authentication is unchanged (core/auth.py, PBKDF2). After signing in, app.py shows Overview."""
import streamlit as st

from core import auth
from ui import theme

E = theme.esc

if st.session_state.get("user_id"):           # already signed in (the page stays reachable, hidden, for the hand-over)
    st.info("You are signed in.", icon=":material/check_circle:")
    st.page_link("pages/overview.py", label="Go to Overview", icon=":material/dashboard:")
    st.stop()

st.html("""<style>
[data-testid="stHeader"] { background: #000000; }
[data-testid="stMainBlockContainer"] { padding-top: 5.5rem; max-width: 1180px; }
.st-key-mlbrand .ml-lockup { margin: 16px 0 20px 0; }
.st-key-mlbrand { background: #000000; border: 1px solid #17171B; border-radius: 16px; padding: 32px 28px;
  min-height: 100%; position: relative; overflow: hidden; }
.st-key-mlbrand::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
  background: var(--ml-gradient); opacity: .7; }
@media (max-width: 640px) {   /* phones: the form first, the brand panel under it */
  [data-testid="stHorizontalBlock"]:has(.st-key-mlbrand) { flex-direction: column-reverse; } }
.ml-benefits { display: flex; flex-direction: column; gap: 16px; margin: 4px 0 0 0; }
.ml-benefit { display: flex; gap: 14px; align-items: flex-start; }
.ml-benefit .ml-ic { flex: none; width: 40px; height: 40px; border-radius: 11px; display: inline-flex; align-items: center;
  justify-content: center; color: #F26A10; background: rgba(242,106,16,.10); border: 1px solid rgba(242,106,16,.25); }
.ml-benefit b { color: #F4F4F5; font-family: 'Plus Jakarta Sans', sans-serif; display: block; margin-bottom: 2px; }
.ml-benefit span.t { color: #A1A1AA; font-size: .9rem; line-height: 1.5; }
.ml-auth-title { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.6rem; margin: 0;
  letter-spacing: -0.02em; color: #F4F4F5; }
.ml-auth-sub { color: #A1A1AA; margin: 4px 0 0 0; font-size: .92rem; }
</style>""")

benefits = [("fuel", "Fuel use per trip", "Measured from MAF, or predicted by the ML model from how you drove."),
            ("gauge", "Your driving, scored", "An eco-score and plain tips that show where the litres go."),
            ("coins", "Cost in rupees", "Every trip priced with your petrol price, tuned to your own car.")]

left, right = st.columns([1, 1.05], gap="large")
with left:
    with st.container(key="mlbrand", horizontal_alignment="center", gap="medium"):
        st.html(theme.brand_lockup(60))
        st.html('<div class="ml-benefits">' + "".join(
            f'<div class="ml-benefit"><span class="ml-ic">{theme.icon(ic, 20)}</span><div><b>{E(t)}</b>'
            f'<span class="t">{E(d)}</span></div></div>' for ic, t, d in benefits) + "</div>")

with right:
    st.page_link("pages/welcome.py", label="Back to Welcome", icon=":material/arrow_back:")
    with st.container(key="mlcard-auth"):
        st.html('<p class="ml-auth-title">Welcome to Motorlytics</p>'
                '<p class="ml-auth-sub">Sign in, or create a free account in a few seconds.</p>')
        mode = st.session_state.get("auth_mode", "Sign in")
        tab_in, tab_up = st.tabs(["Sign in", "Create account"], default=mode)
        with tab_in:
            with st.form("login", border=False):
                email = st.text_input("Email", placeholder="you@example.com")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary", width="stretch"):
                    u = auth.login(email, pw)
                    if u:
                        st.session_state["user_id"] = u["id"]
                        st.session_state.pop("auth_mode", None)
                        st.session_state["goto"] = "overview"
                        st.rerun()                       # app.py now shows the signed-in pages, Overview first
                    st.error("Email or password is not correct.", icon=":material/error:")
            st.caption(":material/info: Just looking? Sign in with the demo account: **demo@motorlytics.test** / "
                       "**demo12345**")
        with tab_up:
            with st.form("register", border=False):
                name = st.text_input("Name")
                email = st.text_input("Email", key="reg_email", placeholder="you@example.com")
                city = st.text_input("City", help="Used for the current temperature (Open-Meteo).")
                pw = st.text_input("Password (8+ characters)", type="password", key="reg_pw")
                pw2 = st.text_input("Repeat password", type="password")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    if pw != pw2:
                        st.error("The passwords do not match.", icon=":material/error:")
                    else:
                        u, err = auth.register(name, email, pw, city)
                        if err:
                            st.error(err, icon=":material/error:")
                        else:
                            st.session_state["user_id"] = u["id"]
                            st.session_state.pop("auth_mode", None)
                            st.session_state["goto"] = "overview"
                            st.rerun()
