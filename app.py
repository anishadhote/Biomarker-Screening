import hmac

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Biomarker screening", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Public+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #EDF1EF;
  --surface: #FBFCFB;
  --ink: #16282C;
  --muted: #5C6E70;
  --line: #D3DCDA;
  --edge: #8FA5A2;
  --accent: #0E5A60;
  --accent-soft: #D9E8E6;
  --flag: #A8651A;
  --flag-soft: #F1E3CF;
}

/* hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

html, body, .stApp, [class*="css"] {
  font-family: 'Public Sans', sans-serif;
  color: var(--ink);
}
.stApp { background: var(--bg); }
.block-container { max-width: 1120px; padding: 4rem 2rem 5rem; }

/* masthead */
.mast h1 {
  font-family: 'Newsreader', serif;
  font-weight: 400;
  font-size: 3.4rem;
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0 0 1rem;
  padding: 0;
  color: var(--ink);
}
.mast p {
  font-size: 1.05rem;
  line-height: 1.6;
  color: var(--muted);
  max-width: 58ch;
  margin: 0 0 3rem;
}

/* form sections */
.sec { margin: 0 0 0.9rem; padding-top: 1.6rem; border-top: 1px solid var(--line); }
.sec h3 {
  font-family: 'Newsreader', serif;
  font-weight: 500;
  font-size: 1.45rem;
  margin: 0 0 0.2rem;
  padding: 0;
  color: var(--ink);
}
.sec p { margin: 0; font-size: 0.9rem; color: var(--muted); }

[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
[data-testid="stNumberInput"] label p, [data-testid="stTextInput"] label p {
  font-size: 0.88rem; font-weight: 500; color: var(--ink);
}
[data-testid="stNumberInput"] div[data-baseweb="input"], [data-testid="stTextInput"] div[data-baseweb="input"] {
  background: var(--surface);
  border: 1.5px solid var(--edge);
  border-radius: 6px;
}
[data-testid="stNumberInput"] div[data-baseweb="input"]:hover, [data-testid="stTextInput"] div[data-baseweb="input"]:hover { border-color: var(--ink); }
[data-testid="stNumberInput"] div[data-baseweb="base-input"], [data-testid="stTextInput"] div[data-baseweb="base-input"] {
  background: transparent; border: none;
}
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within, [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { font-family: 'Public Sans', sans-serif; font-size: 1rem; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { display: none; }

[data-testid="stFormSubmitButton"] button {
  background: var(--accent);
  border: none;
  border-radius: 6px;
  padding: 0.7rem 1.8rem;
  margin-top: 1.2rem;
}
[data-testid="stFormSubmitButton"] button p {
  color: #fff; font-weight: 600; font-size: 0.98rem;
}
[data-testid="stFormSubmitButton"] button:hover { background: #0A474C; }
[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid var(--accent-soft); outline-offset: 2px; }

/* result panel */
.result {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 2rem 1.8rem;
}
.result.idle { background: transparent; border-style: dashed; }
.result h2 {
  font-family: 'Newsreader', serif;
  font-weight: 400;
  font-size: 2rem;
  line-height: 1.15;
  margin: 0 0 1.4rem;
  padding: 0;
}
.result.idle h2 { color: var(--muted); font-size: 1.5rem; margin-bottom: 0.6rem; }
.result p { font-size: 0.92rem; line-height: 1.55; color: var(--muted); margin: 0; }
.result p.note { margin-top: 1.4rem; padding-top: 1.2rem; border-top: 1px solid var(--line); }

.split { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--line); margin-bottom: 0.7rem; }
.split span { display: block; height: 100%; transform-origin: left; animation: grow 0.7s ease-out; }
.split .a { background: var(--accent); }
.split .b { background: var(--flag); }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .split span { animation: none; } }

.legend { display: flex; justify-content: space-between; font-size: 0.88rem; color: var(--ink); }
.legend b { font-weight: 600; }
.legend .l::before, .legend .r::before {
  content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px;
}
.legend .l::before { background: var(--accent); }
.legend .r::before { background: var(--flag); }

.warn {
  background: var(--flag-soft); border: 1px solid #DDBF93; border-radius: 6px;
  padding: 0.8rem 1rem; margin: 1.3rem 0 0; font-size: 0.88rem; line-height: 1.5; color: #5A3A0E;
}
.range { margin: -0.7rem 0 0.9rem; font-size: 0.78rem; line-height: 1.4; color: var(--muted); }
.range b { font-weight: 600; color: var(--ink); }
[data-testid="stNumberInput"] input::placeholder { color: #7D8F8D; opacity: 1; }
.calc { margin-top: 1.2rem !important; color: var(--ink) !important; }

/* login */
.login-head { margin: 5vh 0 2rem; }
.login-head h1 {
  font-family: 'Newsreader', serif; font-weight: 400; font-size: 2.6rem;
  line-height: 1.1; letter-spacing: -0.02em; margin: 0 0 0.6rem; padding: 0;
}
.login-head p { font-size: 0.98rem; line-height: 1.55; color: var(--muted); margin: 0; }

.st-key-signout button {
  background: transparent; border: 1px solid var(--line); border-radius: 6px;
}
.st-key-signout button p { color: var(--muted); font-size: 0.88rem; font-weight: 500; }
.st-key-signout button:hover { border-color: var(--ink); }

@media (max-width: 640px) {
  .mast h1 { font-size: 2.5rem; }
  .block-container { padding: 2.5rem 1.2rem 4rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



# ----------------------------------------------------------------------------
# Login
# ----------------------------------------------------------------------------
def expected_user() -> str:
    try:
        return str(st.secrets.get("APP_USER", "anisha26@"))
    except Exception:
        return "anisha26@"


def expected_password() -> str:
    # Set APP_PASSWORD and APP_USER in .streamlit/secrets.toml (or Streamlit Cloud > Settings > Secrets)
    # to override the defaults.
    try:
        return str(st.secrets.get("APP_PASSWORD", "1234@"))
    except Exception:
        return "1234@"


if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown(
            """
<div class="login-head">
<h1>Biomarker screening</h1>
<p>Sign in with your password. The login ID is optional.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        with st.form("login", border=False):
            uid = st.text_input("Login ID (optional)", value="anisha26@")
            pw = st.text_input("Password", type="password")
            go = st.form_submit_button("Sign in")
        if go:
            id_ok = uid.strip() == "" or hmac.compare_digest(uid.strip().encode(), expected_user().encode())
            pw_ok = hmac.compare_digest(pw.encode(), expected_password().encode())
            if id_ok and pw_ok:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("The login ID or password is not correct. Check both and try again.")
    st.stop()

# ----------------------------------------------------------------------------
# Model (trained from datar2.csv on first load, so it always matches the installed scikit-learn)
# ----------------------------------------------------------------------------
FEATURES = ["age", "bmi", "glucose", "insulin", "homa", "leptin", "adiponectin", "resistin", "mcp_1"]
LABELS = {
    "age": "age", "bmi": "BMI", "glucose": "glucose", "insulin": "insulin", "homa": "HOMA-IR",
    "leptin": "leptin", "adiponectin": "adiponectin", "resistin": "resistin", "mcp_1": "MCP-1",
}


@st.cache_resource
def train_model():
    df = pd.read_csv("datar2.csv")
    X, y = df[FEATURES], df["classification"]
    scaled = lambda est: Pipeline([("scale", StandardScaler()), ("model", est)])
    model = VotingClassifier(
        [
            ("logreg", scaled(LogisticRegression(max_iter=2000))),
            ("svm", scaled(SVC(C=3, probability=True, random_state=0))),
            ("knn", scaled(KNeighborsClassifier(n_neighbors=7, weights="distance"))),
            ("forest", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0)),
            ("boost", GradientBoostingClassifier(max_depth=2, n_estimators=100, learning_rate=0.05, random_state=0)),
        ],
        voting="soft",
    )
    model.fit(X, y)
    return model, X.min(), X.max()


try:
    model, lo, hi = train_model()
except Exception as e:
    st.error(f"The model could not be prepared. Check that datar2.csv is in the same folder as app.py. Details: {e}")
    st.stop()


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="mast">
<h1>Biomarker screening</h1>
<p>Enter nine routine body and blood measurements. The model compares them with 116 recorded cases and shows which group the profile sits closer to.</p>
</div>
""",
    unsafe_allow_html=True,
)


def section(title, hint):
    st.markdown(f'<div class="sec"><h3>{title}</h3><p>{hint}</p></div>', unsafe_allow_html=True)


left, gap, right = st.columns([5, 0.5, 4])


def field(col, key, label, allowed, normal, step, kind="float"):
    """Number box that starts empty, shows the normal range inside the box and the allowed range below it."""
    lo_, hi_ = allowed
    if kind == "int":
        value = col.number_input(label, min_value=int(lo_), max_value=int(hi_), value=None, step=1,
                                 placeholder=f"Normal {normal}" if normal else f"{lo_}-{hi_}", key=key)
    else:
        value = col.number_input(label, min_value=float(lo_), max_value=float(hi_), value=None, step=step,
                                 placeholder=f"Normal {normal}" if normal else f"{lo_}-{hi_}", key=key)
    col.markdown(f'<div class="range">Allowed <b>{lo_}–{hi_}</b></div>', unsafe_allow_html=True)
    return value


with left:
    with st.form("biomarkers", border=False):
        section("Patient", "Basic details")
        c1, c2 = st.columns(2)
        age = field(c1, "age", "Age (years)", (18, 100), "", 1, "int")
        bmi = field(c2, "bmi", "BMI (kg/m²)", (10, 60), "18.5–24.9", 0.1)

        section("Glucose metabolism", "From a fasting blood test. HOMA-IR is calculated from these two values.")
        c1, c2 = st.columns(2)
        glucose = field(c1, "glucose", "Glucose (mg/dL)", (30, 500), "70–99", 1, "int")
        insulin = field(c2, "insulin", "Insulin (µU/mL)", (1, 100), "2–25", 0.1)

        section("Blood markers", "Hormones and inflammation signals measured in serum")
        c1, c2 = st.columns(2)
        leptin = field(c1, "leptin", "Leptin (ng/mL)", (1, 120), "2–25", 0.1)
        adiponectin = field(c2, "adiponectin", "Adiponectin (µg/mL)", (1, 50), "3–30", 0.1)
        c3, c4 = st.columns(2)
        resistin = field(c3, "resistin", "Resistin (ng/mL)", (1, 100), "4–20", 0.1)
        mcp_1 = field(c4, "mcp_1", "MCP-1 (pg/mL)", (10, 2000), "100–500", 1.0)

        st.markdown(
            '<div class="range" style="margin-top:0.2rem">Normal and typical ranges are for adults and vary between laboratories. '
            "Where no normal range is given, the box shows the allowed range.</div>",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Run check")

entered = {"age": age, "BMI": bmi, "glucose": glucose, "insulin": insulin, "leptin": leptin,
           "adiponectin": adiponectin, "resistin": resistin, "MCP-1": mcp_1}
missing = [name for name, v in entered.items() if v is None]

with right:
    if not submitted:
        st.markdown(
            """
<div class="result idle">
<h2>No result yet</h2>
<p>Fill in the measurements and select Run check. The result will appear here.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    elif missing:
        st.markdown(
            f"""
<div class="result">
<h2>Some values are missing</h2>
<p>Enter a value for: {", ".join(missing)}.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        homa = glucose * insulin / 405  # standard HOMA-IR formula
        row = pd.DataFrame(
            [[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]], columns=FEATURES
        )
        proba = dict(zip(model.classes_, model.predict_proba(row)[0]))
        prediction = 2 if proba.get(2, 0.0) >= 0.5 else 1

        margin = 0.1 * (hi - lo)
        outside = [LABELS[f] for f in FEATURES if row[f].iloc[0] < lo[f] - margin[f] or row[f].iloc[0] > hi[f] + margin[f]]
        warn = (
            f'<div class="warn">These values are outside the range of the recorded cases: {", ".join(outside)}. '
            "The result is less reliable for this patient.</div>"
            if outside
            else ""
        )

        p_healthy = float(proba.get(1, 0.0)) * 100
        p_patient = float(proba.get(2, 0.0)) * 100

        headline = "Closer to the patient group" if prediction == 2 else "Closer to the healthy control group"

        st.markdown(
            f"""
<div class="result">
<h2>{headline}</h2>
<div class="split"><span class="a" style="width:{p_healthy:.1f}%"></span><span class="b" style="width:{p_patient:.1f}%"></span></div>
<div class="legend">
<span class="l">Healthy control <b>{p_healthy:.0f}%</b></span>
<span class="r">Patient <b>{p_patient:.0f}%</b></span>
</div>
<p class="calc">Calculated HOMA-IR: <b>{homa:.2f}</b></p>
{warn}
<p class="note">The model learns from 116 recorded cases and is right about three times in four on cases it has not seen. It is a screening aid, not a diagnosis. A clinician should review any result.</p>
</div>
""",
            unsafe_allow_html=True,
        )


st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
if st.button("Sign out", key="signout"):
    st.session_state.authed = False
    st.rerun()
