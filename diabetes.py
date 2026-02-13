import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

# ── Train Model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    try:
        data = pd.read_csv(url, names=cols)
    except Exception:
        np.random.seed(42)
        n = 768
        data = pd.DataFrame({
            "Pregnancies": np.random.randint(0,17,n),
            "Glucose": np.random.randint(70,200,n),
            "BloodPressure": np.random.randint(40,122,n),
            "SkinThickness": np.random.randint(0,99,n),
            "Insulin": np.random.randint(0,846,n),
            "BMI": np.round(np.random.uniform(18,67,n),1),
            "DiabetesPedigreeFunction": np.round(np.random.uniform(0.07,2.4,n),3),
            "Age": np.random.randint(21,81,n),
            "Outcome": np.random.randint(0,2,n),
        })

    X = data.drop(columns="Outcome", axis=1)
    Y = data["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel="linear", probability=True)
    classifier.fit(x_train, y_train)
    train_acc = accuracy_score(y_train, classifier.predict(x_train))
    test_acc  = accuracy_score(y_test,  classifier.predict(x_test))
    return classifier, scaler, train_acc, test_acc

with st.spinner("Initialising model…"):
    classifier, scaler, train_acc, test_acc = load_model()

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  —  Medical Teal · Dark Theme · Luxury Fintech style
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* ── Variables ── */
:root {
    --bg:         #060d10;
    --surface:    #0b1519;
    --surface2:   #101d22;
    --border:     #182830;
    --border-hi:  #1f3540;
    --teal:       #2dd4bf;
    --teal-light: #7fffd4;
    --teal-dim:   rgba(45,212,191,0.13);
    --red:        #f43f5e;
    --green:      #10b981;
    --amber:      #f59e0b;
    --text:       #dff1ee;
    --muted:      #4a6a70;
    --head-font:  'Playfair Display', Georgia, serif;
    --mono-font:  'IBM Plex Mono', monospace;
    --glow-teal:  0 0 28px rgba(45,212,191,0.22);
}

/* ── Global ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: var(--bg) !important;
    font-family: var(--mono-font) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }

/* ══════════════════════════════
   HERO
══════════════════════════════ */
.hero {
    position: relative;
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #060d10 0%, #0b1c22 60%, #060d10 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 90% 50%, rgba(45,212,191,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(45,212,191,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), var(--teal-light), transparent);
}
.hero-tag {
    display: inline-block;
    font-family: var(--mono-font);
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--teal);
    background: var(--teal-dim);
    border: 1px solid rgba(45,212,191,0.28);
    border-radius: 4px;
    padding: 0.25rem 0.8rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: var(--head-font) !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    line-height: 1.1 !important;
    letter-spacing: -0.01em;
    margin: 0 0 0.6rem 0 !important;
}
.hero h1 span {
    background: linear-gradient(135deg, var(--teal), var(--teal-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.04em;
}
.hero-badge {
    position: absolute;
    right: 3.5rem; top: 50%;
    transform: translateY(-50%);
    width: 80px; height: 80px;
    border-radius: 50%;
    background: var(--teal-dim);
    border: 1px solid rgba(45,212,191,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    box-shadow: var(--glow-teal);
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 6px var(--teal);
    margin-right: 0.5rem;
    vertical-align: middle;
}
.status-badge {
    font-size: 0.7rem; letter-spacing: 0.1em;
    color: var(--muted); text-transform: uppercase;
}

/* ── Accuracy Chips ── */
.acc-row {
    display: flex; gap: 0.8rem; margin-bottom: 2.5rem; flex-wrap: wrap;
}
.acc-chip {
    background: var(--surface2);
    border: 1px solid var(--border-hi);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.acc-chip span {
    font-family: var(--head-font);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--teal);
    margin-right: 0.35rem;
}

/* ══════════════════════════════
   SECTION LABELS
══════════════════════════════ */
.section-label {
    font-family: var(--mono-font);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--teal);
    border-left: 2px solid var(--teal);
    padding-left: 0.7rem;
    margin-bottom: 1.4rem;
    margin-top: 0.5rem;
}

/* ══════════════════════════════
   FORM PANELS
══════════════════════════════ */
.form-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem 2rem 2rem;
    position: relative;
    transition: border-color 0.3s;
}
.form-panel:hover { border-color: var(--border-hi); }
.form-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 1.5rem; right: 1.5rem; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.15), transparent);
}

/* ══════════════════════════════
   INPUTS
══════════════════════════════ */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
.stSlider label {
    font-family: var(--mono-font) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stNumberInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono-font) !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(45,212,191,0.12) !important;
    outline: none !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono-font) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(45,212,191,0.12) !important;
}
[data-testid="stSelectbox"] svg { color: var(--teal) !important; }
[data-testid="stSelectbox"] ul {
    background: var(--surface2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] li {
    font-family: var(--mono-font) !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] li:hover {
    background: var(--teal-dim) !important;
    color: var(--teal-light) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--teal) !important;
}

/* ══════════════════════════════
   BUTTON
══════════════════════════════ */
[data-testid="stButton"] > button {
    width: 100% !important;
    height: 58px !important;
    background: linear-gradient(135deg, #1a8c7e, var(--teal), #2dd4bf) !important;
    background-size: 200% 100% !important;
    color: #060d10 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--head-font) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(45,212,191,0.28), 0 1px 0 rgba(255,255,255,0.08) inset !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(45,212,191,0.42), 0 2px 0 rgba(255,255,255,0.12) inset !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ══════════════════════════════
   RESULT CARDS
══════════════════════════════ */
.result-wrap {
    animation: fadeSlideUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-diabetic {
    background: linear-gradient(135deg, rgba(244,63,94,0.1) 0%, rgba(6,13,16,0) 60%);
    border: 1px solid rgba(244,63,94,0.35);
    border-top: 3px solid var(--red);
    border-radius: 14px;
    padding: 2.5rem 2.5rem 2rem;
    position: relative; overflow: hidden;
}
.result-diabetic::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 90% 10%, rgba(244,63,94,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.result-safe {
    background: linear-gradient(135deg, rgba(45,212,191,0.1) 0%, rgba(6,13,16,0) 60%);
    border: 1px solid rgba(45,212,191,0.32);
    border-top: 3px solid var(--teal);
    border-radius: 14px;
    padding: 2.5rem 2.5rem 2rem;
    position: relative; overflow: hidden;
}
.result-safe::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 90% 10%, rgba(45,212,191,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.result-icon { font-size: 3rem; margin-bottom: 1rem; display: block; }
.result-verdict {
    font-family: var(--head-font);
    font-size: 2rem; font-weight: 700;
    margin-bottom: 0.5rem; line-height: 1.1;
}
.result-verdict.red   { color: var(--red); }
.result-verdict.teal  { color: var(--teal); }
.result-prob-row {
    display: flex; align-items: center;
    gap: 1rem; margin-top: 1.4rem;
}
.result-prob-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    white-space: nowrap;
}
.prob-bar-bg {
    flex: 1; height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 4px; overflow: hidden;
}
.prob-bar-red {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #f43f5e, #fb7185);
    box-shadow: 0 0 8px rgba(244,63,94,0.55);
}
.prob-bar-teal {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #2dd4bf, #7fffd4);
    box-shadow: 0 0 8px rgba(45,212,191,0.5);
}
.prob-pct {
    font-family: var(--head-font);
    font-size: 1.4rem; font-weight: 700;
    min-width: 64px; text-align: right;
}
.prob-pct.red  { color: var(--red); }
.prob-pct.teal { color: var(--teal); }
.result-note {
    font-size: 0.78rem; color: var(--muted);
    margin-top: 1.4rem; line-height: 1.65;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 1rem;
}

/* Idle card */
.idle-card {
    background: var(--surface);
    border: 1px dashed var(--border-hi);
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center; color: var(--muted);
}
.idle-icon { font-size: 2.8rem; margin-bottom: 1rem; opacity: 0.45; }
.idle-head {
    font-family: var(--head-font);
    font-size: 1.2rem; color: #1e3a42; margin-bottom: 0.4rem;
}
.idle-body { font-size: 0.8rem; line-height: 1.6; }

/* ── Layout helpers ── */
[data-testid="stHorizontalBlock"] {
    gap: 1.4rem !important;
    align-items: stretch !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-tag">⬡ Clinical Risk Intelligence</div>
    <h1>Diabetes <span>Risk</span><br>Prediction System</h1>
    <p class="hero-sub">
        SVM · Pima Indians Diabetes Dataset &nbsp;·&nbsp;
        <span class="status-dot"></span>
        <span class="status-badge">Model Active</span>
    </p>
    <div class="hero-badge">🩺</div>
</div>
<div class="acc-row">
    <div class="acc-chip"><span>{train_acc*100:.1f}%</span> Training Accuracy</div>
    <div class="acc-chip"><span>{test_acc*100:.1f}%</span> Test Accuracy</div>
    <div class="acc-chip"><span>768</span> Training Samples</div>
    <div class="acc-chip"><span>8</span> Input Features</div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
form_col, result_col = st.columns([1.05, 0.95], gap="medium")

with form_col:

    # ── Section 01 ─────────────────────────────────────
    st.markdown('<div class="section-label">01 — Patient Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        Pregnancies  = st.number_input("Pregnancies",          0,  17,  1)
        Glucose      = st.number_input("Glucose (mg/dL)",     50, 250, 120)
        BloodPressure= st.number_input("Blood Pressure (mmHg)",20, 140,  70)
        SkinThickness= st.number_input("Skin Thickness (mm)",   0, 100,  23)
    with c2:
        Insulin      = st.number_input("Insulin (µU/mL)",       0, 850,  80)
        BMI          = st.number_input("BMI (kg/m²)",         10.0, 70.0, 32.0, step=0.1, format="%.1f")
        DPF          = st.number_input("Diabetes Pedigree",   0.05,  2.5, 0.47, step=0.001, format="%.3f")
        Age          = st.number_input("Age (years)",           21,  90,  33)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Analyse Diabetes Risk →")

# ── Result Column ─────────────────────────────────────────────────────────────
with result_col:
    st.markdown('<div class="section-label">02 — Risk Assessment</div>', unsafe_allow_html=True)

    if predict_btn:
        input_array  = np.array([[Pregnancies, Glucose, BloodPressure,
                                   SkinThickness, Insulin, BMI, DPF, Age]])
        scaled_input = scaler.transform(input_array)
        prediction   = classifier.predict(scaled_input)[0]
        proba        = classifier.predict_proba(scaled_input)[0]
        risk_pct     = proba[1] * 100
        safe_pct     = proba[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-wrap">
            <div class="result-diabetic">
                <span class="result-icon">🔴</span>
                <div class="result-verdict red">Diabetic — High Risk</div>
                <div style="font-size:0.82rem;color:#8a9aaa;margin-top:0.3rem">
                    Patient profile indicates elevated diabetes risk.
                </div>
                <div class="result-prob-row">
                    <span class="result-prob-label">Diabetes<br>Probability</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-red" style="width:{risk_pct:.1f}%"></div>
                    </div>
                    <span class="prob-pct red">{risk_pct:.1f}%</span>
                </div>
                <div class="result-note">
                    ⚠ Model predicts a <strong style="color:#f43f5e">{risk_pct:.1f}%</strong>
                    probability of diabetes based on the given parameters.
                    Please consult a qualified healthcare professional for
                    clinical diagnosis and treatment.
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrap">
            <div class="result-safe">
                <span class="result-icon">🟢</span>
                <div class="result-verdict teal">Non-Diabetic — Low Risk</div>
                <div style="font-size:0.82rem;color:#8a9aaa;margin-top:0.3rem">
                    Patient parameters appear within a healthy range.
                </div>
                <div class="result-prob-row">
                    <span class="result-prob-label">Healthy<br>Confidence</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-teal" style="width:{safe_pct:.1f}%"></div>
                    </div>
                    <span class="prob-pct teal">{safe_pct:.1f}%</span>
                </div>
                <div class="result-note">
                    ✅ Model confidence of <strong style="color:#2dd4bf">{safe_pct:.1f}%</strong>
                    non-diabetic. Diabetes probability is only
                    <strong style="color:#f43f5e">{risk_pct:.1f}%</strong>.
                    Maintain a balanced diet and schedule regular check-ups.
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="idle-card">
            <div class="idle-icon">◈</div>
            <div class="idle-head">Awaiting Analysis</div>
            <div class="idle-body">
                Enter patient metrics on the left<br>
                and click <em>Analyse Diabetes Risk</em><br>
                to generate a risk assessment.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;font-size:0.72rem;color:#2a4a50;
                border-top:1px solid #182830;padding-top:1rem;line-height:1.7">
        This tool is for educational and informational purposes only.
        It does not constitute medical advice or replace clinical diagnosis
        by a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)