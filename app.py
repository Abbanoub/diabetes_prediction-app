import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    min-height: 100vh;
}

/* Main card */
.main-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 2.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* Header */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60efff, #0061ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}

.hero-sub {
    color: rgba(255,255,255,0.45);
    font-size: 0.95rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Section label */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: #60efff;
    text-transform: uppercase;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(255,75,75,0.05));
    border: 1px solid rgba(255,75,75,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}

.result-negative {
    background: linear-gradient(135deg, rgba(0,220,130,0.15), rgba(0,220,130,0.05));
    border: 1px solid rgba(0,220,130,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}

.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.result-sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
}

.prob-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    margin: 0.5rem 0;
}

/* Slider labels */
.stSlider label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.88rem !important;
}

/* Select box */
.stSelectbox label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.88rem !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #0061ff, #60efff);
    color: #0f1117;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
    letter-spacing: 0.03em;
}

.stButton > button:hover {
    opacity: 0.85;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 1.5rem 0 !important;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #60efff;
}

.metric-lbl {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    margin-top: 0.2rem;
}

/* Warning note */
.warning-note {
    background: rgba(255,200,0,0.07);
    border: 1px solid rgba(255,200,0,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: rgba(255,200,0,0.75);
    margin-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter patient data below to assess diabetes risk using an ML model trained on 100,000 records.</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found! Make sure `diabetes_model.pkl` and `scaler.pkl` are in the same folder as `app.py`.")
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", min_value=1, max_value=100, value=35)
    bmi = st.slider("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])

with col2:
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "not current", "ever", "unknown"])
    hba1c = st.slider("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
    blood_glucose = st.slider("Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=100)

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Diabetes Risk"):

    # Encode inputs
    gender_enc = 1 if gender == "Male" else 0
    hypertension_enc = 1 if hypertension == "Yes" else 0
    heart_disease_enc = 1 if heart_disease == "Yes" else 0

    smoking_map = {"current": 0, "ever": 1, "former": 2, "never": 3, "not current": 4, "unknown": 5}
    smoking_enc = smoking_map[smoking_history]

    input_data = np.array([[gender_enc, age, hypertension_enc, heart_disease_enc,
                            smoking_enc, bmi, hba1c, blood_glucose]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ── Result ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-title" style="color:#ff6b6b;">⚠️ High Diabetes Risk</div>
            <div class="prob-number" style="color:#ff6b6b;">{probability*100:.1f}%</div>
            <div class="result-sub">Probability of Diabetes</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-title" style="color:#00dc82;">✅ Low Diabetes Risk</div>
            <div class="prob-number" style="color:#00dc82;">{probability*100:.1f}%</div>
            <div class="result-sub">Probability of Diabetes</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key Factors ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Key Risk Factors Entered</div>', unsafe_allow_html=True)

    hba1c_risk = "🔴 High" if hba1c >= 6.5 else ("🟡 Pre-diabetic" if hba1c >= 5.7 else "🟢 Normal")
    glucose_risk = "🔴 High" if blood_glucose >= 200 else ("🟡 Elevated" if blood_glucose >= 140 else "🟢 Normal")
    bmi_risk = "🔴 Obese" if bmi >= 30 else ("🟡 Overweight" if bmi >= 25 else "🟢 Normal")

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">{hba1c}%</div>
            <div class="metric-lbl">HbA1c — {hba1c_risk}</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{blood_glucose}</div>
            <div class="metric-lbl">Blood Glucose — {glucose_risk}</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{bmi:.1f}</div>
            <div class="metric-lbl">BMI — {bmi_risk}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-note">
        ⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem;">Built with Streamlit · Logistic Regression · 100K Records · by Abanoub</p>',
    unsafe_allow_html=True
)
