
# ================================================================
#  ❤️ Heart Disease Risk Predictor — Streamlit Application
#  Week 5: End-to-End ML Deployment
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
#  Page Configuration
# ----------------------------------------------------------------
st.set_page_config(
    page_title="❤️ Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.who.int/health-topics/cardiovascular-diseases",
        "About": """## Week 5 ML Deployment Project
Built with Scikit-Learn & Streamlit"""
    }
)

# ----------------------------------------------------------------
#  Custom CSS Styling
# ----------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #58a6ff15 0%, #f7816615 50%, #3fb95015 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #f78166, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub { color: #8b949e; font-size: 1.1rem; margin-top: 8px; }

    /* Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #58a6ff; }

    /* Result cards */
    .result-danger {
        background: linear-gradient(135deg, #f7816620, #161b22);
        border: 2px solid #f78166;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #3fb95020, #161b22);
        border: 2px solid #3fb950;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        color: #c9d1d9;
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 20px 0 12px 0;
        font-weight: 700;
        font-size: 1.2rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 14px 32px;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2ea043, #3fb950);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(63, 185, 80, 0.3);
    }

    /* Inputs */
    .stSlider > div { color: #c9d1d9; }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #8b949e !important;
        font-weight: 500;
    }

    /* Warning/info boxes */
    .disclaimer {
        background: #21262d;
        border: 1px solid #f78166;
        border-radius: 8px;
        padding: 12px 16px;
        color: #f78166;
        font-size: 0.85rem;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------
#  Load Model & Preprocessor
# ----------------------------------------------------------------

# -----------------------Replace_Code-----------------------------

from pathlib import Path
import joblib
import json
import streamlit as st

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_ml_artifacts():

    model_path = BASE_DIR / "ml_model" / "heart_disease_model.joblib"
    preprocessor_path = BASE_DIR / "ml_model" / "preprocessor.joblib"

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    if not preprocessor_path.exists():
        st.error(f"Preprocessor not found: {preprocessor_path}")
        st.stop()

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(BASE_DIR / "ml_model" / "metadata.json") as f:
        metadata = json.load(f)

    with open(BASE_DIR / "ml_model" / "feature_columns.json") as f:
        feature_cols = json.load(f)

    return model, preprocessor, metadata, feature_cols


# -----------------------Replace_Code-End-----------------------------

model, preprocessor, metadata, feature_cols = load_ml_artifacts()


# ----------------------------------------------------------------
#  Feature Engineering (mirrors training)
# ----------------------------------------------------------------
def engineer_features(data: dict) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    age      = data['age']
    thalach  = data['thalach']
    chol     = data['chol']
    trestbps = data['trestbps']
    ca       = data['ca']
    exang    = data['exang']
    cp       = data['cp']

    # Engineered features
    data['age_group']          = int(pd.cut([age], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])[0])
    data['heart_rate_reserve'] = thalach / (220 - age) if (220 - age) != 0 else 0
    data['chol_per_age']       = chol / age if age != 0 else 0
    data['bp_category']        = int(trestbps > 130)
    data['risk_composite']     = int(ca) + int(exang) + (1 if cp == 3 else 0)

    return pd.DataFrame([data])[feature_cols]


# ----------------------------------------------------------------
#  Prediction Function
# ----------------------------------------------------------------
def predict_risk(input_data: dict):
    """Run prediction pipeline and return results."""
    # Validate inputs
    if not (20 <= input_data['age'] <= 100):
        raise ValueError("Age must be between 20 and 100 years.")
    if not (50 <= input_data['trestbps'] <= 220):
        raise ValueError("Resting blood pressure must be between 50–220 mmHg.")
    if not (100 <= input_data['chol'] <= 600):
        raise ValueError("Cholesterol must be between 100–600 mg/dl.")
    if not (60 <= input_data['thalach'] <= 250):
        raise ValueError("Max heart rate must be between 60–250 bpm.")
    if not (0 <= input_data['oldpeak'] <= 10):
        raise ValueError("ST Depression (oldpeak) must be between 0–10.")

    # Engineer features & preprocess
    df_input    = engineer_features(input_data.copy())
    X_processed = preprocessor.transform(df_input)

    # Predict
    prediction = model.predict(X_processed)[0]
    proba      = model.predict_proba(X_processed)[0]
    risk_pct   = proba[1] * 100

    return {
        "prediction": int(prediction),
        "risk_percent": float(risk_pct),
        "confidence_no_disease": float(proba[0] * 100),
        "confidence_disease": float(proba[1] * 100),
        "risk_level": (
            "🔴 HIGH" if risk_pct >= 60 else
            "🟡 MODERATE" if risk_pct >= 35 else
            "🟢 LOW"
        )
    }


# ================================================================
#  HERO SECTION
# ================================================================
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">❤️ Heart Disease Risk Predictor</p>
    <p class="hero-sub">AI-powered cardiovascular risk assessment · Powered by Machine Learning</p>
    <p style="color:#58a6ff; margin-top:12px; font-size:0.9rem;">⚕️ For educational purposes only — Not a medical diagnosis tool</p>
</div>
""", unsafe_allow_html=True)


# ================================================================
#  SIDEBAR — Model Info
# ================================================================
with st.sidebar:
    st.markdown("## 🤖 Model Information")
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#8b949e; font-size:0.8rem;">MODEL</div>
        <div style="color:#58a6ff; font-size:1rem; font-weight:700; margin-top:4px;">{metadata['model_name']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Test Accuracy", f"{metadata['test_accuracy']*100:.1f}%")
        st.metric("F1 Score",      f"{metadata['test_f1']*100:.1f}%")
    with col_s2:
        st.metric("ROC-AUC", f"{metadata['test_auc']:.3f}")
        st.metric("Features", str(metadata['feature_count']))

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.info(f"""**UCI Heart Disease**

"""
            f"""Training samples: {metadata['training_samples']}

"""
            f"""Saved: {metadata['saved_at'][:10]} """)

    st.markdown("---")
    st.markdown("### 🔑 Risk Levels")
    st.markdown("🟢 **LOW** — Risk < 35%")
    st.markdown("🟡 **MODERATE** — Risk 35–60%")
    st.markdown("🔴 **HIGH** — Risk > 60%")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("Built with **Scikit-Learn**, **XGBoost** & **Streamlit** as part of a Data Science Internship — Week 5 Deployment Project.")


# ================================================================
#  MAIN FORM — Patient Input
# ================================================================
st.markdown('<p class="section-header">🧑‍⚕️ Patient Information</p>', unsafe_allow_html=True)

with st.form(key="prediction_form", clear_on_submit=False):

    # ---- Row 1: Demographics ----
    st.markdown("**👤 Demographics**")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("🔢 Age (years)", min_value=20, max_value=100, value=54,
                        help="Patient age in years")
    with col2:
        sex = st.selectbox("⚧ Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female",
                           help="Biological sex")
    with col3:
        cp = st.selectbox("💛 Chest Pain Type", options=[0, 1, 2, 3],
                          format_func=lambda x: {
                              0: "Typical Angina", 1: "Atypical Angina",
                              2: "Non-anginal Pain", 3: "Asymptomatic"
                          }[x],
                          help="Type of chest pain experienced")

    st.markdown("---")

    # ---- Row 2: Vitals ----
    st.markdown("**🫀 Vital Signs**")
    col4, col5, col6 = st.columns(3)
    with col4:
        trestbps = st.number_input("🩺 Resting Blood Pressure (mmHg)",
                                    min_value=50, max_value=220, value=130,
                                    help="Resting blood pressure in mmHg")
    with col5:
        chol = st.number_input("🧬 Cholesterol (mg/dl)",
                                min_value=100, max_value=600, value=245,
                                help="Serum cholesterol level")
    with col6:
        thalach = st.number_input("💓 Max Heart Rate (bpm)",
                                   min_value=60, max_value=250, value=150,
                                   help="Maximum heart rate achieved during exercise")

    st.markdown("---")

    # ---- Row 3: Clinical Features ----
    st.markdown("**🔬 Clinical Tests**")
    col7, col8, col9 = st.columns(3)
    with col7:
        fbs = st.selectbox("🍬 Fasting Blood Sugar", options=[0, 1],
                            format_func=lambda x: "≤ 120 mg/dl (Normal)" if x == 0 else "> 120 mg/dl (High)",
                            help="Fasting blood sugar level")
    with col8:
        restecg = st.selectbox("📈 Resting ECG Result", options=[0, 1, 2],
                                format_func=lambda x: {
                                    0: "Normal", 1: "ST-T Wave Abnormality", 2: "LV Hypertrophy"
                                }[x], help="Resting electrocardiographic result")
    with col9:
        exang = st.selectbox("🏃 Exercise-Induced Angina", options=[0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes",
                              help="Chest pain during exercise?")

    st.markdown("")
    col10, col11, col12, col13 = st.columns(4)
    with col10:
        oldpeak = st.number_input("📉 ST Depression (oldpeak)",
                                   min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                   help="ST depression induced by exercise relative to rest")
    with col11:
        slope = st.selectbox("📐 ST Slope", options=[0, 1, 2],
                              format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
                              help="Slope of peak exercise ST segment")
    with col12:
        ca = st.selectbox("🔭 Major Vessels (Fluoroscopy)", options=[0, 1, 2, 3],
                           help="Number of major vessels colored by fluoroscopy (0-3)")
    with col13:
        thal = st.selectbox("🧪 Thalassemia", options=[0, 1, 2],
                             format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x],
                             help="Type of thalassemia")

    st.markdown("")
    predict_btn = st.form_submit_button("🔬 Predict Heart Disease Risk", use_container_width=True)


# ================================================================
#  PREDICTION OUTPUT
# ================================================================
if predict_btn:
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': float(oldpeak), 'slope': slope,
        'ca': ca, 'thal': thal
    }

    with st.spinner("🔄 Analyzing patient data..."):
        try:
            result = predict_risk(input_data)
        except ValueError as e:
            st.error(f"❌ Input Validation Error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            st.stop()

    st.markdown("---")
    st.markdown('<p class="section-header">📊 Prediction Results</p>', unsafe_allow_html=True)

    # ---- Main Result ----
    is_disease = result["prediction"] == 1
    card_class  = "result-danger" if is_disease else "result-safe"
    icon        = "🚨" if is_disease else "✅"
    verdict     = "Heart Disease Detected" if is_disease else "No Heart Disease Detected"
    color       = "#f78166" if is_disease else "#3fb950"

    st.markdown(f"""
    <div class="{card_class}">
        <div style="font-size:3rem;">{icon}</div>
        <div style="font-size:2rem; font-weight:800; color:{color}; margin:12px 0;">{verdict}</div>
        <div style="font-size:1.4rem; color:#c9d1d9;">Risk Level: <strong>{result['risk_level']}</strong></div>
        <div style="font-size:3rem; font-weight:800; color:{color}; margin:12px 0;">{result['risk_percent']:.1f}%</div>
        <div style="color:#8b949e;">Probability of Heart Disease</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ---- Metrics Row ----
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("🔴 Disease Risk",   f"{result['confidence_disease']:.1f}%")
    with m2: st.metric("🟢 No-Disease",     f"{result['confidence_no_disease']:.1f}%")
    with m3: st.metric("🎯 Risk Level",     result['risk_level'])
    with m4: st.metric("⏰ Assessed",       datetime.now().strftime("%H:%M"))

    # ---- Gauge Chart ----
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['risk_percent'],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Disease Risk Score (%)", "font": {"size": 18, "color": "#c9d1d9"}},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        delta={"reference": 50, "increasing": {"color": "#f78166"}, "decreasing": {"color": "#3fb950"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#c9d1d9", "tickfont": {"color": "#c9d1d9"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 35],  "color": "#3fb95020"},
                {"range": [35, 60], "color": "#ffa65720"},
                {"range": [60, 100],"color": "#f7816620"}
            ],
            "threshold": {
                "line": {"color": "#ffa657", "width": 3},
                "thickness": 0.75, "value": 50
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#161b22", font_color="#c9d1d9",
        height=300, margin=dict(t=50, b=10, l=20, r=20)
    )

    # ---- Probability Bar Chart ----
    fig_bar = go.Figure([
        go.Bar(
            x=["No Heart Disease", "Heart Disease"],
            y=[result['confidence_no_disease'], result['confidence_disease']],
            marker_color=["#3fb950", "#f78166"],
            text=[f"{result['confidence_no_disease']:.1f}%", f"{result['confidence_disease']:.1f}%"],
            textposition="auto",
            textfont={"size": 16, "color": "white", "family": "Arial Black"},
            width=0.5
        )
    ])
    fig_bar.update_layout(
        title="Prediction Confidence",
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        yaxis_title="Probability (%)", yaxis_range=[0, 100],
        height=280, margin=dict(t=40, b=10, l=20, r=20)
    )

    chart_col1, chart_col2 = st.columns([1.2, 1])
    with chart_col1: st.plotly_chart(fig_gauge, use_container_width=True)
    with chart_col2: st.plotly_chart(fig_bar,   use_container_width=True)

    # ---- Clinical Recommendations ----
    st.markdown('<p class="section-header">💡 Clinical Recommendations</p>', unsafe_allow_html=True)

    risk_pct = result['risk_percent']
    if risk_pct >= 60:
        st.error("""**🚨 HIGH RISK — Immediate Action Recommended:**

"""
                 """- 🏥 Consult a cardiologist immediately
"""
                 """- 🩺 Schedule comprehensive cardiac evaluation
"""
                 """- 💊 Review current medications with your physician
"""
                 """- 🚫 Avoid strenuous activities until cleared by a doctor
"""
                 """- 📋 Get an ECG, Echo, and stress test done""")
    elif risk_pct >= 35:
        st.warning("""**🟡 MODERATE RISK — Preventive Measures:**

"""
                   """- 👨‍⚕️ Schedule a routine cardiac checkup
"""
                   """- 🏃 Start a supervised exercise program
"""
                   """- 🥗 Adopt a heart-healthy Mediterranean diet
"""
                   """- 🧂 Reduce sodium and saturated fat intake
"""
                   """- 📊 Monitor blood pressure and cholesterol regularly""")
    else:
        st.success("""**🟢 LOW RISK — Maintain Healthy Lifestyle:**

"""
                   """- ✅ Continue regular exercise (150+ min/week)
"""
                   """- 🥦 Maintain a balanced, nutritious diet
"""
                   """- 🚭 Avoid smoking and limit alcohol consumption
"""
                   """- 😴 Ensure 7–9 hours of quality sleep
"""
                   """- 📅 Annual health checkups recommended""")

    # ---- Disclaimer ----
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and informational purposes only.
        It does <strong>NOT</strong> constitute medical advice or replace professional medical consultation.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)

    # ---- Input Summary ----
    with st.expander("📋 View Input Summary"):
        summary_df = pd.DataFrame({
            "Parameter": list(input_data.keys()),
            "Value":     list(input_data.values())
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ================================================================
#  FOOTER
# ================================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:0.85rem; padding:16px 0;">
    ❤️ Heart Disease Risk Predictor · Week 5 ML Deployment Project<br>
    Built with Scikit-Learn · XGBoost · Streamlit · Plotly<br>
    <strong>Dataset:</strong> UCI Heart Disease Repository
</div>
""", unsafe_allow_html=True)
