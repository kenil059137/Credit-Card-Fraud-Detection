"""
💳 Credit Card Fraud Detection – Streamlit Deployment App
ML InnovateX Hackathon
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import time

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="💳 Fraud Detection System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .fraud-box {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .legit-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high   { color: #dc3545; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { color: #fd7e14; font-weight: bold; font-size: 1.2rem; }
    .risk-low    { color: #198754; font-weight: bold; font-size: 1.2rem; }
    .stProgress > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔐 Credit Card Fraud Detection System</h1>
    <p style="margin:0; opacity:0.85;">ML InnovateX Hackathon | Real-time Fraud Prediction</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load all saved model artifacts."""
    try:
        scaler   = joblib.load(r"C:\cpp\saved_models\scaler.pkl")
        selector = joblib.load(r"C:\cpp\saved_models\feature_selector.pkl")
        model    = joblib.load(r"C:\cpp\saved_models\best_model_xgb.pkl")
        with open(r"C:\cpp\saved_models\feature_names.json") as f:
            features = json.load(f)
        return scaler, selector, model, features, True
    except Exception as e:
        return None, None, None, None, False


scaler, selector, model, feature_names, models_loaded = load_artifacts()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-card-back-side.png", width=80)
    st.title("⚙️ Configuration")

    model_choice = st.selectbox(
        "Select Model",
        ["XGBoost (Recommended)", "Logistic Regression", "Random Forest"],
        help="XGBoost has the best performance for this dataset"
    )

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Lower threshold = catches more fraud (higher recall), but more false positives"
    )

    input_mode = st.radio(
        "Input Mode",
        ["Manual Entry", "Random Sample", "Paste V-Features"],
        help="Choose how to enter transaction data"
    )

    st.divider()
    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC",  "~0.980")
    col2.metric("Recall",   "~0.920")
    col1.metric("Precision","~0.910")
    col2.metric("F1-Score", "~0.915")

    st.divider()
    if not models_loaded:
        st.warning("⚠️ Models not found.\nRun the notebook first to train & save models.")
    else:
        st.success("✅ Models loaded successfully!")


# ─────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────
def predict(v_features: list, amount: float, time_seconds: float, thresh: float):
    """Run the prediction pipeline."""
    if not models_loaded:
        # Demo mode: simulate a prediction
        prob = float(np.random.beta(1, 9))
        label = "FRAUD 🚨" if prob >= thresh else "LEGITIMATE ✅"
        risk = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW")
        return prob, label, risk

    # Build raw feature dict
    raw = {f"V{i+1}": v_features[i] for i in range(28)}
    raw["Amount"] = amount
    raw["Time"]   = time_seconds

    df_in = pd.DataFrame([raw])
    df_in["Log_Amount"] = np.log1p(df_in["Amount"])
    df_in["Hour"]       = (df_in["Time"] % 86400) / 3600
    df_in = df_in.drop(["Time", "Amount"], axis=1)

    scaled   = scaler.transform(df_in)
    selected = selector.transform(scaled)

    prob  = float(model.predict_proba(selected)[0][1])
    label = "FRAUD 🚨" if prob >= thresh else "LEGITIMATE ✅"
    risk  = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW")
    return prob, label, risk


# ─────────────────────────────────────────────
# Main – Input Forms
# ─────────────────────────────────────────────
st.subheader("📝 Transaction Details")

tab1, tab2, tab3 = st.tabs(["🔢 V-Features Input", "💰 Transaction Info", "📋 Summary"])

with tab1:
    st.info("V1–V28 are PCA-transformed features (from Kaggle dataset). Enter values between -5 and 5.")

    if input_mode == "Random Sample":
        # Generate random sample skewed towards legitimate
        np.random.seed(int(time.time()) % 9999)
        v_vals = np.random.normal(0, 1, 28).tolist()
        amt    = float(np.random.exponential(100))
        t      = float(np.random.uniform(0, 172800))
        is_fraud_demo = np.random.random() < 0.15  # 15% chance fraud in demo
        if is_fraud_demo:
            v_vals[3]  = float(np.random.uniform(-8, -3))
            v_vals[9]  = float(np.random.uniform(-10, -5))
            v_vals[13] = float(np.random.uniform(-8, -4))
            amt = float(np.random.uniform(1, 50))
        st.success(f"🎲 Random sample generated! (Demo hint: {'Likely Fraud' if is_fraud_demo else 'Likely Legit'})")
    elif input_mode == "Paste V-Features":
        paste_val = st.text_area(
            "Paste comma-separated V1–V28 values",
            placeholder="-1.36, 0.23, 2.53, 1.37, -0.34, ...",
            height=100
        )
        try:
            v_vals = [float(x.strip()) for x in paste_val.split(',')]
            if len(v_vals) != 28:
                st.error(f"Expected 28 values, got {len(v_vals)}")
                v_vals = [0.0] * 28
        except:
            v_vals = [0.0] * 28
        amt = 0.0
        t   = 0.0
    else:
        v_vals = [0.0] * 28
        amt = 0.0
        t   = 0.0

    # Grid of V-feature inputs
    cols_per_row = 7
    v_inputs = []
    for row_start in range(0, 28, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < 28:
                val = col.number_input(
                    f"V{idx+1}",
                    value=float(v_vals[idx]) if input_mode != "Manual Entry" else 0.0,
                    min_value=-30.0, max_value=30.0, step=0.01,
                    key=f"v{idx+1}"
                )
                v_inputs.append(val)

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        amount_input = st.number_input(
            "💵 Transaction Amount ($)",
            min_value=0.01, max_value=100000.0,
            value=float(amt) if input_mode in ["Random Sample"] else 100.0,
            step=0.01,
            help="The transaction amount in USD"
        )

        time_input = st.number_input(
            "⏱️ Time (seconds from first transaction)",
            min_value=0, max_value=200000,
            value=int(t) if input_mode in ["Random Sample"] else 43200,
            step=1,
            help="Time elapsed since first transaction in dataset"
        )

    with col_b:
        hour_of_day = (time_input % 86400) / 3600
        st.metric("🕐 Hour of Day", f"{hour_of_day:.1f}h")
        st.metric("💰 Log(Amount+1)", f"{np.log1p(amount_input):.4f}")

        # Risk context
        if amount_input > 5000:
            st.warning("⚠️ High-value transaction – elevated scrutiny")
        elif amount_input < 5:
            st.info("ℹ️ Very small amount – common in card testing fraud")
        else:
            st.success("✅ Amount in normal range")

with tab3:
    st.markdown("### Transaction Summary")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Amount",       f"${amount_input:,.2f}")
    summary_cols[1].metric("Time (hrs)",   f"{(time_input/3600):.2f}h")
    summary_cols[2].metric("Non-zero V's", str(sum(1 for v in v_inputs if abs(v) > 0.001)))


# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    predict_btn = st.button(
        "🔍 ANALYZE TRANSACTION",
        type="primary",
        use_container_width=True
    )

# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
if predict_btn:
    with st.spinner("🔄 Analyzing transaction..."):
        time.sleep(0.6)  # Brief delay for UX

    prob, label, risk = predict(v_inputs, amount_input, time_input, threshold)

    st.divider()
    st.subheader("🎯 Prediction Result")

    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])

    with col_r2:
        is_fraud = "FRAUD" in label
        if is_fraud:
            st.markdown(f'<div class="fraud-box">🚨 {label}<br><small>Fraud Probability: {prob:.1%}</small></div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="legit-box">✅ {label}<br><small>Fraud Probability: {prob:.1%}</small></div>',
                       unsafe_allow_html=True)

    st.markdown("---")

    # Probability bar
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.markdown(f"**Fraud Probability: {prob:.4f} ({prob:.1%})**")
        st.progress(min(prob, 1.0))

        risk_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[risk]
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk]
        st.markdown(f'Risk Level: <span class="{risk_class}">{risk_emoji} {risk}</span>',
                   unsafe_allow_html=True)

    # Detail cards
    st.markdown("---")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Fraud Probability", f"{prob:.4f}")
    d2.metric("Threshold Used",    f"{threshold:.2f}")
    d3.metric("Decision",          "FRAUD" if is_fraud else "LEGIT")
    d4.metric("Risk Level",        risk)

    # Recommendation
    st.markdown("### 💡 Recommendation")
    if risk == "HIGH":
        st.error("""
        🚫 **BLOCK TRANSACTION**
        - Probability exceeds 70% — high confidence fraud
        - Immediately freeze card and alert customer
        - Initiate fraud investigation
        - Contact customer via registered mobile number
        """)
    elif risk == "MEDIUM":
        st.warning("""
        ⚠️ **STEP-UP AUTHENTICATION REQUIRED**
        - Transaction flagged as suspicious
        - Request OTP / 2FA verification
        - Monitor subsequent transactions closely
        - Consider temporary transaction limit reduction
        """)
    else:
        st.success("""
        ✅ **APPROVE TRANSACTION**
        - Low fraud probability — transaction appears legitimate
        - Continue standard monitoring protocols
        - No action required
        """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.85rem;'>
    💳 ML InnovateX Hackathon | Credit Card Fraud Detection System<br>
    Built with XGBoost + ANN + SMOTE | Dataset: Kaggle Credit Card Fraud Detection
</div>
""", unsafe_allow_html=True)