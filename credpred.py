import streamlit
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        color: #222;
    }
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .sub-title {
        font-size: 16px;
        text-align: center;
        color: #555;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0c5cd7;
        transform: scale(1.05);
    }
    .metric-card {
        background: white;
        padding: 1.2em;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<h1 class="main-title">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict the likelihood of a transaction being fraudulent</p>', unsafe_allow_html=True)

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except:
    st.error("⚠️ Missing model/scaler. Please add 'fraud_model.pkl' and 'scaler.pkl' in the app directory.")
    st.stop()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Time (seconds)", min_value=0, max_value=200000, value=50000, step=1000)
    amount = st.number_input("Amount (€)", min_value=0.0, max_value=5000.0, value=120.0, step=10.0)

with col2:
    st.markdown("##### PCA Features (V1–V28)")
    with st.expander("Enter PCA Feature Values (V1–V28)", expanded=False):
        v_values = []
        for i in range(1, 29):
            val = st.number_input(f"V{i}", -10.0, 10.0, 0.0, key=f"v{i}")
            v_values.append(val)

if st.button("🔍 Predict Fraud", use_container_width=True):
    try:
        features = np.array([[time, amount] + v_values])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]
        
        # -----------------------------
        # Result Display
        # -----------------------------
        st.markdown("<hr>", unsafe_allow_html=True)
        colL, colR = st.columns([2, 1])

        with colL:
            if prediction == 1:
                st.markdown('<div class="metric-card"><h3 style="color:#e53935;">🚨 FRAUD DETECTED</h3>'
                            f'<p><b>Confidence:</b> {prob*100:.2f}%</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h3 style="color:#43a047;">✅ Legitimate Transaction</h3>'
                            f'<p><b>Confidence:</b> {(1-prob)*100:.2f}%</p></div>', unsafe_allow_html=True)
        
        # -----------------------------
        # Gauge Chart (Probability Meter)
        # -----------------------------
        with colR:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Fraud Probability", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1a73e8"},
                    'steps': [
                        {'range': [0, 40], 'color': "#c8e6c9"},
                        {'range': [40, 70], 'color': "#fff59d"},
                        {'range': [70, 100], 'color': "#ffcdd2"}
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr style="margin-top: 3em; border: 0.5px solid #ccc;">
<p style="text-align:center; color:#777;">Powered by XGBoost • Built with Streamlit</p>
""", unsafe_allow_html=True)
