import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Insurance Risk & Claims Analytics", layout="wide")

# -------------------------------------------------
# DARK ML THEME (MATCHES HR PAGE)
# -------------------------------------------------
st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background: linear-gradient(180deg, #020617, #020617);
    color: #e5e7eb;
}

/* HIDE SIDEBAR */
[data-testid="stSidebar"] { display: none; }

/* MAIN CONTAINER */
.block-container {
    padding: 1.6rem 2.2rem;
}

/* RADIO / FORMS / UPLOAD */
section[data-testid="stRadio"],
section[data-testid="stFileUploader"],
div[data-testid="stForm"] {
    background: rgba(2,6,23,0.97);
    border: 1px solid rgba(56,189,248,0.5);
    border-radius: 18px;
    padding: 1.3rem;
    margin-bottom: 1.6rem;
}

/* RADIO TEXT */
section[data-testid="stRadio"] label {
    color: #e5e7eb !important;
    font-weight: 700;
}
section[data-testid="stRadio"] span {
    color: #cbd5f5 !important;
}

/* ALL BUTTONS */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    color: white !important;
    font-weight: 800;
    border-radius: 14px;
    padding: 0.65rem 1.6rem;
    border: none;
    box-shadow: 0 10px 28px rgba(14,165,233,0.45);
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    transform: translateY(-2px);
}

/* FILE UPLOAD BUTTON */
button[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #14b8a6, #22d3ee);
    color: #020617 !important;
    font-weight: 800;
    border-radius: 12px;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    border: 1px solid rgba(56,189,248,0.45);
    overflow: hidden;
}

/* HEADINGS */
h1, h2, h3 { color: #e5e7eb; }

/* DIVIDER */
hr { border: 1px dashed rgba(56,189,248,0.45); }

/* SELECT CURSOR */
div[data-baseweb="select"],
div[data-baseweb="menu"] * { cursor: pointer !important; }

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
for k in ["raw_df", "result_df", "prediction_done", "input_method"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "prediction_done" else False

# -------------------------------------------------
# LOAD MODEL & PREPROCESSOR
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("models/insurance_model.pkl"),
        joblib.load("models/insurance_preprocessor.pkl")
    )

model, preprocessor = load_artifacts()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Insurance Risk & Claims Analytics")
st.write("Detect potentially fraudulent insurance claims using machine learning.")
st.divider()

# -------------------------------------------------
# INPUT METHOD
# -------------------------------------------------
input_method = st.radio(
    "Select Data Input Method:",
    ["Use Sample Dataset", "Manual Entry", "Upload CSV"]
)

if st.session_state.input_method != input_method:
    st.session_state.raw_df = None
    st.session_state.result_df = None
    st.session_state.prediction_done = False
    st.session_state.input_method = input_method

st.divider()

# -------------------------------------------------
# SAMPLE DATASET (FIXED SWITCHING)
# -------------------------------------------------
if input_method == "Use Sample Dataset":
    data_folder = "data"
    files = [f for f in os.listdir(data_folder) if f.endswith(".csv")] if os.path.exists(data_folder) else []

    if not files:
        st.warning("No CSV files found.")
    else:
        selected = st.selectbox("Select dataset:", files)

        # KEY FIX FOR DATASET SWITCHING
        if st.button("Load Dataset", key=f"load_{selected}"):
            st.session_state.raw_df = pd.read_csv(os.path.join(data_folder, selected))
            st.session_state.prediction_done = False
            st.success(f"Loaded: {selected}")

# -------------------------------------------------
# MANUAL ENTRY
# -------------------------------------------------
elif input_method == "Manual Entry":
    with st.form("insurance_manual"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            policy = st.selectbox("Policy Type", ["Comprehensive", "Third Party"])
            vehicle = st.selectbox("Vehicle Type", ["Car", "Bike", "Truck"])

        with c2:
            severity = st.selectbox("Accident Severity", ["Low", "Medium", "High"])
            claim_type = st.selectbox("Claim Type", ["Collision", "Theft", "Fire"])
            amount = st.number_input("Claim Amount", 5000, 500000, 50000)
            tenure = st.number_input("Policy Tenure", 1, 30, 5)
            prev = st.number_input("Previous Claims", 0, 10, 0)

        if st.form_submit_button("Add Claim"):
            st.session_state.raw_df = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "PolicyType": policy,
                "VehicleType": vehicle,
                "AccidentSeverity": severity,
                "ClaimType": claim_type,
                "ClaimAmount": amount,
                "PolicyTenure": tenure,
                "PreviousClaims": prev
            }])
            st.session_state.prediction_done = False
            st.success("Claim added successfully")

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
elif input_method == "Upload CSV":
    file = st.file_uploader("Upload Insurance CSV", type="csv")
    if file:
        st.session_state.raw_df = pd.read_csv(file)
        st.session_state.prediction_done = False
        st.success("CSV uploaded successfully")

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------
if st.session_state.raw_df is None:
    st.info("Please load data to continue.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(st.session_state.raw_df, use_container_width=True)

# -------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------
st.divider()
if st.button("Run Prediction"):
    df = st.session_state.raw_df.copy()
    X = df.drop(columns=["Fraud"], errors="ignore")
    Xp = preprocessor.transform(X)

    df["Fraud Prediction"] = model.predict(Xp)
    df["Fraud Probability (%)"] = (model.predict_proba(Xp)[:, 1] * 100).round(2)

    st.session_state.result_df = df
    st.session_state.prediction_done = True
    st.success("Prediction completed")

# -------------------------------------------------
# RESULTS + BUSINESS INSIGHTS + VISUALS
# -------------------------------------------------
if st.session_state.prediction_done:
    df = st.session_state.result_df.copy()

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    # ---------------- BUSINESS INSIGHTS (UNCHANGED)
    st.divider()
    st.subheader("Insurance Business Insights")

    def risk_bucket(p):
        if p < 30:
            return "Low Risk"
        elif p < 70:
            return "Medium Risk"
        return "High Risk"

    df["Risk Category"] = df["Fraud Probability (%)"].apply(risk_bucket)

    def explain(row):
        reasons = []
        if row["ClaimAmount"] > 100000:
            reasons.append("High claim amount")
        if row["AccidentSeverity"] == "High":
            reasons.append("Severe accident")
        if row["PreviousClaims"] >= 2:
            reasons.append("Multiple past claims")
        if row["PolicyTenure"] <= 2:
            reasons.append("Short policy tenure")
        return ", ".join(reasons) if reasons else "No major risk indicators"

    df["Why This Claim Is Risky"] = df.apply(explain, axis=1)

    st.dataframe(
        df[[
            "Fraud Prediction",
            "Fraud Probability (%)",
            "Risk Category",
            "Why This Claim Is Risky"
        ]],
        use_container_width=True
    )

    # ---------------- VISUAL INSIGHTS (FIXED SIZE)
    if input_method != "Manual Entry":
        st.divider()
        st.subheader("Visual Insights")

        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            df["Risk Category"].value_counts().plot(
                kind="pie", autopct="%1.1f%%", startangle=90, ax=ax1
            )
            ax1.set_title("Risk Category Distribution")
            ax1.set_ylabel("")
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            df["Claim Bucket"] = pd.cut(
                df["ClaimAmount"],
                bins=[0, 50000, 100000, 200000, 500000],
                labels=["Low", "Medium", "High", "Very High"]
            )
            df.groupby("Claim Bucket")["Fraud Probability (%)"].mean().plot(
                marker="o", ax=ax2
            )
            ax2.set_title("Fraud Risk vs Claim Amount")
            ax2.set_ylabel("Avg Fraud Probability (%)")
            st.pyplot(fig2)

    # ---------------- DOWNLOAD
    st.download_button(
        "⬇️ Download Insurance Report",
        df.to_csv(index=False).encode("utf-8"),
        file_name="insurance_fraud_report.csv",
        mime="text/csv"
    )
