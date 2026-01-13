import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Banking Fraud & Credit Risk Analytics",
    layout="wide"
)

# -------------------------------------------------
# DARK ML THEME (CONSISTENT WITH HR & INSURANCE)
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

/* RADIO / FORM / UPLOAD */
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

/* BUTTONS */
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

/* FILE UPLOADER BUTTON */
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

/* HEADERS */
h1, h2, h3 { color: #e5e7eb; }

/* DIVIDER */
hr { border: 1px dashed rgba(56,189,248,0.45); }

/* SELECT CURSOR */
div[data-baseweb="select"],
div[data-baseweb="menu"] * {
    cursor: pointer !important;
}

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
        joblib.load("models/banking_model.pkl"),
        joblib.load("models/banking_preprocessor.pkl")
    )

model, preprocessor = load_artifacts()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Banking Fraud & Credit Risk Analytics")
st.write(
    "Detect suspicious banking transactions using machine learning "
    "and generate actionable business insights."
)

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
        st.warning("No datasets found.")
    else:
        selected = st.selectbox("Select dataset:", files)

        if st.button("Load Dataset", key=f"bank_load_{selected}"):
            st.session_state.raw_df = pd.read_csv(os.path.join(data_folder, selected))
            st.session_state.prediction_done = False
            st.success(f"Loaded dataset: {selected}")

# -------------------------------------------------
# MANUAL ENTRY
# -------------------------------------------------
elif input_method == "Manual Entry":
    with st.form("manual_banking"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 90, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            account_type = st.selectbox("Account Type", ["Savings", "Current"])
            txn_amount = st.number_input("Transaction Amount", 100, 500000, 25000)
            txn_type = st.selectbox("Transaction Type", ["ATM", "POS", "Online", "Transfer"])

        with c2:
            balance = st.number_input("Account Balance", 0, 1000000, 100000)
            credit_score = st.number_input("Credit Score", 300, 850, 720)
            merchant = st.selectbox("Merchant Category", ["Retail", "Electronics", "Food", "Travel"])
            device = st.selectbox("Device Type", ["Mobile", "Laptop", "ATM", "POS"])
            location = st.selectbox("Location", ["Domestic", "International"])
            intl = st.selectbox("Is International?", ["Yes", "No"])
            prev_frauds = st.number_input("Previous Frauds", 0, 20, 0)

        if st.form_submit_button("Add Transaction"):
            st.session_state.raw_df = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "AccountType": account_type,
                "TransactionAmount": txn_amount,
                "TransactionType": txn_type,
                "AccountBalance": balance,
                "CreditScore": credit_score,
                "MerchantCategory": merchant,
                "DeviceType": device,
                "Location": location,
                "IsInternational": intl,
                "PreviousFrauds": prev_frauds
            }])
            st.session_state.prediction_done = False
            st.success("Transaction added successfully")

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
elif input_method == "Upload CSV":
    file = st.file_uploader("Upload Banking CSV", type=["csv"])
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
st.subheader("Fraud Prediction")

if st.button("Run Prediction"):
    df = st.session_state.raw_df.copy()

    X = df.drop(columns=["Fraud"], errors="ignore")
    X_processed = preprocessor.transform(X)

    df["Fraud Prediction"] = model.predict(X_processed)
    df["Fraud Probability (%)"] = (model.predict_proba(X_processed)[:, 1] * 100).round(2)

    st.session_state.result_df = df
    st.session_state.prediction_done = True
    st.success("Prediction completed successfully")

# -------------------------------------------------
# RESULTS + VISUALS + BUSINESS INSIGHTS
# -------------------------------------------------
if st.session_state.prediction_done:
    df = st.session_state.result_df

    st.subheader("Prediction Report")
    st.dataframe(df, use_container_width=True)

    if input_method != "Manual Entry":
        st.divider()
        st.subheader("Visual Insights")

        c1, c2 = st.columns(2)

        # LINE PLOT
        with c1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.lineplot(
                x=df["TransactionAmount"],
                y=df["Fraud Probability (%)"],
                marker="o",
                ax=ax1
            )
            ax1.set_title("Fraud Risk vs Transaction Amount")
            ax1.set_ylabel("Fraud Probability (%)")
            st.pyplot(fig1)

        # PIE CHART
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            df["Fraud Prediction"].value_counts().plot(
                kind="pie",
                autopct="%1.1f%%",
                startangle=90,
                ax=ax2
            )
            ax2.set_title("Fraud vs Non-Fraud Share")
            ax2.set_ylabel("")
            st.pyplot(fig2)

        # ---------------- BUSINESS INSIGHTS (UNCHANGED)
        st.subheader("Banking Business Insights")

        fraud_rate = (df["Fraud Prediction"] == 1).mean() * 100
        high_risk = (df["Fraud Probability (%)"] > 70).sum()

        st.markdown(f"""
        **Key Insights**
        - Fraud Rate: **{fraud_rate:.2f}%**
        - High-Risk Transactions (>70%): **{high_risk}**

        **Recommended Actions**
        - Enable real-time fraud alerts  
        - Strengthen device & location rules  
        - Manual verification for high-risk cases  
        """)

    # -------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------
    st.download_button(
        "⬇️ Download Banking Fraud Report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="banking_fraud_report.csv",
        mime="text/csv"
    )
