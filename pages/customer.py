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
    page_title="Customer Churn Analytics",
    layout="wide"
)

# -------------------------------------------------
# DARK ML THEME (CONSISTENT ACROSS PROJECT)
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
        joblib.load("models/customer_model.pkl"),
        joblib.load("models/customer_preprocessor.pkl")
    )

model, preprocessor = load_artifacts()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Customer Churn Analytics")
st.write(
    "Predict whether a customer is likely to churn and generate actionable retention insights."
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

        if st.button("Load Dataset", key=f"cust_load_{selected}"):
            st.session_state.raw_df = pd.read_csv(os.path.join(data_folder, selected))
            st.session_state.prediction_done = False
            st.success(f"Loaded dataset: {selected}")

# -------------------------------------------------
# MANUAL ENTRY
# -------------------------------------------------
elif input_method == "Manual Entry":
    with st.form("manual_customer"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 100, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure (months)", 0, 120, 12)
            subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            monthly_charges = st.number_input("Monthly Charges", 100, 20000, 999)

        with c2:
            total_charges = st.number_input("Total Charges", 0, 500000, 15000)
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber", "None"])
            support_tickets = st.number_input("Support Tickets", 0, 20, 1)
            usage_hours = st.number_input("Usage Hours / Month", 0, 1000, 120)

        if st.form_submit_button("Add Customer"):
            st.session_state.raw_df = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Tenure": tenure,
                "SubscriptionType": subscription,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "ContractType": contract_type,
                "PaymentMethod": payment_method,
                "InternetService": internet_service,
                "SupportTickets": support_tickets,
                "UsageHours": usage_hours
            }])
            st.session_state.prediction_done = False
            st.success("Customer added successfully")

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
elif input_method == "Upload CSV":
    file = st.file_uploader("Upload Customer CSV", type=["csv"])
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
st.subheader("Churn Prediction")

if st.button("Run Prediction"):
    df = st.session_state.raw_df.copy()

    X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
    X_processed = preprocessor.transform(X)

    df["Churn Prediction"] = model.predict(X_processed)
    df["Churn Probability (%)"] = (model.predict_proba(X_processed)[:, 1] * 100).round(2)

    st.session_state.result_df = df
    st.session_state.prediction_done = True
    st.success("Prediction completed successfully")

# -------------------------------------------------
# RESULTS + VISUALS + INSIGHTS
# -------------------------------------------------
if st.session_state.prediction_done:
    df = st.session_state.result_df

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    if input_method != "Manual Entry":
        st.divider()
        st.subheader("Visual Insights")

        c1, c2 = st.columns(2)

        # LINE PLOT (FIXED SIZE)
        with c1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.lineplot(
                x=df["Tenure"],
                y=df["Churn Probability (%)"],
                marker="o",
                ax=ax1
            )
            ax1.set_title("Churn Risk vs Tenure")
            ax1.set_ylabel("Churn Probability (%)")
            st.pyplot(fig1)

        # PIE CHART (FIXED SIZE)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            df["Churn Prediction"].value_counts().plot(
                kind="pie",
                autopct="%1.1f%%",
                startangle=90,
                ax=ax2
            )
            ax2.set_title("Churn vs Retained Share")
            ax2.set_ylabel("")
            st.pyplot(fig2)

        # ---------------- BUSINESS INSIGHTS (UNCHANGED)
        st.subheader("Business Insights")

        high_risk = (df["Churn Probability (%)"] > 70).sum()

        st.markdown(f"""
        - 🔴 High-risk customers: **{high_risk}**
        - 🎯 Focus on long-term contracts
        - 💬 Improve support response time
        - 🎁 Offer loyalty & retention benefits
        """)

    # -------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------
    st.download_button(
        "⬇️ Download Customer Churn Report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_report.csv",
        mime="text/csv"
    )
