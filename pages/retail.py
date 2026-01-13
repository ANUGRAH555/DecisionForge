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
    page_title="Retail & E-Commerce Intelligence",
    layout="wide"
)

# -------------------------------------------------
# DARK ML THEME (UI ONLY)
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
        joblib.load("models/retail_model.pkl"),
        joblib.load("models/retail_preprocessor.pkl")
    )

model, preprocessor = load_artifacts()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Retail & E-Commerce Intelligence")
st.write(
    "Predict high-performing products using machine learning and "
    "gain actionable retail business insights."
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

        if st.button("Load Dataset", key=f"retail_load_{selected}"):
            st.session_state.raw_df = pd.read_csv(os.path.join(data_folder, selected))
            st.session_state.prediction_done = False
            st.success(f"Loaded dataset: {selected}")

# -------------------------------------------------
# MANUAL ENTRY
# -------------------------------------------------
elif input_method == "Manual Entry":
    with st.form("manual_retail"):
        c1, c2 = st.columns(2)

        with c1:
            category = st.selectbox("Category", ["Electronics", "Clothing", "Grocery", "Home", "Beauty"])
            region = st.selectbox("Region", ["North", "South", "East", "West"])
            season = st.selectbox("Season", ["Regular", "Festival", "Off-Season"])
            price = st.number_input("Price", 100, 10000, 2000)

        with c2:
            discount = st.selectbox("Discount (%)", [0, 5, 10, 20, 30])
            marketing_spend = st.number_input("Marketing Spend", 500, 100000, 10000)
            units_sold = st.number_input("Units Sold", 1, 1000, 100)

        if st.form_submit_button("Add Product"):
            revenue = price * units_sold * (1 - discount / 100)
            st.session_state.raw_df = pd.DataFrame([{
                "Category": category,
                "Region": region,
                "Season": season,
                "Price": price,
                "DiscountPercent": discount,
                "MarketingSpend": marketing_spend,
                "UnitsSold": units_sold,
                "Revenue": revenue
            }])
            st.session_state.prediction_done = False
            st.success("Product added successfully")

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
elif input_method == "Upload CSV":
    file = st.file_uploader("Upload Retail CSV", type=["csv"])
    if file:
        st.session_state.raw_df = pd.read_csv(file)
        st.session_state.prediction_done = False
        st.success("CSV uploaded successfully")

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------
if st.session_state.raw_df is None:
    st.info("Please load a dataset to continue.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(st.session_state.raw_df, use_container_width=True)

# -------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------
st.divider()
st.subheader("Sales Prediction")

if st.button("Run Prediction"):
    df = st.session_state.raw_df.copy()

    X_processed = preprocessor.transform(df)
    preds = model.predict(X_processed)
    probs = model.predict_proba(X_processed)

    df["High Sales Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
    df["High Sales Probability (%)"] = (probs[:, 1] * 100).round(2)

    st.session_state.result_df = df
    st.session_state.prediction_done = True
    st.success("Sales prediction completed successfully")

# -------------------------------------------------
# RESULTS + INSIGHTS
# -------------------------------------------------
if st.session_state.prediction_done:
    df = st.session_state.result_df

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    # ---------------- BUSINESS INSIGHTS ----------------
    st.subheader("📌 Retail Business Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", len(df))
    col2.metric("High Sales %",
                f"{(df['High Sales Prediction'] == 'Yes').mean()*100:.1f}%")
    col3.metric(
        "Avg Revenue (High Sales)",
        f"{df[df['High Sales Prediction']=='Yes']['Revenue'].mean():,.0f}"
        if (df['High Sales Prediction']=='Yes').any() else "N/A"
    )

    # ---------------- VISUAL INSIGHTS ----------------
    if input_method != "Manual Entry":
        st.divider()
        st.subheader("Visual Insights")

        c1, c2 = st.columns(2)

        # SCATTER – Price vs Revenue
        with c1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.scatterplot(
                data=df,
                x="Price",
                y="Revenue",
                hue="High Sales Prediction",
                ax=ax1
            )
            ax1.set_title("Price vs Revenue")
            st.pyplot(fig1)

        # BOX – Revenue by Category
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.boxplot(
                data=df,
                x="Category",
                y="Revenue",
                ax=ax2
            )
            ax2.set_title("Revenue by Category")
            st.pyplot(fig2)

    # -------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------
    st.download_button(
        "⬇️ Download Retail Sales Report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="retail_sales_prediction_report.csv",
        mime="text/csv"
    )
