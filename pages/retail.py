import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Retail & E-Commerce Intelligence", layout="wide")

# -------------------------------------------------
# DARK ML THEME
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background: #020617; color: #e5e7eb; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 1.6rem 2.2rem; }
section[data-testid="stRadio"],
section[data-testid="stFileUploader"],
div[data-testid="stForm"] {
    background: rgba(2,6,23,0.97);
    border: 1px solid rgba(56,189,248,0.5);
    border-radius: 18px;
    padding: 1.3rem;
    margin-bottom: 1.6rem;
}
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg,#0284c7,#0ea5e9);
    color:white;font-weight:800;border-radius:14px;
}
[data-testid="stDataFrame"] {
    border-radius: 16px;
    border: 1px solid rgba(56,189,248,0.45);
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
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("models/retail_model.pkl"),
        joblib.load("models/retail_preprocessor.pkl")
    )

model, preprocessor = load_artifacts()

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Retail & E-Commerce Intelligence")
st.write("Predict high-performing products using machine learning.")
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
    st.session_state.prediction_done = False
    st.session_state.input_method = input_method

st.divider()

# -------------------------------------------------
# SAMPLE DATA
# -------------------------------------------------
if input_method == "Use Sample Dataset":
    files = [f for f in os.listdir("data") if f.endswith(".csv")]
    selected = st.selectbox("Select dataset:", files)
    if st.button("Load Dataset"):
        st.session_state.raw_df = pd.read_csv(f"data/{selected}")
        st.session_state.prediction_done = False
        st.success("Dataset loaded")

# -------------------------------------------------
# MANUAL ENTRY
# -------------------------------------------------
elif input_method == "Manual Entry":
    with st.form("retail_manual"):
        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox("Category", ["Electronics","Clothing","Grocery","Home","Beauty"])
            region = st.selectbox("Region", ["North","South","East","West"])
            season = st.selectbox("Season", ["Regular","Festival","Off-Season"])
            price = st.number_input("Price", 100, 10000, 2000)
        with c2:
            discount = st.selectbox("Discount (%)", [0,5,10,20,30])
            marketing = st.number_input("Marketing Spend", 500, 100000, 10000)
            units = st.number_input("Units Sold", 1, 1000, 100)

        if st.form_submit_button("Add Product"):
            revenue = price * units * (1 - discount/100)
            st.session_state.raw_df = pd.DataFrame([{
                "Category": category,
                "Region": region,
                "Season": season,
                "Price": price,
                "DiscountPercent": discount,
                "MarketingSpend": marketing,
                "UnitsSold": units,
                "Revenue": revenue
            }])
            st.session_state.prediction_done = False

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
elif input_method == "Upload CSV":
    file = st.file_uploader("Upload Retail CSV", type="csv")
    if file:
        st.session_state.raw_df = pd.read_csv(file)
        st.session_state.prediction_done = False

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------
if st.session_state.raw_df is None:
    st.stop()

st.subheader("Data Preview")
st.dataframe(st.session_state.raw_df, use_container_width=True)

# -------------------------------------------------
# RUN PREDICTION (FIXED)
# -------------------------------------------------
st.divider()
st.subheader("Sales Prediction")

if st.button("Run Prediction"):
    df = st.session_state.raw_df.copy()
    Xp = preprocessor.transform(df)

    preds = model.predict(Xp)
    scores = model.decision_function(Xp)
    probs = sigmoid(scores)

    df["High Sales Prediction"] = ["Yes" if p==1 else "No" for p in preds]
    df["High Sales Probability (%)"] = (probs*100).round(2)

    st.session_state.result_df = df
    st.session_state.prediction_done = True
    st.success("Prediction completed")

# -------------------------------------------------
# RESULTS + VISUALS
# -------------------------------------------------
if st.session_state.prediction_done:
    df = st.session_state.result_df

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("Visual Insights")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Price", y="Revenue",
                        hue="High Sales Prediction", ax=ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Category", y="Revenue", ax=ax)
        st.pyplot(fig)

    st.download_button(
        "⬇️ Download Retail Report",
        df.to_csv(index=False).encode(),
        "retail_report.csv",
        "text/csv"
    )
