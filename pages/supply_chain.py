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
    page_title="Supply Chain & Inventory Optimization",
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
        joblib.load("models/supply_chain_model.pkl"),
        joblib.load("models/supply_chain_preprocessor.pkl")
    )

model, preprocessor = load_artifacts()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Supply Chain & Inventory Optimization")
st.write(
    "Optimize inventory levels, predict demand, and identify reorder risks "
    "using machine learning."
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

        if st.button("Load Dataset", key=f"supply_load_{selected}"):
            st.session_state.raw_df = pd.read_csv(os.path.join(data_folder, selected))
            st.session_state.prediction_done = False
            st.success(f"Loaded dataset: {selected}")

# -------------------------------------------------
# MANUAL ENTRY
# -------------------------------------------------
elif input_method == "Manual Entry":
    with st.form("manual_supply"):
        c1, c2 = st.columns(2)

        with c1:
            product_id = st.text_input("Product ID", "P1001")
            category = st.selectbox("Product Category", ["Electronics", "Grocery", "Clothing", "Furniture"])
            warehouse = st.selectbox("Warehouse Location", ["North", "South", "East", "West"])
            supplier = st.selectbox("Supplier", ["Supplier A", "Supplier B", "Supplier C"])
            lead_time = st.number_input("Lead Time (days)", 1, 60, 15)

        with c2:
            daily = st.number_input("Daily Demand", 1, 1000, 50)
            monthly = st.number_input("Monthly Demand", 10, 50000, 1500)
            stock = st.number_input("Current Stock", 0, 100000, 500)
            reorder = st.number_input("Reorder Point", 0, 50000, 300)
            holding = st.number_input("Holding Cost", 1.0, 500.0, 20.0)
            shortage = st.number_input("Shortage Cost", 1.0, 1000.0, 80.0)

        if st.form_submit_button("Add Product"):
            st.session_state.raw_df = pd.DataFrame([{
                "ProductID": product_id,
                "ProductCategory": category,
                "WarehouseLocation": warehouse,
                "Supplier": supplier,
                "LeadTime": lead_time,
                "DailyDemand": daily,
                "MonthlyDemand": monthly,
                "CurrentStock": stock,
                "ReorderPoint": reorder,
                "HoldingCost": holding,
                "ShortageCost": shortage
            }])
            st.session_state.prediction_done = False
            st.success("Product added successfully")

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
elif input_method == "Upload CSV":
    file = st.file_uploader("Upload Supply Chain CSV", type=["csv"])
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

df = st.session_state.raw_df.copy()

st.subheader("Data Preview")
st.dataframe(df, use_container_width=True)

# -------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------
st.divider()
st.subheader("Sales & Inventory Prediction")

required_cols = {
    "ProductID", "ProductCategory", "WarehouseLocation", "Supplier",
    "LeadTime", "DailyDemand", "MonthlyDemand", "CurrentStock",
    "ReorderPoint", "HoldingCost", "ShortageCost"
}

if not required_cols.issubset(df.columns):
    st.error(f"Missing required columns: {list(required_cols - set(df.columns))}")
    st.stop()

if st.button("Run Prediction"):
    X = df[list(required_cols)]
    Xp = preprocessor.transform(X)
    preds = model.predict(Xp)

    result = df.copy()
    result["Predicted Sales"] = preds.round(2)
    result["Stock Status"] = result.apply(
        lambda x: "⚠️ Reorder Required" if x["CurrentStock"] < x["ReorderPoint"] else "✅ Stock Sufficient",
        axis=1
    )
    result["Estimated Holding Cost"] = (result["CurrentStock"] * result["HoldingCost"]).round(2)
    result["Estimated Shortage Risk Cost"] = (
        (result["ReorderPoint"] - result["CurrentStock"]).clip(lower=0)
        * result["ShortageCost"]
    ).round(2)

    st.session_state.result_df = result
    st.session_state.prediction_done = True
    st.success("Prediction completed successfully")

# -------------------------------------------------
# RESULTS + VISUALS
# -------------------------------------------------
if st.session_state.prediction_done:
    result = st.session_state.result_df

    st.subheader("Optimization Results")
    st.dataframe(result, use_container_width=True)

    if input_method != "Manual Entry":
        st.divider()
        st.subheader("Visual Insights")

        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            ax1.plot(result["MonthlyDemand"], label="Monthly Demand", marker="o")
            ax1.plot(result["Predicted Sales"], label="Predicted Sales", marker="s")
            ax1.set_title("Demand vs Predicted Sales")
            ax1.legend()
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.fill_between(range(len(result)), result["CurrentStock"], alpha=0.5, label="Current Stock")
            ax2.fill_between(range(len(result)), result["ReorderPoint"], alpha=0.5, label="Reorder Point")
            ax2.set_title("Inventory vs Reorder Threshold")
            ax2.legend()
            st.pyplot(fig2)

    st.download_button(
        "⬇️ Download Supply Chain Optimization Report",
        result.to_csv(index=False).encode("utf-8"),
        file_name="supply_chain_optimization_report.csv",
        mime="text/csv"
    )
