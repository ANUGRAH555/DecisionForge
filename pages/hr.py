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
    page_title="HR & Workforce Analytics",
    layout="wide"
)

# -------------------------------------------------
# DARK ML THEME (UI ONLY)
# -------------------------------------------------
st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background: radial-gradient(circle at top, #020617, #020617);
    color: #e5e7eb;
}

/* HIDE SIDEBAR */
[data-testid="stSidebar"] {
    display: none;
}

/* MAIN CONTAINER */
.block-container {
    padding: 1.6rem 2.2rem;
}

/* RADIO / FORMS / UPLOAD BOX */
section[data-testid="stRadio"],
section[data-testid="stFileUploader"],
div[data-testid="stForm"] {
    background: rgba(2, 6, 23, 0.96);
    border: 1px solid rgba(56,189,248,0.45);
    border-radius: 18px;
    padding: 1.3rem;
    margin-bottom: 1.5rem;
}

/* RADIO LABELS */
section[data-testid="stRadio"] label {
    color: #e5e7eb !important;
    font-weight: 600;
}

section[data-testid="stRadio"] span {
    color: #cbd5f5 !important;
}

/* BUTTONS (ALL) */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    border-radius: 14px;
    color: #ffffff !important;
    font-weight: 700;
    border: none;
    padding: 0.65rem 1.6rem;
    box-shadow: 0 8px 24px rgba(14,165,233,0.45);
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
h1, h2, h3 {
    color: #e5e7eb;
}

/* DIVIDER */
hr {
    border: 1px dashed rgba(56,189,248,0.45);
}

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
for key in ["raw_df", "result_df", "prediction_done", "input_method"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "prediction_done" else False

# -------------------------------------------------
# LOAD MODEL & PREPROCESSOR
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/hr_model.pkl")
    preprocessor = joblib.load("models/hr_preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("HR & Workforce Analytics")
st.write("Predict employee attrition and generate HR-ready insights.")
st.divider()

# -------------------------------------------------
# INPUT METHOD
# -------------------------------------------------
input_method = st.radio(
    "Select Data Input Method:",
    ["Use Sample Dataset", "Manual Entry", "Upload CSV"]
)

# RESET WHEN METHOD CHANGES
if st.session_state.input_method != input_method:
    st.session_state.raw_df = None
    st.session_state.result_df = None
    st.session_state.prediction_done = False
    st.session_state.input_method = input_method

st.divider()

# -------------------------------------------------
# OPTION 1: SAMPLE DATASET (FIXED)
# -------------------------------------------------
if input_method == "Use Sample Dataset":
    data_folder = "data"
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")] if os.path.exists(data_folder) else []

    if not csv_files:
        st.warning("No CSV files found in data/ folder.")
    else:
        selected_file = st.selectbox("Select sample dataset:", csv_files)

        # 🔑 KEY FIX — button depends on selected file
        if st.button("Load Sample Dataset", key=f"load_{selected_file}"):
            st.session_state.raw_df = pd.read_csv(
                os.path.join(data_folder, selected_file)
            )
            st.session_state.prediction_done = False
            st.success(f"Loaded dataset: {selected_file}")

# -------------------------------------------------
# OPTION 2: MANUAL ENTRY
# -------------------------------------------------
elif input_method == "Manual Entry":
    with st.form("manual_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            department = st.selectbox("Department", ["IT", "HR", "Sales", "Finance", "Operations"])
            job_role = st.text_input("Job Role", "Software Engineer")

        with c2:
            income = st.number_input("Monthly Income", 10000, 200000, 40000)
            satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            years = st.number_input("Years at Company", 0, 40, 5)

        if st.form_submit_button("Add Employee"):
            st.session_state.raw_df = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Department": department,
                "JobRole": job_role,
                "MonthlyIncome": income,
                "JobSatisfaction": satisfaction,
                "OverTime": overtime,
                "YearsAtCompany": years
            }])
            st.session_state.prediction_done = False
            st.success("Employee data added.")

# -------------------------------------------------
# OPTION 3: CSV UPLOAD
# -------------------------------------------------
elif input_method == "Upload CSV":
    uploaded = st.file_uploader("Upload HR CSV", type="csv")

    if uploaded:
        st.session_state.raw_df = pd.read_csv(uploaded)
        st.session_state.prediction_done = False
        st.success("CSV uploaded successfully.")

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
if st.button("Run Attrition Prediction"):
    X = st.session_state.raw_df.drop(columns=["Attrition"], errors="ignore")
    Xp = preprocessor.transform(X)

    df = st.session_state.raw_df.copy()
    df["Predicted Attrition"] = model.predict(Xp)
    df["Attrition Probability (%)"] = (model.predict_proba(Xp)[:, 1] * 100).round(2)

    st.session_state.result_df = df
    st.session_state.prediction_done = True
    st.success("Prediction completed successfully.")

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
if st.session_state.prediction_done:
    df = st.session_state.result_df

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    if input_method != "Manual Entry":
        st.subheader("HR Visual Insights")

        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots()
            df["Predicted Attrition"].value_counts().plot(
                kind="bar", ax=ax1, color=["#22c55e", "#ef4444"]
            )
            ax1.set_title("Attrition Count")
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(data=df, x="Department", y="Attrition Probability (%)", ax=ax2)
            ax2.set_title("Attrition Risk by Department")
            st.pyplot(fig2)

    st.download_button(
        "⬇️ Download HR Report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="hr_attrition_report.csv",
        mime="text/csv"
    )