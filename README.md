# 🚀 DecisionForge – Unified Business Decision Intelligence Platform

DecisionForge is a **full-scale, multi-domain Machine Learning decision support system** designed to solve **real-world business problems** using data-driven intelligence.

This platform integrates **multiple industry use-cases** into a **single Streamlit-based application**, enabling organizations to analyze, predict, and optimize business outcomes across departments.

It is built with **production-grade ML pipelines**, robust preprocessing, clean UI, and deployment-ready architecture.

---

## 🌟 Why DecisionForge?

✔ One platform – multiple business domains  
✔ End-to-end ML workflow (data → prediction → insights → visuals)  
✔ Clean, responsive & professional UI  
✔ Real industry datasets & logic  
✔ Deployment-ready (Streamlit Cloud)  
✔ Strong Data Science portfolio project  

---

## 📊 Business Domains Covered

### 🛍 Retail & E-Commerce Intelligence
- Predict **high-performing products**
- Analyze **price vs revenue**
- Identify **category-wise sales patterns**
- Support pricing & marketing decisions

### 🏦 Banking Fraud & Credit Risk Analytics
- Detect **suspicious transactions**
- Analyze **fraud probability**
- Identify **high-risk customers**
- Improve fraud monitoring systems

### 🏥 Insurance Risk & Claims Analytics
- Predict **fraudulent insurance claims**
- Classify **risk categories**
- Provide **explainable fraud reasons**
- Support claim verification teams

### 👥 HR & Workforce Analytics
- Predict **employee attrition**
- Identify **high-risk employees**
- Support HR retention strategies

### 🔄 Supply Chain & Inventory Optimization
- Predict **sales demand**
- Identify **reorder risks**
- Optimize **holding & shortage cost**
- Improve inventory planning

### 📉 Customer Churn Analytics
- Predict **customer churn**
- Analyze churn probability
- Support customer retention strategies

---

## 🧠 Machine Learning Algorithms Used

### 🔹 Classification Algorithms
Used in:
- Banking Fraud
- Insurance Fraud
- HR Attrition
- Customer Churn
- Retail High Sales Classification

Algorithms:
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**

---

### 🔹 Regression Algorithms
Used in:
- Supply Chain Demand Forecasting
- Retail Sales Prediction (numeric outputs)

Algorithms:
- **Decision Tree Regressor**
- **Random Forest Regressor**

---

## 🧩 Data Preprocessing Techniques (Very Important 🔥)

Each domain uses a **saved preprocessing pipeline** to ensure consistency between training & prediction.

### Numerical Features
- **SimpleImputer (strategy = mean)**
- **StandardScaler**

### Categorical Features
- **SimpleImputer (strategy = most_frequent)**
- **OneHotEncoder (handle_unknown = ignore)**

### Pipeline Tools
- **Pipeline**
- **ColumnTransformer**

👉 All preprocessing is saved using `joblib` and reused during prediction.

---

## 📈 Visual Analytics (Domain-Specific)

Each domain has **unique visuals** (no repeated plots):

- 📊 Bar Charts
- 📈 Line Plots
- 🥧 Pie Charts
- 📦 Box Plots
- 🔵 Scatter Plots
- 📉 Area Charts

Visuals are generated using:
- **Matplotlib**
- **Seaborn**

---

## 🛠 Tech Stack

| Category | Tools |
|-------|------|
| Language | Python 3.11 |
| ML Library | Scikit-Learn 1.8.0 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Storage | Joblib |
| Version Control | Git & GitHub |
| Deployment | Streamlit Cloud |

---

## 🖥 Project Architecture

