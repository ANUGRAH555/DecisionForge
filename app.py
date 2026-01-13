import streamlit as st

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="DecisionForge",
    layout="wide"
)

# -------------------------------------------------
# Navigation helper
# -------------------------------------------------
def go_to(page):
    st.switch_page(page)

# -------------------------------------------------
# DARK THEME STYLES (ML PROJECT STYLE)
# -------------------------------------------------
st.markdown("""
<style>

/* HIDE SIDEBAR */
[data-testid="stSidebar"] {
    display: none;
}

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        -45deg,
        #020617,
        #020617,
        #020617,
        #020617
    );
}

/* REMOVE EXTRA PADDING */
.main {
    padding: 1.2rem 2.2rem;
}

/* HERO SECTION */
.hero {
    text-align: center;
    padding: 2.8rem 1rem 1.8rem 1rem;
}

.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    color: #e5e7eb;
    margin-bottom: 0.6rem;
    letter-spacing: 0.6px;
}

.hero p {
    font-size: 1.05rem;
    color: #cbd5f5;
    max-width: 900px;
    margin: auto;
    line-height: 1.7;
}

/* SECTION TITLE */
.section-title {
    font-size: 2.05rem;
    font-weight: 800;
    margin: 2.3rem 0 1.6rem 0;
    color: #e5e7eb;
}

/* DOMAIN CARD BUTTON */
button[kind="secondary"] {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 22px;
    padding: 26px;
    height: 285px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    text-align: left;
    transition: all 0.35s ease;
    cursor: pointer;
    white-space: normal;
    backdrop-filter: blur(12px);
}

/* HOVER EFFECT */
button[kind="secondary"]:hover {
    transform: translateY(-8px);
    border-color: #38bdf8;
    box-shadow: 0 20px 45px rgba(56,189,248,0.45);
}

/* DOMAIN TITLE */
button[kind="secondary"] strong {
    font-size: 1.4rem;
    font-weight: 800;
    color: #e5e7eb;
    display: block;
    margin-bottom: 0.6rem;
}

/* DESCRIPTION */
button[kind="secondary"] p {
    font-size: 0.96rem;
    color: #cbd5e1;
    line-height: 1.65;
    margin-bottom: 1.1rem;
}

/* TAGS */
button[kind="secondary"] em {
    font-size: 0.78rem;
    font-style: normal;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* CONTACT SECTION */
.contact-section {
    margin-top: 3.8rem;
    padding: 2.3rem;
    background: rgba(15, 23, 42, 0.85);
    border-radius: 18px;
    border: 1px solid rgba(56,189,248,0.25);
    backdrop-filter: blur(12px);
}

.contact-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 1rem;
    color: #e5e7eb;
}

.contact-list {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.8;
}

.contact-list a {
    color: #38bdf8;
    text-decoration: none;
    font-weight: 600;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 3.2rem;
    color: #94a3b8;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO
# -------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>DecisionForge</h1>
    <p>
        A unified <b>Machine Learning Decision Intelligence Platform</b> that enables
        HR leaders and business teams to transform enterprise data into confident,
        real-time strategic decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DOMAINS
# -------------------------------------------------
st.markdown('<div class="section-title">Enterprise Intelligence Domains</div>', unsafe_allow_html=True)

row1 = st.columns(3)
row2 = st.columns(3)

domains = [
    ("Retail & E-Commerce Intelligence",
     "Forecast sales, analyze demand patterns, and optimize pricing strategies to maximize revenue.",
     "Sales • Demand • Pricing",
     "pages/retail.py"),

    ("Supply Chain & Inventory Optimization",
     "Predict demand, reduce overstocking, and streamline inventory planning decisions.",
     "Inventory • Forecasting • Optimization",
     "pages/supply_chain.py"),

    ("Banking Fraud & Credit Risk Analytics",
     "Detect suspicious transactions and assess credit risk for safer financial operations.",
     "Fraud • Credit Risk • Banking",
     "pages/banking.py"),

    ("Customer Analytics & Churn Intelligence",
     "Analyze customer behavior, predict churn risk, and design effective retention strategies.",
     "Churn • Segmentation • Loyalty",
     "pages/customer.py"),

    ("HR & Workforce Analytics",
     "Predict employee attrition and analyze workforce trends for strategic HR planning.",
     "Attrition • Workforce • HR",
     "pages/hr.py"),

    ("Insurance Risk & Claims Analytics",
     "Estimate insurance risk, predict claims, and identify potentially fraudulent activities.",
     "Risk • Claims • Insurance",
     "pages/insurance.py"),
]

# -------------------------------------------------
# RENDER DOMAIN CARDS
# -------------------------------------------------
for col, d in zip(row1 + row2, domains):
    with col:
        if st.button(
            f"""
**{d[0]}**

{d[1]}

*{d[2]}*
""",
            use_container_width=True,
            key=d[0]
        ):
            go_to(d[3])

# -------------------------------------------------
# CONTACT
# -------------------------------------------------
st.markdown("""
<div class="contact-section">
    <div class="contact-title">📞 Contact Me</div>
    <div class="contact-list">
        • 🔗 LinkedIn: <a href="https://www.linkedin.com/in/anugrah-pratap-singh-48249028b/" target="_blank">View Profile</a><br>
        • 💻 GitHub: <a href="https://github.com/ANUGRAH555" target="_blank">View Repositories</a><br>
        • ✉️ Email: <a href="mailto:anugrahcse12@gmail.com">anugrahcse12@gmail.com</a><br>
        • 📱 Phone: +91-7068464328
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<div class="footer">
    DecisionForge © 2026 | Enterprise Machine Learning Decision Platform
</div>
""", unsafe_allow_html=True)
