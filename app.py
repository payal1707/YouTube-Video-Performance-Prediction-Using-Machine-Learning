import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Titles */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
}
.sub-title {
    font-size: 18px;
    color: #d1d5db;
    margin-bottom: 30px;
}

/* Glass cards */
.card {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
    text-align: center;
    transition: transform 0.25s ease;
}
.card:hover {
    transform: scale(1.05);
}
.card h3 {
    font-size: 18px;
    color: #e5e7eb;
}
.card h1 {
    font-size: 36px;
    margin-top: 10px;
}

/* Section headings */
.section {
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-title">📺 YouTube Video Performance Analytics</div>
<div class="sub-title">
Real-Time Machine Learning Dashboard with Interactive Predictions
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎯 Video Input Parameters")

likes = st.sidebar.number_input("👍 Likes", min_value=0, step=10)
comments = st.sidebar.number_input("💬 Comments", min_value=0, step=5)
duration = st.sidebar.number_input("⏱ Duration (minutes)", min_value=0.0, step=0.5)
upload_hour = st.sidebar.slider("🕒 Upload Hour", 0, 23)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📂 Batch Prediction", "ℹ️ About"])

# ================= TAB 1 =================
with tab1:
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)

    if st.button("🚀 Predict Video Views"):
        with st.spinner("Running ML model..."):
            time.sleep(1.2)

        input_data = np.array([[likes, comments, duration, upload_hour]])
        prediction = int(model.predict(input_data)[0])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>📈 Predicted Views</h3>
                <h1>{prediction:,}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>👍 Likes</h3>
                <h1>{likes}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card">
                <h3>💬 Comments</h3>
                <h1>{comments}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="card">
                <h3>🕒 Upload Hour</h3>
                <h1>{upload_hour}</h1>
            </div>
            """, unsafe_allow_html=True)

        # Chart
        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        st.subheader("📊 Engagement Overview")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(
            ["Likes", "Comments", "Duration", "Upload Hour"],
            [likes, comments, duration, upload_hour]
        )
        ax.set_facecolor("#0f2027")
        fig.patch.set_facecolor("#0f2027")
        ax.tick_params(colors="white")
        ax.set_title("Current Video Parameters", color="white")

        st.pyplot(fig)

# ================= TAB 2 =================
with tab2:
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.subheader("📂 Batch Prediction (CSV Upload)")

    st.markdown(
        "Upload a CSV with columns: **likes, comments, duration_minutes, upload_hour**"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    REQUIRED_COLUMNS = ['likes', 'comments', 'duration_minutes', 'upload_hour']

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}")
            elif df[REQUIRED_COLUMNS].isnull().sum().any():
                st.error("❌ CSV contains missing values.")
            else:
                with st.spinner("Generating predictions..."):
                    time.sleep(1)

                df['Predicted_Views'] = model.predict(df[REQUIRED_COLUMNS])
                st.success("✅ Predictions generated successfully")
                st.dataframe(df)

                st.download_button(
                    "⬇️ Download Results",
                    df.to_csv(index=False),
                    "youtube_predictions.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ================= TAB 3 =================
with tab3:
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.markdown("""
### ℹ️ About This Project

This project predicts **YouTube video views** using engagement metrics
and presents results through a **real-time interactive analytics dashboard**.

**Key Highlights**
- Real-time ML inference
- Glassmorphism UI using HTML & CSS
- Interactive KPI cards
- Batch prediction via CSV upload
- Professional analytics-style dashboard

**Tech Stack**
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit

🎓 *Built for Data Science Internship & Portfolio*
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Designed & Developed by Payal | Internship-Ready ML Project")
