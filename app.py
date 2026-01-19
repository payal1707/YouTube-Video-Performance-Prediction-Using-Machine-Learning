import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.title-text {
    font-size: 40px;
    font-weight: bold;
}
.sub-text {
    color: #AAAAAA;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- HEADER ----------------
st.markdown('<div class="title-text">📺 YouTube Video Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Real-time Machine Learning based View Prediction</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎯 Video Input Parameters")

likes = st.sidebar.number_input("👍 Likes", min_value=0, step=10)
comments = st.sidebar.number_input("💬 Comments", min_value=0, step=5)
duration = st.sidebar.number_input("⏱ Duration (minutes)", min_value=0.0, step=0.5)
upload_hour = st.sidebar.slider("🕒 Upload Hour", 0, 23)

# ---------------- PREDICTION ----------------
st.markdown("## 🔮 Real-Time Prediction")

col1, col2, col3 = st.columns(3)

if st.button("🚀 Predict Video Views"):
    input_data = np.array([[likes, comments, duration, upload_hour]])
    prediction = int(model.predict(input_data)[0])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Predicted Views</h3>
            <h1>{prediction:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👍 Likes</h3>
            <h1>{likes}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💬 Comments</h3>
            <h1>{comments}</h1>
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("Updating dashboard..."):
        time.sleep(1)

# ---------------- VISUALIZATION ----------------
st.markdown("## 📊 Engagement Overview")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    ["Likes", "Comments", "Duration (min)", "Upload Hour"],
    [likes, comments, duration, upload_hour]
)
ax.set_facecolor("#0E1117")
fig.patch.set_facecolor("#0E1117")
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.set_title("Current Video Parameters", color="white")

st.pyplot(fig)

# ---------------- CSV UPLOAD ----------------
st.markdown("## 📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    features = df[['likes', 'comments', 'duration_minutes', 'upload_hour']]
    df['Predicted_Views'] = model.predict(features)

    st.success("Predictions Generated Successfully")
    st.dataframe(df)

    st.download_button(
        label="⬇️ Download Predictions",
        data=df.to_csv(index=False),
        file_name="youtube_predictions.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "💡 **Technologies Used:** Python | Pandas | NumPy | Scikit-learn | Streamlit  \n"
    "🎓 **Project Type:** Internship-Ready Machine Learning Dashboard"
)