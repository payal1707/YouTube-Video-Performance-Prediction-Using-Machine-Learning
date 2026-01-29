import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from googleapiclient.discovery import build
import isodate

# ---------------- BASE DIRECTORY ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("📂 App Directory:", BASE_DIR)

# ---------------- YOUTUBE API SETUP ----------------
API_KEY = "YOUR_API_KEY_HERE"   # 🔴 replace with real API key
youtube = build("youtube", "v3", developerKey=API_KEY)

def parse_duration(duration):
    seconds = isodate.parse_duration(duration).total_seconds()
    return round(seconds / 60, 2)

def get_video_stats(video_id):
    request = youtube.videos().list(
        part="statistics,contentDetails",
        id=video_id
    )
    response = request.execute()

    if not response["items"]:
        return None

    stats = response["items"][0]["statistics"]
    duration = response["items"][0]["contentDetails"]["duration"]

    return {
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
        "duration_minutes": parse_duration(duration)
    }

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model_path = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(model_path):
    st.error("❌ model.pkl not found in project folder")
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ---------------- HEADER ----------------
st.title("📺 YouTube Video Performance Dashboard")
st.subheader("ML-based View Prediction")
st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔗 Real-Time YouTube Prediction", "📊 Manual Prediction"])

# ================= TAB 1 =================
with tab1:
    st.subheader("Real-Time Video Prediction")
    video_url = st.text_input("Enter YouTube Video URL")

    if st.button("🚀 Predict from YouTube"):
        if video_url.strip() == "":
            st.warning("Enter a valid YouTube URL")
        else:
            try:
                if "v=" in video_url:
                    video_id = video_url.split("v=")[1].split("&")[0]
                else:
                    video_id = video_url.split("/")[-1]

                stats = get_video_stats(video_id)

                if stats is None:
                    st.error("Video not found")
                else:
                    input_data = np.array([[
                        stats["likes"],
                        stats["comments"],
                        stats["duration_minutes"],
                        12   # default upload hour
                    ]])

                    prediction = int(model.predict(input_data)[0])

                    st.success(f"📈 Predicted Views: {prediction:,}")
                    st.info(stats)

            except Exception as e:
                st.error(f"Error: {e}")

# ================= TAB 2 =================
with tab2:
    st.subheader("Manual Prediction")

    likes = st.number_input("👍 Likes", min_value=0)
    comments = st.number_input("💬 Comments", min_value=0)
    duration = st.number_input("⏱ Duration (minutes)", min_value=0.0)
    upload_hour = st.slider("🕒 Upload Hour", 0, 23)

    if st.button("Predict Views"):
        input_data = np.array([[likes, comments, duration, upload_hour]])
        prediction = int(model.predict(input_data)[0])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>Predicted Views</h3>
            <h1>{prediction:,}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>Likes</h3>
            <h1>{likes}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
            <h3>Comments</h3>
            <h1>{comments}</h1>
            </div>
            """, unsafe_allow_html=True)

        with st.spinner("Updating dashboard..."):
            time.sleep(1)

        # -------- Visualization --------
        fig, ax = plt.subplots()
        ax.bar(
            ["Likes", "Comments", "Duration", "Upload Hour"],
            [likes, comments, duration, upload_hour]
        )
        st.pyplot(fig)

# ---------------- CSV UPLOAD ----------------
st.markdown("---")
st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    required_cols = ['likes', 'comments', 'duration_minutes', 'upload_hour']

    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain: likes, comments, duration_minutes, upload_hour")
    else:
        df["Predicted_Views"] = model.predict(df[required_cols])
        st.success("Predictions generated")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "youtube_predictions.csv",
            "text/csv"
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Python | ML | Streamlit | YouTube API")
