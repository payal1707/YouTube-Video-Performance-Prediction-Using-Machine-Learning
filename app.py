import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

from googleapiclient.discovery import build
import isodate

# ---------------- YouTube API Setup ----------------
API_KEY = "YOUR_API_KEY"  # 🔴 Replace with your actual API key
youtube = build("youtube", "v3", developerKey=API_KEY)

def parse_duration(duration):
    """Convert ISO 8601 duration to minutes"""
    seconds = isodate.parse_duration(duration).total_seconds()
    return round(seconds / 60, 2)

def get_video_stats(video_id):
    """Fetch likes, comments, and duration of a YouTube video"""
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
.main { background-color: #0E1117; }
.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.title-text { font-size: 40px; font-weight: bold; }
.sub-text { color: #AAAAAA; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- HEADER ----------------
st.markdown('<div class="title-text">📺 YouTube Video Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Machine Learning based View Prediction</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎯 Manual Input Parameters")

likes = st.sidebar.number_input("👍 Likes", min_value=0, step=10)
comments = st.sidebar.number_input("💬 Comments", min_value=0, step=5)
duration = st.sidebar.number_input("⏱ Duration (minutes)", min_value=0.0, step=0.5)
upload_hour = st.sidebar.slider("🕒 Upload Hour", 0, 23)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📡 Real-Time Prediction", "📊 Manual Prediction"])

# ================= TAB 1 =================
with tab1:
    st.subheader("🔗 Real-Time YouTube Prediction")

    video_url = st.text_input("Enter YouTube Video URL")
    real_upload_hour = st.slider("Upload Hour (0–23)", 0, 23, 12)

    if st.button("🚀 Predict Real-Time Video"):
        if video_url.strip() == "":
            st.warning("⚠️ Please enter a valid YouTube URL")
        else:
            try:
                if "v=" in video_url:
                    video_id = video_url.split("v=")[1].split("&")[0]
                else:
                    video_id = video_url.split("/")[-1]

                stats = get_video_stats(video_id)

                if stats is None:
                    st.error("❌ Video not found")
                else:
                    input_data = [[
                        stats["likes"],
                        stats["comments"],
                        stats["duration_minutes"],
                        real_upload_hour
                    ]]

                    prediction = int(model.predict(input_data)[0])

                    st.success(f"📈 Predicted Views: {prediction:,}")
                    st.info(
                        f"👍 Likes: {stats['likes']} | "
                        f"💬 Comments: {stats['comments']} | "
                        f"⏱ Duration: {stats['duration_minutes']} min"
                    )

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ================= TAB 2 =================
with tab2:
    st.subheader("🔮 Manual Prediction")

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
    ["Likes", "Comments", "Duration", "Upload Hour"],
    [likes, comments, duration, upload_hour]
)
ax.set_facecolor("#0E1117")
fig.patch.set_facecolor("#0E1117")
ax.tick_params(colors="white")
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")
ax.set_title("Current Video Parameters", color="white")

st.pyplot(fig)

# ---------------- CSV UPLOAD ----------------
st.markdown("## 📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    features = df[["likes", "comments", "duration_minutes", "upload_hour"]]
    df["Predicted_Views"] = model.predict(features)

    st.success("✅ Predictions Generated")
    st.dataframe(df)

    st.download_button(
        "⬇️ Download Predictions",
        df.to_csv(index=False),
        "youtube_predictions.csv",
        "text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "💡 **Technologies:** Python | Pandas | NumPy | Scikit-learn | Streamlit  \n"
    "🎓 **Project:** Internship / Final-Year Ready ML Dashboard"
)
