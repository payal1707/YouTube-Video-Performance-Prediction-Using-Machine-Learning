# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ---------------------------
# 1️⃣ Page Title
# ---------------------------
st.set_page_config(page_title="YouTube Views Predictor", layout="wide")
st.title("📊 YouTube Video Views Predictor")
st.write("Predict expected YouTube views based on likes, comments, video duration, and upload hour.")

# ---------------------------
# 2️⃣ Load Dataset for Visualization
# ---------------------------
df = pd.read_csv("data/youtube_dataset.csv")

# ---------------------------
# 3️⃣ Load Trained Model
# ---------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# 4️⃣ User Inputs for Prediction
# ---------------------------
st.sidebar.header("Enter Video Details")

likes = st.sidebar.number_input("Likes", min_value=0, value=100)
comments = st.sidebar.number_input("Comments", min_value=0, value=10)
duration = st.sidebar.number_input("Duration (minutes)", min_value=1, value=10)
hour = st.sidebar.number_input("Upload Hour (0-23)", min_value=0, max_value=23, value=12)

# ---------------------------
# 5️⃣ Prediction Button
# ---------------------------
if st.sidebar.button("Predict Views"):
    prediction = model.predict([[likes, comments, duration, hour]])
    st.success(f"Predicted Views: {int(prediction[0])}")

# ---------------------------
# 6️⃣ Visualizations
# ---------------------------
st.subheader("📈 Exploratory Data Analysis")

# Likes vs Views
fig1, ax1 = plt.subplots()
ax1.scatter(df['likes'], df['views'], color='blue', alpha=0.6)
ax1.set_xlabel("Likes")
ax1.set_ylabel("Views")
ax1.set_title("Likes vs Views")
st.pyplot(fig1)

# Actual vs Predicted Views
st.subheader("📊 Model Prediction Performance")

# Predict on whole dataset for visualization
X = df[['likes','comments','duration_minutes','upload_hour']]
y_actual = df['views']
y_pred = model.predict(X)

fig2, ax2 = plt.subplots()
ax2.scatter(y_actual, y_pred, color='green', alpha=0.6)
ax2.plot([y_actual.min(), y_actual.max()],
         [y_actual.min(), y_actual.max()],
         'r--', lw=2)
ax2.set_xlabel("Actual Views")
ax2.set_ylabel("Predicted Views")
ax2.set_title("Actual vs Predicted Views")
st.pyplot(fig2)

# ---------------------------
# 7️⃣ Dataset Preview
# ---------------------------
st.subheader("📋 Sample Dataset")
st.dataframe(df.head())