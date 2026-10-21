
from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import sqlite3
from PIL import Image

MODEL_PATH = Path("models/fruit_quality_model.keras")
CLASS_PATH = Path("models/class_names.json")

DB_PATH = Path("prediction_history.db")


def init_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fruit_name TEXT NOT NULL,
                freshness TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
                    DEFAULT (datetime('now', 'localtime'))
            )
        """)


def save_prediction(fruit_name, freshness, confidence):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (fruit_name, freshness, confidence)
            VALUES (?, ?, ?)
            """,
            (fruit_name, freshness, confidence)
        )


def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT fruit_name, freshness, confidence, created_at
            FROM predictions
            ORDER BY id DESC
        """).fetchall()


init_database()


def delete_all_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions")


st.set_page_config(
    page_title="Fruit Quality Checker",
    page_icon="🍎",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    text-align: center;
    color: #167d4a;
    font-size: 34px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 16px;
}
.result-card {
    padding: 20px;
    border-radius: 14px;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🍎 Fruit & Vegetable Quality Checker</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered fruit and vegetable freshness prediction</div>',
    unsafe_allow_html=True
)

st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r") as file:
        return json.load(file)

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error("Model files nahi mile. Pehle train.py run karo.")
    st.stop()

model = load_model()
class_names = load_class_names()

st.subheader("📸 Upload an Image")
st.write("Choose a fruit or vegetable image to check its predicted freshness.")

camera_file = st.camera_input("📸 Take a photo using your camera")

uploaded_file = st.file_uploader(
    "Select image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if camera_file is not None:
    uploaded_file = camera_file

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("🔍 Check Quality", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing your image..."):
            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            parts = predicted_class.split("_")
            freshness = parts[0].capitalize()
            item_name = " ".join(parts[1:]).title()

            save_prediction(item_name, freshness, confidence)

        st.divider()
        st.subheader("📋 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fruit / Vegetable", item_name)

        with col2:
            st.metric("Predicted Freshness", freshness)

        st.progress(min(max(confidence / 100, 0.0), 1.0))
        st.metric("Model Confidence", f"{confidence:.2f}%")

        if freshness == "Fresh":
            st.success("The model classified this image as Fresh.")
        else:
            st.warning("The model classified this image as Stale.")

        st.caption(
            "Note: This is an AI prediction, not a guarantee of actual food quality. "
            "The model supports only its 12 trained classes."
        )

st.divider()
st.caption("Final Year Project | Fruit and Vegetable Quality Grading")


st.divider()


history = get_history()

st.subheader("📜 Prediction History")
history = get_history()

if history:
    csv_data = pd.DataFrame(
        [
            {
                "Fruit / Vegetable": row[0],
                "Freshness": row[1],
                "Confidence": row[2],
                "Date & Time": row[3],
            }
            for row in history
        ]
    ).to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Prediction History (CSV)",
        data=csv_data,
        file_name="fruit_prediction_history.csv",
        mime="text/csv"
    )


if history:
    st.dataframe(
        [
            {
                "Fruit / Vegetable": row[0],
                "Freshness": row[1],
                "Confidence": f"{row[2]:.2f}%",
                "Date & Time": row[3],
            }
            for row in history
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Abhi koi prediction history nahi hai.")

    
# Dashboard
st.divider()
st.header("📊 Prediction Dashboard")

history = get_history()

total_predictions = len(history)
fresh_count = sum(1 for row in history if row[1] == "Fresh")
stale_count = sum(1 for row in history if row[1] == "Stale")

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total_predictions)
col2.metric("Fresh", fresh_count)
col3.metric("Stale", stale_count)

if total_predictions > 0:
    st.subheader("Fresh vs Stale")

    chart_data = {
        "Category": ["Fresh", "Stale"],
        "Count": [fresh_count, stale_count]
    }

    st.bar_chart(
        {
            "Fresh": [fresh_count],
            "Stale": [stale_count]
        }
    )
else:
    st.info("Dashboard dikhane ke liye pehle fruit images test karo.")


# Delete Prediction History
st.divider()
st.subheader("🗑️ Manage Prediction History")

if st.button("Delete All History"):
    st.session_state["confirm_delete_history"] = True

if st.session_state.get("confirm_delete_history", False):
    st.warning("Kya tum sach mein saari saved prediction history delete karna chahte ho?")

    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button("Yes, Delete All", type="primary"):
            delete_all_history()
            st.session_state["confirm_delete_history"] = False
            st.success("Prediction history delete ho gayi!")
            st.rerun()

    with col_no:
        if st.button("Cancel"):
            st.session_state["confirm_delete_history"] = False
            st.rerun()


# -------------------------------
# SIMULATED AUTOMATIC SORTING
# -------------------------------

st.divider()
st.header("🏭 Automatic Fruit Sorting System")
st.write(
    "AI prediction ke basis par fruits aur vegetables "
    "ko Fresh aur Stale bins mein virtually sort karein."
)

st.info(
    "Note: Yeh software simulation hai. "
    "Physical conveyor belt ya motor connected nahi hai."
)

if "sorting_results" not in st.session_state:
    st.session_state["sorting_results"] = []

if st.button("📦 Add Latest Prediction to Sorting"):
    history = get_history()

    if history:
        latest = history[0]

        st.session_state["sorting_results"].append({
            "Fruit / Vegetable": latest[0],
            "Freshness": latest[1],
            "Confidence": latest[2]
        })

        st.success(
            f"{latest[0]} successfully sent to "
            f"{latest[1]} sorting bin!"
        )
    else:
        st.warning("Pehle kisi fruit ki prediction karo.")

sorting_results = st.session_state["sorting_results"]

fresh_items = [
    item for item in sorting_results
    if item["Freshness"] == "Fresh"
]

stale_items = [
    item for item in sorting_results
    if item["Freshness"] == "Stale"
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Fresh Fruit Bin")
    st.metric("Fresh Items", len(fresh_items))

    if fresh_items:
        st.dataframe(fresh_items, hide_index=True)
    else:
        st.info("Fresh bin abhi empty hai.")

with col2:
    st.subheader("🔴 Stale / Reject Bin")
    st.metric("Stale Items", len(stale_items))

    if stale_items:
        st.dataframe(stale_items, hide_index=True)
    else:
        st.info("Reject bin abhi empty hai.")

if st.button("🔄 Reset Sorting Simulation"):
    st.session_state["sorting_results"] = []
    st.rerun()
