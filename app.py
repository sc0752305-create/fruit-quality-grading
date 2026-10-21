
from pathlib import Path
import json
import sqlite3
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/fruit_quality_model.keras")
CLASS_PATH = Path("models/class_names.json")
DB_PATH = Path("prediction_history.db")


# -------------------------------
# DATABASE
# -------------------------------

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
        conn.execute("""
            INSERT INTO predictions
                (fruit_name, freshness, confidence)
            VALUES (?, ?, ?)
        """, (fruit_name, freshness, confidence))


def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT fruit_name, freshness, confidence, created_at
            FROM predictions
            ORDER BY id DESC
        """).fetchall()


def delete_all_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions")


init_database()


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Fruit Quality Checker",
    page_icon="🍎",
    layout="wide"
)

# -------------------------------
# LOGIN SESSION SETUP
# -------------------------------

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
# -------------------------------
# PROFESSIONAL STYLE
# -------------------------------

st.markdown("""
<style>
    .stApp {
        background: #f4f7fb;
        color: #1f2937;
    }

    .block-container {
       max-width: 1250px;
       padding-top: 2rem;
       padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #14532d, #15803d);
        padding: 32px;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(20, 83, 45, 0.12);
    }

    .hero h1 {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e8f5e9;
        font-size: 16px;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
        padding: 12px 16px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    hr {
        border-color: #e2e8f0;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 13px;
        padding: 18px 0 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# LOGIN SYSTEM
# -------------------------------

if not st.session_state["logged_in"]:
    st.markdown("""
<div style="
    background: linear-gradient(120deg, #14532d, #15803d);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 20px;
">
    <h2 style="color: white; margin: 0;">
        🍎 Welcome Back!
    </h2>
    <p style="color: #e8f5e9; margin: 8px 0 0 0;">
        Fruit & Vegetable Quality Grading and Sorting System
    </p>
</div>
""", unsafe_allow_html=True)
    st.write("Please enter your login credentials to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True
        )

    if login_button:
        correct_username = st.secrets["auth"]["username"]
        correct_password = st.secrets["auth"]["password"]

        if (
            username == correct_username
            and password == correct_password
        ):
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Incorrect username or password. Please try again.")

    st.stop()

st.sidebar.success("✅ Logged in")
if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()


# -------------------------------
# HEADER
# -------------------------------


st.markdown("""
<div class="hero">
    <h1>🍎 Fruit & Vegetable Quality Checker</h1>
    <p>AI-powered freshness prediction and virtual sorting system</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 24px;
">
    <h2 style="color: #14532d; margin-bottom: 10px;">
        Welcome to the Quality Checker 👋
    </h2>
    <p style="color: #475569; font-size: 16px;">
        Use AI-powered image classification to check fruit and vegetable
        freshness, simulate sorting, and review prediction reports.
    </p>
</div>
""", unsafe_allow_html=True)
# -------------------------------
# LOAD MODEL
# -------------------------------

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

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Model load nahi ho paya: {error}")
    st.stop()


# -------------------------------
# TABS
# -------------------------------

home_tab, prediction_tab, sorting_tab, history_tab = st.tabs([
    "🏠 Home",
    "📸 Prediction",
    "🏭 Sorting",
    "📊 History & Dashboard"
])


# -------------------------------
# HOME TAB
# -------------------------------

with home_tab:
    st.subheader("Welcome to the Quality Checker")

    st.write(
        "This project uses a trained deep learning model to predict "
        "whether supported fruits and vegetables belong to the Fresh "
        "or Stale category."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:white; padding:22px; border-radius:16px;
                    border:1px solid #dbe7df; min-height:170px;">
            <div style="font-size:30px;">📸</div>
            <h3 style="color:#166534;">Image Prediction</h3>
            <p style="color:#475569;">
                Upload an image or take a photo to check freshness.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:white; padding:22px; border-radius:16px;
                    border:1px solid #dbe7df; min-height:170px;">
            <div style="font-size:30px;">🏭</div>
            <h3 style="color:#166534;">Virtual Sorting</h3>
            <p style="color:#475569;">
                Place the latest prediction into a virtual Fresh or Reject bin.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:white; padding:22px; border-radius:16px;
                    border:1px solid #dbe7df; min-height:170px;">
            <div style="font-size:30px;">📊</div>
            <h3 style="color:#166534;">Reports & History</h3>
            <p style="color:#475569;">
                Review prediction history, statistics and download CSV data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:white; padding:24px; border-radius:16px;
                border:1px solid #dbe7df; margin-top:20px;">
        <h3 style="color:#166534; margin-top:0;">🚀 How to use</h3>
        <p style="color:#475569;">Follow these simple steps to check quality:</p>
        <ol style="color:#334155; line-height:2;">
            <li>Open the <b>Prediction</b> tab.</li>
            <li>Upload a fruit or vegetable image, or take a photo.</li>
            <li>Click <b>Check Quality</b> to view the prediction.</li>
            <li>Open <b>Sorting</b> to place the latest prediction into a virtual bin.</li>
            <li>Open <b>History & Dashboard</b> to review saved predictions.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "ℹ️ The sorting system is a software simulation. "
        "No physical conveyor belt or motor is connected."
    )

# -------------------------------
# PREDICTION TAB
# -------------------------------

with prediction_tab:
    st.subheader("📸 Check Fruit & Vegetable Quality")

    st.write(
        "Upload a supported image or use your camera to predict freshness."
    )

    input_method = st.radio(
        "Choose image source",
        ["Upload Image", "Use Camera"],
        horizontal=True
    )

    if input_method == "Upload Image":
        image_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="upload_image"
        )
    else:
        image_file = st.camera_input(
            "Take a photo",
            key="camera_image"
        )

    if image_file is not None:
        try:
            image = Image.open(image_file).convert("RGB")

            st.image(
                image,
                caption="Selected Image",
                width=350
            )

            if st.button(
                "🔍 Check Quality",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("AI is analyzing the image..."):
                    resized_image = image.resize((224, 224))
                    image_array = np.array(
                        resized_image,
                        dtype=np.float32
                    )
                    image_array = np.expand_dims(image_array, axis=0)

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    predicted_index = int(np.argmax(predictions))
                    predicted_class = class_names[predicted_index]
                    confidence = float(predictions[predicted_index]) * 100

                    parts = predicted_class.split("_")
                    freshness = parts[0].capitalize()
                    item_name = " ".join(parts[1:]).title()

                    save_prediction(
                        item_name,
                        freshness,
                        confidence
                    )

                    st.session_state["latest_prediction"] = {
                        "Fruit / Vegetable": item_name,
                        "Freshness": freshness,
                        "Confidence": confidence
                    }

                st.success("Prediction completed!")

                col1, col2, col3 = st.columns(3)

                col1.metric("🍎 Fruit / Vegetable", item_name)
                col2.metric("🌿 Predicted Freshness", freshness)
                col3.metric("🎯 Model Confidence", f"{confidence:.2f}%")

                st.progress(
                    min(max(confidence / 100, 0.0), 1.0)
                )

                if freshness == "Fresh":
                    st.success(
                        "The model classified this image as Fresh."
                    )
                else:
                    st.warning(
                        "The model classified this image as Stale."
                    )

                st.caption(
                    "AI prediction is not a guarantee of actual food quality. "
                    "Only the model's trained classes are supported."
                )

        except Exception as error:
            st.error(f"Image process nahi ho payi: {error}")


# -------------------------------
# SORTING TAB
# -------------------------------

with sorting_tab:
    st.markdown("""
    <div style="background:linear-gradient(120deg,#14532d,#15803d);
                padding:24px; border-radius:16px; margin-bottom:20px;">
        <h2 style="color:white; margin:0;">
            🏭 Virtual Fruit & Vegetable Sorting
        </h2>
        <p style="color:#e8f5e9; margin:8px 0 0 0;">
            Organize predicted items into Fresh and Stale bins
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Add the latest saved prediction to a virtual Fresh or Stale bin."
    )

    st.info(
        "Software simulation only — no physical conveyor belt or motor "
        "is connected."
    )

    if "sorting_results" not in st.session_state:
        st.session_state["sorting_results"] = []

    if st.button(
        "📦 Add Latest Prediction to Sorting",
        type="primary"
    ):
        history = get_history()

        if history:
            latest = history[0]

            st.session_state["sorting_results"].append({
                "Fruit / Vegetable": latest[0],
                "Freshness": latest[1],
                "Confidence": latest[2]
            })

            st.success(
                f"{latest[0]} sent to the {latest[1]} sorting bin!"
            )
        else:
            st.warning("Pehle Prediction tab mein image test karo.")

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
        st.markdown("### 🟢 Fresh Fruit Bin")
        st.metric("Fresh Items", len(fresh_items))

        if fresh_items:
            st.dataframe(
                pd.DataFrame(fresh_items),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Fresh bin abhi empty hai.")

    with col2:
        st.markdown("### 🔴 Stale / Reject Bin")
        st.metric("Stale Items", len(stale_items))

        if stale_items:
            st.dataframe(
                pd.DataFrame(stale_items),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Stale / Reject bin abhi empty hai.")

    if st.button("🔄 Reset Sorting Simulation"):
        st.session_state["sorting_results"] = []
        st.rerun()


# -------------------------------
# HISTORY & DASHBOARD TAB
# -------------------------------

with history_tab:
    st.markdown("""
    <div style="background:linear-gradient(120deg,#14532d,#15803d);
                padding:24px; border-radius:16px; margin-bottom:20px;">
        <h2 style="color:white; margin:0;">
            📊 Prediction History & Dashboard
        </h2>
        <p style="color:#e8f5e9; margin:8px 0 0 0;">
            Track freshness predictions and view quality statistics
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = get_history()

    total_predictions = len(history)
    fresh_count = sum(
        1 for row in history if row[1] == "Fresh"
    )
    stale_count = sum(
        1 for row in history if row[1] == "Stale"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("📦 Total Predictions", total_predictions)
    col2.metric("🟢 Fresh Items", fresh_count)
    col3.metric("🔴 Stale Items", stale_count)

    if total_predictions > 0:
        st.markdown("### 📈 Fresh vs Stale Quality Analysis")

        chart_data = pd.DataFrame({
            "Category": ["Fresh", "Stale"],
            "Count": [fresh_count, stale_count]
        })

        st.bar_chart(
            chart_data.set_index("Category"),
            y="Count"
        )
    else:
        st.info("Dashboard dekhne ke liye pehle image test karo.")

    st.divider()
    st.markdown("### 📜 Saved Prediction Records")

    if history:
        history_df = pd.DataFrame(
            [
                {
                    "Fruit / Vegetable": row[0],
                    "Freshness": row[1],
                    "Confidence": round(row[2], 2),
                    "Date & Time": row[3]
                }
                for row in history
            ]
        )

        st.download_button(
            label="📥 Download History (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="fruit_prediction_history.csv",
            mime="text/csv"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Abhi koi prediction history nahi hai.")

    st.divider()
    st.markdown("### 🗑️ Manage Saved History")

    if st.button("Delete All History"):
        st.session_state["confirm_delete_history"] = True

    if st.session_state.get("confirm_delete_history", False):
        st.warning(
            "Kya tum sach mein saari saved prediction history delete "
            "karna chahte ho?"
        )

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
# FOOTER
# -------------------------------

st.divider()

st.markdown(
    '<div class="footer">'
    'Final Year Project | Fruit and Vegetable Quality Grading and Sorting'
    '</div>',
    unsafe_allow_html=True
)
