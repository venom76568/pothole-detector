import streamlit as st
import cv2
import supervision as sv
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
import json
import detect_potholes as dp

st.set_page_config(page_title="Pothole Detector", layout="wide")

# --- Title ---
st.title("🕳️ Road Surface Irregularity Detection")
st.markdown("Enhance road safety with our real-time pothole detection system.")

# --- Sidebar for Navigation ---
option = st.sidebar.radio(
    "Choose Mode",
    ["Detect from Image", "Detect from Video", "Detect from Live Camera", "Model Training Results"]
)

# --- Resize helper ---
def resize_image(image, target_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))

# -------------------- Detection Modes --------------------

if option == "Detect from Image":
    st.header("📷 Upload Image for Pothole Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        detected = dp.detect_potholes(image)
        resized = resize_image(detected, 600)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption="Detected Image", use_column_width=True)

elif option == "Detect from Video":
    st.header("🎞️ Upload Video for Pothole Detection")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output = dp.detect_potholes(frame)
            resized = resize_image(output, 600)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels="RGB", use_column_width=True)
        cap.release()

elif option == "Detect from Live Camera":
    st.header("📡 Live Pothole Detection")
    st.warning("Note: Webcam access is only available when running locally.")
    run_live = st.checkbox("Start Live Detection")
    if run_live:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run_live and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = dp.detect_potholes(frame)
            resized = resize_image(processed, 600)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels="RGB", use_column_width=True)
        cap.release()

elif option == "Model Training Results":
    st.header("📊 Model Training Overview")
    if Path("results/training_metrics.json").exists():
        with open("results/training_metrics.json", "r") as f:
            training_metrics = json.load(f)

        final = training_metrics["final"]
        st.subheader("📌 Final Training Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Loss", f"{final['train_loss']:.4f}")
            st.metric("Val Loss", f"{final['val_loss']:.4f}")
        with col2:
            st.metric("Precision", f"{final['precision']:.2%}")
            st.metric("Recall", f"{final['recall']:.2%}")
            st.metric("mAP@0.5", f"{final['map50']:.2%}")

        st.subheader("📉 Loss Curve")
        st.image("results/loss_curve.png", use_column_width=True)

        st.subheader("📈 Precision / Recall / mAP")
        st.image("results/metrics_curve.png", use_column_width=True)

        st.subheader("📌 Confusion Matrix")
        st.image("results/confusion_matrix.png", use_column_width=True)
    else:
        st.warning("No training results found. Please run training first.")
