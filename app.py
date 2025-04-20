import streamlit as st
import cv2
import supervision as sv
import tempfile
import numpy as np
from PIL import Image
import detect_potholes as dp

st.set_page_config(page_title="Pothole Detector", layout="wide")

# --- Title ---
st.title("🕳️ Road Surface Irregularity Detection")
st.markdown("Enhance road safety with our real-time pothole detection system.")

# --- Sidebar for Selection ---
option = st.sidebar.radio(
    "Choose Detection Mode",
    ["Detect from Image", "Detect from Video", "Detect from Live Camera"]
)

# --- Resize Function ---
def resize_image(image, target_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))

# --- Image Detection ---
if option == "Detect from Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        detected = dp.detect_potholes(image)
        resized = resize_image(detected, 600)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption="Detected Image", use_column_width=True)

# --- Video Detection ---
elif option == "Detect from Video":
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

# --- Live Camera Detection (limited to local) ---
elif option == "Detect from Live Camera":
    st.warning("Note: Streamlit cannot access webcam directly in hosted environments. Run locally to use this feature.")
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
