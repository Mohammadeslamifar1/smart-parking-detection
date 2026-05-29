from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.parking_spaces import PARKING_SPACES
from src.parking_detector import detect_parking_spaces


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")



st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🅿️",
    layout="wide",
)

st.title("Smart Parking Space Detection")

st.markdown(
    """
    This computer vision app detects available and occupied parking spaces from a parking lot image.

    It uses a pretrained YOLO model to detect vehicles, then compares the detected vehicle boxes with manually selected parking spaces.
    """
)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a parking image",
    type=["jpg", "jpeg", "png"],
)
st.caption(
    "Note: This demo works best with the same camera angle used when selecting the parking spaces."
)

st.sidebar.title("Project Settings")

st.sidebar.markdown(
    """
    This app detects parking space availability using:

    YOLO object detection  
    OpenCV image processing  
    Custom parking space logic  
    Streamlit web interface
    """
)

st.sidebar.markdown("### Color Legend")
st.sidebar.markdown("Green boxes mean available spaces.")
st.sidebar.markdown("Red boxes mean occupied spaces.")

st.sidebar.markdown("### Model Controls")

confidence_threshold = st.sidebar.slider(
    "YOLO confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
)

overlap_threshold = st.sidebar.slider(
    "Parking overlap threshold",
    min_value=0.05,
    max_value=0.50,
    value=0.12,
    step=0.01,
)

st.sidebar.markdown("### How to use")
st.sidebar.markdown(
    """
    1. Upload a parking lot image.
    2. The model detects vehicles.
    3. The app checks selected parking spaces.
    4. Green spaces are available.
    5. Red spaces are occupied.
    """
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    result_image, total_count, empty_count, occupied_count = detect_parking_spaces(
    image=image_np,
    model=model,
    parking_spaces=PARKING_SPACES,
    confidence_threshold=confidence_threshold,
    overlap_threshold=overlap_threshold,
)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Spaces", total_count)
    col2.metric("Available", empty_count)
    col3.metric("Occupied", occupied_count)

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(image_np, use_container_width=True)

    with right:
        st.subheader("Detection Result")
        st.image(result_image, channels="RGB", use_container_width=True)

    output_path = Path("outputs/streamlit_result.jpg")
    output_path.parent.mkdir(exist_ok=True)

    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result_bgr)

    success, encoded_image = cv2.imencode(".jpg", result_bgr)

    if success:
        st.download_button(
            label="Download Result Image",
            data=encoded_image.tobytes(),
            file_name="smart_parking_result.jpg",
            mime="image/jpeg",
        )

else:
    st.info("Upload an image to start detection.")