from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.parking_spaces import PARKING_SPACES
from src.parking_detector import detect_parking_spaces
from src.parking_detector_pixels import detect_parking_spaces_by_pixels


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

def process_video(
    video_file,
    model,
    parking_spaces,
    detection_method,
    confidence_threshold,
    overlap_threshold,
    pixel_threshold,
):
    input_path = Path("outputs/temp_uploaded_video.mp4")
    output_path = Path("outputs/streamlit_video_result.mp4")

    input_path.parent.mkdir(exist_ok=True)

    with open(input_path, "wb") as file:
        file.write(video_file.read())

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        success, frame_bgr = cap.read()

        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if detection_method == "YOLO Based":
            result_rgb, total_count, available_count, occupied_count = detect_parking_spaces(
                image=frame_rgb,
                model=model,
                parking_spaces=parking_spaces,
                confidence_threshold=confidence_threshold,
                overlap_threshold=overlap_threshold,
            )
        else:
            result_rgb, total_count, available_count, occupied_count = detect_parking_spaces_by_pixels(
                image=frame_rgb,
                parking_spaces=parking_spaces,
                pixel_threshold=pixel_threshold,
            )

        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)

        frame_count += 1

        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        status_text.text(f"Processed {frame_count} frames")

    cap.release()
    out.release()

    return output_path, total_count, available_count, occupied_count


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

input_mode = st.radio(
    "Choose input type",
    ["Image", "Video"],
)

model = load_model()

if input_mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload a parking image",
        type=["jpg", "jpeg", "png"],
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a parking video",
        type=["mp4"],
    )
    
st.caption(
    "Note: This demo works best with the same camera angle used when selecting the parking spaces."
)

st.sidebar.title("Project Settings")

detection_method = st.sidebar.selectbox(
    "Detection method",
    ["Pixel Based", "YOLO Based"],
)

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

pixel_threshold = st.sidebar.slider(
    "Pixel threshold",
    min_value=100,
    max_value=3000,
    value=150,
    step=50,
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
    if input_mode == "Image":
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        if detection_method == "YOLO Based":
            result_image, total_count, empty_count, occupied_count = detect_parking_spaces(
                image=image_np,
                model=model,
                parking_spaces=PARKING_SPACES,
                confidence_threshold=confidence_threshold,
                overlap_threshold=overlap_threshold,
            )
        else:
            result_image, total_count, empty_count, occupied_count = detect_parking_spaces_by_pixels(
                image=image_np,
                parking_spaces=PARKING_SPACES,
                pixel_threshold=pixel_threshold,
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
        st.warning(
            "Video processing may take some time. For best results, use the same fixed camera view used when selecting parking spaces."
        )

        if st.button("Process Video"):
            output_video_path, total_count, empty_count, occupied_count = process_video(
                video_file=uploaded_file,
                model=model,
                parking_spaces=PARKING_SPACES,
                detection_method=detection_method,
                confidence_threshold=confidence_threshold,
                overlap_threshold=overlap_threshold,
                pixel_threshold=pixel_threshold,
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Spaces", total_count)
            col2.metric("Available", empty_count)
            col3.metric("Occupied", occupied_count)

            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="Download Result Video",
                    data=file.read(),
                    file_name="smart_parking_video_result.mp4",
                    mime="video/mp4",
                )
else:
    st.info("Upload an image or video to start detection.")