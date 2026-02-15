import streamlit as st
import easyocr
from PIL import Image
import cv2
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.write("Box Detection · OCR Analysis · Video Monitoring")

mode = st.sidebar.radio(
    "Select Module",
    ["Image Inspection", "Video Monitoring"]
)

# ---------------------------------------------------
# LOAD OCR MODEL (Deep Learning Based)
# ---------------------------------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# ---------------------------------------------------
# RISK RULE ENGINE
# ---------------------------------------------------
def generate_response(text):
    text = text.lower()

    if "fragile" in text:
        return "⚠️ FRAGILE detected. Use protective packaging and avoid stacking."

    if "handle with care" in text:
        return "⚠️ Handle With Care detected. Assign trained personnel."

    if "glass" in text:
        return "⚠️ Glass material detected. High break risk."

    if "flammable" in text:
        return "🔥 Flammable material detected. Store away from heat sources."

    if text.strip() == "":
        return "No readable text detected."

    return "No specific warehouse risk keywords detected."


# ---------------------------------------------------
# OCR + BOX DRAWING FUNCTION
# ---------------------------------------------------
def process_frame(frame):
    results = reader.readtext(frame)

    extracted_text = ""

    for (bbox, text, confidence) in results:
        extracted_text += text + " "

        # Draw bounding box
        pts = np.array(bbox).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Put detected text
        cv2.putText(
            frame,
            text,
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame, extracted_text.strip()


# ===================================================
# IMAGE MODE
# ===================================================
if mode == "Image Inspection":

    uploaded_file = st.file_uploader(
        "Upload warehouse package image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        processed_frame, extracted_text = process_frame(img_cv)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(processed_frame, channels="BGR")

        with col2:
            st.subheader("OCR Text")
            st.code(extracted_text if extracted_text else "No text detected")

            st.subheader("Operational Intelligence")
            st.success(generate_response(extracted_text))


# ===================================================
# VIDEO MODE
# ===================================================
elif mode == "Video Monitoring":

    uploaded_video = st.file_uploader(
        "Upload warehouse video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:

        tfile = open("temp_video.mp4", "wb")
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        frame_placeholder = st.empty()
        text_placeholder = st.empty()
        risk_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, extracted_text = process_frame(frame)

            frame_placeholder.image(processed_frame, channels="BGR")
            text_placeholder.code(extracted_text if extracted_text else "No text detected")
            risk_placeholder.success(generate_response(extracted_text))

        cap.release()