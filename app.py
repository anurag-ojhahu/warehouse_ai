import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import tempfile

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.write("Box Detection · OCR Analysis · Video Monitoring")

# ------------------------------------------------
# RULE ENGINE
# ------------------------------------------------

def generate_response(text: str):
    if not text:
        return "No specific warehouse risk keywords detected."

    text = text.lower()

    if "fragile" in text:
        return "⚠️ FRAGILE detected. Use protective packaging and avoid stacking."

    if "handle with care" in text:
        return "⚠️ Handle With Care detected. Assign trained handling personnel."

    if "glass" in text:
        return "⚠️ Glass material detected. High break risk."

    if "flammable" in text:
        return "🔥 Flammable material detected. Store away from heat sources."

    return "No specific warehouse risk keywords detected."


# ------------------------------------------------
# OCR FUNCTION
# ------------------------------------------------

def extract_text_from_image(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    text = pytesseract.image_to_string(thresh)
    return text.strip()


# ------------------------------------------------
# MODULE SELECTOR
# ------------------------------------------------

module = st.radio("Select Module", ["Image Inspection", "Video Monitoring"])


# =================================================
# IMAGE INSPECTION
# =================================================

if module == "Image Inspection":

    uploaded_image = st.file_uploader(
        "Upload warehouse package image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, width="stretch")

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        extracted_text = extract_text_from_image(image_cv)

        st.subheader("OCR Text")
        if extracted_text:
            st.code(extracted_text)
        else:
            st.write("No text detected.")

        st.subheader("Operational Intelligence")
        result = generate_response(extracted_text)
        st.success(result)


# =================================================
# VIDEO MONITORING
# =================================================

elif module == "Video Monitoring":

    uploaded_video = st.file_uploader(
        "Upload warehouse video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video is not None:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        frame_display = st.empty()

        collected_text = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(rgb_frame, width="stretch")

            text = extract_text_from_image(frame)
            if text:
                collected_text += " " + text

        cap.release()

        st.subheader("Detected Text (Video)")
        if collected_text.strip():
            st.code(collected_text.strip())
        else:
            st.write("No text detected in video.")

        st.subheader("Operational Intelligence")
        result = generate_response(collected_text)
        st.success(result)