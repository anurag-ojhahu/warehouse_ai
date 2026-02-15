import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import tempfile

st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.write("Box Detection · OCR Analysis · Video Monitoring")

# -----------------------------------------
# RULE ENGINE
# -----------------------------------------

def generate_response(text):
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


# -----------------------------------------
# MODULE SELECTION
# -----------------------------------------

module = st.radio("Select Module", ["Image Inspection", "Video Monitoring"])

# =========================================
# IMAGE MODULE
# =========================================

if module == "Image Inspection":

    uploaded_image = st.file_uploader(
        "Upload warehouse package image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, width="stretch")

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        extracted_text = pytesseract.image_to_string(img_cv)

        st.subheader("OCR Text")
        st.code(extracted_text)

        st.subheader("Operational Intelligence")
        result = generate_response(extracted_text)
        st.success(result)


# =========================================
# VIDEO MODULE
# =========================================

if module == "Video Monitoring":

    uploaded_video = st.file_uploader(
        "Upload warehouse video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        detected_text = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)

            stframe.image(pil_frame, width="stretch")

            text = pytesseract.image_to_string(frame)

            if text.strip() != "":
                detected_text += text

        cap.release()

        st.subheader("Detected Text (Video)")
        st.code(detected_text)

        st.subheader("Operational Intelligence")
        result = generate_response(detected_text)
        st.success(result)