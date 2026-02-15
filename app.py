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


# ------------------------------------------------
# RULE ENGINE
# ------------------------------------------------

def generate_response(text):
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
def extract_text(image_array):
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Upscale image (critical for small text)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Increase contrast
    gray = cv2.equalizeHist(gray)

    # Apply bilateral filter (keeps edges sharp)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold (better for uneven lighting)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,
        5
    )

    # OCR config
    custom_config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(thresh, config=custom_config)

    return text.strip()

# ------------------------------------------------
# MODULE SELECTOR
# ------------------------------------------------

module = st.radio("Select Module", ["Image Inspection", "Video Monitoring"])


# =================================================
# IMAGE MODULE
# =================================================

if module == "Image Inspection":

    uploaded_image = st.file_uploader(
        "Upload warehouse package image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")
        st.image(image)  # ← SAFE

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        extracted_text = extract_text(image_cv)

        st.subheader("OCR Text")
        if extracted_text:
            st.code(extracted_text)
        else:
            st.write("No text detected.")

        st.subheader("Operational Intelligence")
        result = generate_response(extracted_text)
        st.success(result)


# =================================================
# VIDEO MODULE
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
        stframe = st.empty()

        collected_text = ""
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Only process every 15th frame (important)
            if frame_count % 15 != 0:
                continue

            # Show frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb)

            # Strong preprocessing (same as image)
            text = extract_text(frame)

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