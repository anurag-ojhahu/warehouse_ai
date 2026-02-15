import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.write("OCR-Based Risk Detection · Operational Intelligence")

# -----------------------------------------
# RULE ENGINE
# -----------------------------------------

def generate_response(text):
    text = text.lower()

    if "fragile" in text:
        return "⚠️ FRAGILE detected. Use protective packaging and avoid stacking."

    if "handle with care" in text:
        return "⚠️ Handle With Care detected. Assign to trained handling personnel."

    if "glass" in text:
        return "⚠️ Glass material detected. High break risk."

    if "flammable" in text:
        return "🔥 Flammable material detected. Store away from heat sources."

    return "No specific warehouse risk keywords detected."


# -----------------------------------------
# IMAGE UPLOAD
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Upload warehouse package image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Extract text
    extracted_text = pytesseract.image_to_string(img_cv)

    st.subheader("Extracted Text")
    st.code(extracted_text)

    # Risk analysis
    st.subheader("Operational Intelligence")
    response = generate_response(extracted_text)
    st.success(response)