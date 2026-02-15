import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
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

# -------------------------------------
# BOX DETECTION (Simple Contour-Based)
# -------------------------------------
def detect_box(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        approx = cv2.approxPolyDP(
            cnt, 0.02 * cv2.arcLength(cnt, True), True
        )

        if len(approx) == 4 and cv2.contourArea(cnt) > 2000:
            cv2.drawContours(image_bgr, [approx], -1, (0, 255, 0), 3)
            cv2.putText(
                image_bgr,
                "BOX",
                (approx[0][0][0], approx[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    return image_bgr


# -------------------------------------
# OCR WITH PROPER PREPROCESSING
# -------------------------------------
def extract_text(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 150, 255, cv2.THRESH_BINARY
    )[1]

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6"
    )

    return text.strip()


# -------------------------------------
# RULE ENGINE
# -------------------------------------
def rule_engine(text, query):
    text = text.lower()
    query = query.lower()

    if "fragile" in text:
        return "Fragile marking detected. Do not stack. Handle with care."

    if "flammable" in text:
        return "Flammable material detected. Store in ventilated zone."

    if query:
        return "No specific warehouse risk keywords detected."

    return "No specific warehouse risk keywords detected."


# =====================================
# IMAGE MODE
# =====================================
if mode == "Image Inspection":

    uploaded_file = st.file_uploader(
        "Upload warehouse package image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        processed = detect_box(image_bgr.copy())
        extracted_text = extract_text(processed)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            )

        with col2:
            st.subheader("OCR Text")
            if extracted_text:
                st.write(extracted_text)
            else:
                st.write("No text detected")

            query = st.text_input("Operational Query")

            if query or extracted_text:
                response = rule_engine(extracted_text, query)
                st.success(response)


# =====================================
# VIDEO MODE (UPLOAD VIDEO — HF SAFE)
# =====================================
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

        query_video = st.text_input("Operational Query")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = detect_box(frame.copy())
            extracted_text = extract_text(processed)

            frame_placeholder.image(
                cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            )

            if extracted_text:
                response = rule_engine(extracted_text, query_video)
                st.success(response)

        cap.release()