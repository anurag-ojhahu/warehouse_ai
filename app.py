import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import tempfile

st.set_page_config(page_title="Warehouse Intelligence System", layout="wide")

st.title("Warehouse Intelligence System")
st.write("Box Detection · OCR Analysis · Video Monitoring")

mode = st.sidebar.radio(
    "Select Module",
    ["Image Inspection", "Video Monitoring"]
)

# -----------------------------------------------------
# RULE ENGINE
# -----------------------------------------------------
def generate_response(has_box, extracted_text, query):

    text = extracted_text.lower()
    query = query.lower()

    if "fragile" in text:
        return "Fragile marking detected. Use shock-absorbing packaging. Do not stack."

    if "handle with care" in text:
        return "Handle With Care detected. Manual handling recommended."

    if has_box and "stack" in query:
        return "Box detected. Ensure stacking weight limits are respected."

    if has_box:
        return "Standard box detected. No special risk indicators."

    return "No operational risk detected."


# -----------------------------------------------------
# IMAGE MODE
# -----------------------------------------------------
if mode == "Image Inspection":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Box detection via contours
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_box = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                x,y,w,h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)

                if 0.5 < aspect_ratio < 2.0:
                    has_box = True
                    cv2.rectangle(image_np, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(image_np, "BOX", (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # OCR
        extracted_text = pytesseract.image_to_string(image_np)

        col1, col2 = st.columns([2,1])

        with col1:
            st.image(image_np)

        with col2:
            st.subheader("OCR Text")
            st.write(extracted_text if extracted_text else "No text detected")

            query = st.text_input("Operational Query")

            if query:
                response = generate_response(has_box, extracted_text, query)
                st.success(response)


# -----------------------------------------------------
# VIDEO MODE
# -----------------------------------------------------
elif mode == "Video Monitoring":

    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())

        cap = cv2.VideoCapture(temp_file.name)
        frame_placeholder = st.empty()
        query_video = st.text_input("Operational Query")

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blur, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            has_box = False

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h)

                    if 0.5 < aspect_ratio < 2.0:
                        has_box = True
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        cv2.putText(frame, "BOX", (x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            frame_placeholder.image(frame, channels="BGR")

            if query_video:
                response = generate_response(has_box, "", query_video)
                st.info(response)

        cap.release()