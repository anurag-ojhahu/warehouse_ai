import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import tempfile

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.write("Image Classification · Video Monitoring · Operational Intelligence")

mode = st.sidebar.radio(
    "Select Module",
    ["Image Inspection", "Video Monitoring"]
)

# -------------------------------------
# LOAD MODEL (Pretrained ResNet18)
# Auto-downloads weights (HF safe)
# -------------------------------------
@st.cache_resource
def load_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model, weights

model, weights = load_model()
categories = weights.meta["categories"]

# -------------------------------------
# TRANSFORM PIPELINE
# -------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------------
# RULE-BASED HANDLING ENGINE
# -------------------------------------
def generate_response(label, query):
    label = label.lower()
    query = query.lower()

    if "bottle" in label or "glass" in label:
        return "Fragile object detected. Avoid stacking and use protective packaging."

    if "person" in label:
        return "Human detected in operational zone. Ensure safety compliance."

    if "box" in label or "crate" in label:
        return "Container detected. Verify weight classification before stacking."

    if query:
        return f"No specific warehouse rule triggered for detected object: {label}"

    return "No risk indicators detected."


# =====================================
# IMAGE MODE
# =====================================
if mode == "Image Inspection":

    uploaded_file = st.file_uploader(
        "Upload warehouse image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)

        predicted_label = categories[top_catid.item()]
        confidence = top_prob.item() * 100

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, width="stretch")

        with col2:
            st.subheader("Prediction")
            st.write(f"Label: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}%")

            query = st.text_input("Operational Query (optional)")

            if query:
                response = generate_response(predicted_label, query)
                st.info(response)


# =====================================
# VIDEO MODE (HF SAFE)
# =====================================
elif mode == "Video Monitoring":

    video_file = st.file_uploader(
        "Upload warehouse video",
        type=["mp4", "mov", "avi"]
    )

    query_live = st.text_input("Operational Query (optional)")

    if video_file:

        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_catid = torch.topk(probabilities, 1)

            predicted_label = categories[top_catid.item()]
            confidence = top_prob.item() * 100

            cv2.putText(
                frame,
                f"{predicted_label} ({confidence:.1f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            frame_placeholder.image(frame, channels="BGR", width="stretch")

            if query_live:
                response = generate_response(predicted_label, query_live)
                st.info(response)

        cap.release()