import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.write("Image Classification · Live Monitoring · Operational Intelligence")

mode = st.sidebar.radio(
    "Select Module",
    ["Image Inspection", "Live Camera"]
)

# -------------------------------------
# LOAD MODEL (Pretrained ResNet18)
# -------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# -------------------------------------
# LOAD IMAGENET LABELS (FROM TORCH)
# -------------------------------------
categories = models.ResNet18_Weights.DEFAULT.meta["categories"]

# -------------------------------------
# TRANSFORM
# -------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------------
# HANDLING ENGINE
# -------------------------------------
def generate_response(label, query):
    label = label.lower()
    query = query.lower()

    if "bottle" in label or "glass" in label:
        return "Fragile object detected. Avoid stacking and use protective packaging."

    if "person" in label:
        return "Human detected in operational zone. Ensure safety compliance."

    if query:
        return f"No specific warehouse rule triggered for detected object: {label}"

    return "No risk indicators detected."


# =====================================
# IMAGE MODE
# =====================================
if mode == "Image Inspection":

    uploaded_file = st.file_uploader("Upload warehouse image", type=["jpg", "jpeg", "png"])

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
            st.image(image)

        with col2:
            st.subheader("Prediction")
            st.write(f"Label: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}%")

            query = st.text_input("Operational Query (optional)")

            if query:
                response = generate_response(predicted_label, query)
                st.info(response)


# =====================================
# LIVE CAMERA MODE
# =====================================
elif mode == "Live Camera":

    st.warning("⚠️ Live camera may not work on HuggingFace Spaces (browser sandbox limitation).")

    run = st.checkbox("Start Live Camera")

    frame_placeholder = st.empty()
    query_live = st.text_input("Operational Query (optional)")

    if run:
        cap = cv2.VideoCapture(0)

        while True:
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

            frame_placeholder.image(frame, channels="BGR")

            if query_live:
                response = generate_response(predicted_label, query_live)
                st.info(response)

        cap.release()