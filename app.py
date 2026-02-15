import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import tempfile

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.caption("Classification · Live Monitoring · Operational Intelligence")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(
        torch.load("section2_ml/resnet_classifier.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

class_names = ["Fragile", "Damaged", "Normal"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------------------------
# SMART DECISION ENGINE
# -------------------------------------------------
def generate_decision(predicted_label, confidence, query):

    q = query.lower()

    if confidence < 0.60:
        return "Low confidence prediction. Manual inspection recommended."

    if predicted_label == "Fragile":
        if "stack" in q:
            return "Fragile item detected. Stacking not recommended."
        if "machinery" in q:
            return "Use soft-grip mechanical handling."
        return "Fragile classification confirmed. Handle with care and shock absorption."

    if predicted_label == "Damaged":
        return "Damaged item detected. Isolate and log incident report."

    if predicted_label == "Normal":
        if "fragile" in q:
            return "No fragile indicator detected."
        return "Standard warehouse handling procedure."

    return "No operational rule matched."

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
mode = st.sidebar.radio(
    "Select Module",
    ["Image Inspection", "Video Monitoring", "Live Camera"]
)

query = st.sidebar.text_input("Operational Query (optional)")

# =================================================
# IMAGE INSPECTION
# =================================================
if mode == "Image Inspection":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = class_names[predicted.item()]
        conf = confidence.item()

        color = (0,255,0) if label == "Normal" else (0,165,255) if label == "Fragile" else (0,0,255)

        cv2.rectangle(img_np, (30,30), (img_np.shape[1]-30, img_np.shape[0]-30), color, 3)
        cv2.putText(img_np, f"{label} ({conf*100:.1f}%)",
                    (40,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2)

        col1, col2 = st.columns([2,1])

        with col1:
            st.image(img_np, use_container_width=True)

        with col2:
            st.subheader("Prediction")
            st.write("Class:", label)
            st.write("Confidence:", f"{conf*100:.2f}%")

            decision = generate_decision(label, conf, query)
            st.subheader("System Decision")
            st.info(decision)

# =================================================
# VIDEO MONITORING
# =================================================
elif mode == "Video Monitoring":

    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()

        process = st.checkbox("Start Processing")

        if process:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                input_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)

                label = class_names[predicted.item()]
                conf = confidence.item()

                color = (0,255,0) if label == "Normal" else (0,165,255) if label == "Fragile" else (0,0,255)

                cv2.rectangle(frame, (30,30), (frame.shape[1]-30, frame.shape[0]-30), color, 3)
                cv2.putText(frame, f"{label} ({conf*100:.1f}%)",
                            (40,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2)

                frame_placeholder.image(frame, channels="BGR")

            cap.release()

# =================================================
# LIVE CAMERA
# =================================================
elif mode == "Live Camera":

    run = st.checkbox("Start Live Camera")

    frame_window = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = class_names[predicted.item()]
        conf = confidence.item()

        color = (0,255,0) if label == "Normal" else (0,165,255) if label == "Fragile" else (0,0,255)

        cv2.rectangle(frame, (30,30), (frame.shape[1]-30, frame.shape[0]-30), color, 3)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)",
                    (40,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

        frame_window.image(frame, channels="BGR")

    cap.release()