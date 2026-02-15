import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

st.title("Warehouse Intelligence System")
st.write("Box Detection · Classification · Operational Intelligence")

# ----------------------------------------
# LOAD MODEL (Pretrained ResNet18)
# ----------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# ----------------------------------------
# LOAD IMAGENET LABELS (AUTO FROM WEIGHTS)
# ----------------------------------------
weights = models.ResNet18_Weights.DEFAULT
categories = weights.meta["categories"]

# ----------------------------------------
# IMAGE TRANSFORM
# ----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------------------
# SIMPLE RULE ENGINE
# ----------------------------------------
def generate_response(label, query):
    label = label.lower()
    query = query.lower()

    if "box" in label or "package" in label:
        return "Standard warehouse box detected. Ensure proper stacking."

    if "bottle" in label or "glass" in label:
        return "Fragile object detected. Avoid stacking and use protective padding."

    if "person" in label:
        return "Human detected in operational zone. Follow safety protocol."

    if query:
        return f"No specific warehouse rule triggered for detected object: {label}"

    return "No operational risk detected."

# =========================================
# IMAGE UPLOAD
# =========================================
uploaded_file = st.file_uploader("Upload warehouse image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ----------------------------------------
    # OPENCV BOX DETECTION
    # ----------------------------------------
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                image_np,
                f"W:{w}px H:{h}px",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    # ----------------------------------------
    # CLASSIFICATION
    # ----------------------------------------
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

    predicted_label = categories[top_catid.item()]
    confidence = top_prob.item() * 100

    # ----------------------------------------
    # DISPLAY
    # ----------------------------------------
    col1, col2 = st.columns([2,1])

    with col1:
        st.image(image_np)

    with col2:
        st.subheader("Classification")
        st.write(f"Label: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}%")

        query = st.text_input("Operational Query (optional)")

        if query:
            response = generate_response(predicted_label, query)
            st.success(response)