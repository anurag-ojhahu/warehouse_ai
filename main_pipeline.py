import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Import semantic search function
from section3_nlp.semantic_search import query_knowledge_base


# -------------------------
# Load Classifier
# -------------------------

device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("section2_ml/resnet_classifier.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -------------------------
# Step 1: Object Detection
# -------------------------

image = cv2.imread("section1_cv/box.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No object detected.")
    exit()

cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)

cropped = image[y:y+h, x:x+w]

print(f"Object detected at coordinates: ({x}, {y})")


# -------------------------
# Step 2: Classification
# -------------------------

pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
input_tensor = transform(pil_image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    predicted = torch.argmax(output, 1).item()

class_map = {
    0: "Fragile",
    1: "Heavy",
    2: "Standard"
}

predicted_label = class_map[predicted]

print(f"Predicted category: {predicted_label}")


# -------------------------
# Step 3: Semantic Retrieval
# -------------------------

query = f"How should a {predicted_label} package be handled in the warehouse?"
response = query_knowledge_base(query)

print("\nRecommended Handling Procedure:")
print(response)