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

# Import enhanced CV function
from section1_cv.image_detection import detect_boxes

# -------------------------
# Step 1: Object Detection
# -------------------------

image_path = "section1_cv/box.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Image not found.")
    exit()

# Use the enhanced function
output_image, boxes = detect_boxes(image)

if not boxes:
    print("No object detected.")
    exit()

# Take the first detected box for classification
# Box format from detect_boxes is (x, y, w, h)
x, y, w, h = boxes[0]

cropped = image[y:y+h, x:x+w]

# Calculate center for reporting
cx = x + w // 2
cy = y + h // 2

print(f"Object detected at coordinates: ({x}, {y}) Center: ({cx}, {cy}) Dimensions: {w}x{h}")
print("Image with detections saved to 'detected_output.jpg'")
cv2.imwrite("detected_output.jpg", output_image)


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