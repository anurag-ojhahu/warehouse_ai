import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import tempfile
import torch
from torchvision import models, transforms
from section1_cv.image_detection import detect_boxes
from section3_nlp.semantic_search import query_knowledge_base

st.set_page_config(
    page_title="Warehouse Intelligence System",
    layout="wide"
)

# ------------------------------------------------
# MODEL LOADING
# ------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    try:
        model.load_state_dict(torch.load("section2_ml/resnet_classifier.pth", map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'section2_ml/resnet_classifier.pth' exists.")
        return None

model_resnet = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_map = {0: "Fragile", 1: "Heavy", 2: "Standard"}

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def process_frame(frame):
    # Detect
    output_image, boxes = detect_boxes(frame)
    detected_objects = []

    if boxes and model_resnet:
        for (x, y, w, h) in boxes:
            cropped = frame[y:y+h, x:x+w]
            try:
                pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_crop).unsqueeze(0)
                with torch.no_grad():
                    output = model_resnet(input_tensor)
                    predicted_idx = torch.argmax(output, 1).item()
                    label = class_map.get(predicted_idx, "Unknown")
                    detected_objects.append(label)
                    
                    # Draw label
                    cv2.putText(output_image, label, (x, y-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                pass # Skip if crop fails
                
    return output_image, list(set(detected_objects))


def extract_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing for better OCR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Dilation to make text clearer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(thresh, config=custom_config).strip()

# ------------------------------------------------
# UI LAYOUT
# ------------------------------------------------
st.title("Warehouse Intelligence System")
st.write("Box Detection · Classification · Q&A System")

module = st.radio("Select Input Type", ["Image", "Video"])

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = []
if "context" not in st.session_state:
    st.session_state.context = ""

# -----------------
# IMAGE MODE
# -----------------
if module == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # 1. OCR First (to help classification)
        ocr_text = extract_ocr(image)
        ocr_lower = ocr_text.lower() if ocr_text else ""
        
        # 2. Process Image (Pass OCR context to improve labeling if needed)
        processed_img, objects = process_frame(image.copy())
        
        # 3. Intelligent Override (Rule-based correction)
        # If ML says "Standard" but OCR says "Fragile", trust OCR.
        final_objects = []
        for obj in objects:
            if obj == "Standard" and ("fragile" in ocr_lower or "handle with care" in ocr_lower):
                final_objects.append("Fragile")
            else:
                final_objects.append(obj)
        
        # Display
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Analyzed Image")
        
        # Update Context
        context_str = f"Detected Objects: {', '.join(final_objects)}. "
        if ocr_text:
            context_str += f"Detected Text: {ocr_text}."
        
        st.info(context_str)
        st.session_state.context = context_str
        st.session_state.detected_objects = final_objects

# -----------------
# VIDEO MODE
# -----------------
elif module == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        all_objects = set()
        all_text = ""
        
        frame_count = 0
        skip_frames = 5
        
        # Persistent storage for detections between skipped frames
        current_boxes = []
        current_labels = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run Detection & ML only every 'skip_frames'
            if frame_count % skip_frames == 0:
                # Resize for faster detection (optional, but good for CPU)
                frame_small = cv2.resize(frame, (960, 540))
                scale_x = frame.shape[1] / 960
                scale_y = frame.shape[0] / 540
                
                # Detect on small
                _, small_boxes = detect_boxes(frame_small)
                
                # Update persistent data
                current_boxes = []
                current_labels = []
                
                if small_boxes:
                    for (sx, sy, sw, sh) in small_boxes:
                        # Scale back to original
                        x = int(sx * scale_x)
                        y = int(sy * scale_y)
                        w = int(sw * scale_x)
                        h = int(sh * scale_y)
                        current_boxes.append((x, y, w, h))
                        
                        # Run Classification (on original crop for quality)
                        if model_resnet:
                            crop = frame[y:y+h, x:x+w]
                            if crop.size > 0:
                                try:
                                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                    input_tensor = transform(pil_crop).unsqueeze(0)
                                    with torch.no_grad():
                                        output = model_resnet(input_tensor)
                                        predicted_idx = torch.argmax(output, 1).item()
                                        label = class_map.get(predicted_idx, "Unknown")
                                        current_labels.append(label)
                                except:
                                    current_labels.append("Unknown")
            
            # Update accumulators
            all_objects.update(current_labels)
            
            # OCR Sampling (Image based)
            if frame_count % 30 == 0:
                text = extract_ocr(frame)
                if text:
                    all_text += " " + text
            
            # DRAWING (Runs on every frame using latest 'current_*')
            output_render = frame.copy()
            for i, (x, y, w, h) in enumerate(current_boxes):
                # Box
                cv2.rectangle(output_render, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Center
                cx, cy = x + w//2, y + h//2
                cv2.circle(output_render, (cx, cy), 5, (0, 0, 255), -1)
                # Label
                if i < len(current_labels):
                    label = current_labels[i]
                    text = f"{label} ({w}x{h})"
                    
                    # Draw label background for better visibility
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Ensure label stays on screen
                    label_y = y - 10 if y - 25 > 0 else y + h + 20
                    
                    cv2.rectangle(output_render, (x, label_y - text_h - 5), (x + text_w, label_y + 5), (0, 255, 0), -1)
                    
                    cv2.putText(output_render, text, (x, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            stframe.image(cv2.cvtColor(output_render, cv2.COLOR_BGR2RGB))
            
        cap.release()
        
        # Clarify context for Q&A
        unique_objects = list(all_objects)
        context_str = f"The user has uploaded a video. Detected objects: {', '.join(unique_objects)}. Extracted text: {all_text}."
        
        # Store for Q&A
        st.session_state.detected_objects = unique_objects
        
        # Only update context if it's new
        if st.session_state.context != context_str:
             st.session_state.context = context_str
             st.session_state.messages = [] 
             st.rerun() 
             
        st.success("Analysis Complete. Ask questions below.")

# -----------------
# Q&A INTERFACE
# -----------------
st.subheader("Interactive Q&A")
st.write("Ask questions about the uploaded media or general guidelines.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Smart Query Construction
    # Combine user question with detected labels to guide the semantic search
    # e.g. "How to handle?" + "Fragile" -> "How to handle? Fragile"
    context_tags = " ".join(st.session_state.detected_objects)
    enhanced_query = f"{prompt} {context_tags}"
    
    response = query_knowledge_base(enhanced_query)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)