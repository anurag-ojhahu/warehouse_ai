---
title: Warehouse AI
emoji: 📦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Warehouse Risk Detection System

AI-powered warehouse inspection system with:
- **Computer Vision**: Object detection, dimension estimation, and center tracking.
- **Machine Learning**: Image classification (Fragile, Heavy, Hazardous, etc.).
- **RAG (NLP)**: Retrieval-Augmented Generation for operational guidelines.

## Setup Instructions

1.  **Create Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Full Pipeline (Integration Demo)
Detects object in `section1_cv/box.jpg`, classifies it, and retrieves handling logic.
```bash
python main_pipeline.py
```
*Output will check for object, print class, and show handling instructions. An annotated image `detected_output.jpg` will be saved.*

### 2. Component Demos

**Computer Vision (Image)**
```bash
python section1_cv/image_detection.py
```

**Computer Vision (Video)**
```bash
python section1_cv/video_detection.py
```
*Press 'ESC' to exit video window.*

**Train Image Classifier**
To train a new model, organize data in `data/images/` with subfolders for each class, then run:
```bash
python section2_ml/train_image_model.py
```

**Semantic Search (RAG)**
```bash
python section3_nlp/semantic_search.py
```

### 3. Streamlit App (Web Interface)
```bash
streamlit run app.py
```

## Implementation Details

-   **CV**: Uses OpenCV for contour detection and bounding box approximation.
-   **ML**: ResNet18 (PyTorch) for robust image classification.
-   **NLP**: SentenceTransformers + FAISS for semantic search over technical guidelines.

## Challenges & Solutions
-   **Small object noise**: Implemented area filtering in CV to ignore non-relevant contours.
-   **Text matching**: Switched from simple keyword matching to Semantic Search (Embeddings) for better query understanding.# warehouse_ai
