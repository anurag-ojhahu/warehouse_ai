
# Warehouse AI System - Codebase Explanation

This document explains the purpose and functionality of every file in the project. Use this guide to present the system to your team or stakeholders.

## 1. Core Application
### `app.py`
**Purpose**: The main entry point for the **Web Interface**. It runs the Streamlit application that users interact with.
-   **Key Features**:
    -   **Unified Input**: Handles both Image and Video uploads.
    -   **Model Loading**: Loads the pre-trained PyTorch ResNet model (`section2_ml/resnet_classifier.pth`).
    -   **Pipeline Integration**: Calls `detect_boxes` (CV), extracts OCR text, and runs classification.
    -   **Smart Logic**: Overrides ML classification if OCR detects keywords (e.g., "Fragile").
    -   **Interactive Q&A**: Maintains a chat session where users can ask questions. It constructs smart queries by appending detected object tags to the user's question before sending it to the RAG system.

### `main_pipeline.py`
**Purpose**: A command-line script to demonstrate the **end-to-end pipeline** without the web UI.
-   **Workflow**:
    1.  Loads an image from `section1_cv/box.jpg`.
    2.  Runs object detection to find the box.
    3.  Classifies the box using the ML model.
    4.  Queries the RAG system for handling instructions.
    5.  Saves the result as `detected_output.jpg`.
-   **Use Case**: Quick verification of the backend logic or for batch processing.

---

## 2. Computer Vision (CV) - `section1_cv/`
### `image_detection.py`
**Purpose**: Contains the core logic for **detecting objects** in images.
-   **`detect_boxes(image)`**:
    -   Converts image to grayscale and blurs it to reduce noise.
    -   Uses Canny Edge Detection to find edges.
    -   Finds contours (shapes) and filters them based on area (to ignore small noise) and aspect ratio (to find box-like shapes).
    -   Draws bounding boxes and calculates the center point and dimensions (Width x Height).
    -   **Returns**: The processed image with drawings and a list of bounding box coordinates.

### `video_detection.py`
**Purpose**: A script to test object detection on **video files**. (Note: The main video logic is now also integrated into `app.py`).

---

## 3. Machine Learning (ML) - `section2_ml/`
### `train_image_model.py`
**Purpose**: The script used to **train the Neural Network**.
-   **Architecture**: Uses **ResNet18**, a powerful pre-trained Convolutional Neural Network (CNN).
-   **Process**:
    -   Loads images from a dataset structure (`data/items/`).
    -   Applies transformations (augmentation) like rotation and flipping to make the model robust.
    -   Trains the model to classify images into 3 categories: **Fragile, Heavy, Standard**.
    -   Saves the trained weights to `resnet_classifier.pth`.

### `resnet_classifier.pth`
**Purpose**: The **saved brain** of the AI. This binary file contains the learned patterns (weights) that allow the system to recognize differences between a "Fragile" box and a "Heavy" box.

---

## 4. Natural Language Processing (NLP) - `section3_nlp/`
### `semantic_search.py`
**Purpose**: The **Retrieval-Augmented Generation (RAG)** engine. This is the "brain" that knows the safety rules.
-   **Knowledge Base**: Contains a list of handling rules (e.g., "Fragile packages must be handled with reduced gripper...").
-   **`create_embeddings()`**: Converts these text rules into numerical vectors using a Sentence Transformer model.
-   **`query_knowledge_base(query)`**:
    -   Takes a user question (e.g., "How to handle? Fragile").
    -   Converts it to a vector.
    -   Finds the most similar rule in the knowledge base using FAISS (Facebook AI Similarity Search).
    -   Returns the best matching rule as the answer.

### `debug_nlp.py`
**Purpose**: A temporary utility script to test and debug the RAG retrieval accuracy.

---

## 5. Configuration & Documentation
### `requirements.txt`
**Purpose**: Lists all the **python libraries** needed to run the project (e.g., `streamlit`, `opencv`, `torch`, `sentence-transformers`).

### `README.md`
**Purpose**: The main specific documentation for this project. It contains setup instructions, how to run the app, and an overview of the features.
