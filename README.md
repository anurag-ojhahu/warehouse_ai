# Warehouse AI System

An end-to-end AI pipeline for warehouse automation including:

• Object Detection (Computer Vision)  
• Package Classification (Transfer Learning with ResNet18)  
• Semantic Knowledge Retrieval (Sentence Transformers + FAISS)  
• Integrated Decision Pipeline  

---

## System Architecture

Image/Video Input  
        ↓  
Section 1 – Object Detection (OpenCV)  
        ↓  
Section 2 – Package Classification (ResNet18)  
        ↓  
Section 3 – Semantic Knowledge Retrieval  
        ↓  
Automated Handling Recommendation  

---

## Sections

### Section 1 – Computer Vision
Detects packages from images and video streams using contour-based detection.

### Section 2 – Machine Learning
Fine-tuned ResNet18 on selected CIFAR-10 classes using transfer learning.

### Section 3 – NLP / Semantic Search
Uses sentence-transformers (all-MiniLM-L6-v2) and FAISS for warehouse knowledge retrieval.

---

## Run the Full Pipeline

```bash
python main_pipeline.py
