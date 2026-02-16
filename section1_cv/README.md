# Section 1 – Classical Computer Vision Detection

## Objective
Detect warehouse boxes from both static images and video streams using classical computer vision techniques.

---

## Methodology

The detection pipeline consists of:

1. Grayscale conversion  
2. Canny edge detection  
3. Contour extraction  
4. Area-based filtering  
5. Bounding box visualization  

---

## Detection Logic

- Edges are extracted using Canny with thresholds (50, 150).
- External contours are retrieved.
- Contours with area greater than:
    - 1000 (image)
    - 5000 (video)
  are considered valid object candidates.
- Bounding rectangles are drawn around detected regions.

---

## Files

- `image_detection.py` – Static image detection
- `video_detection.py` – Frame-by-frame video detection
- `box.jpg` – Sample test image
- `warehouse_video.mp4` – Test video

---

## How to Run

Activate environment:

source venv/bin/activate

Run image detection:

python image_detection.py

Run video detection:

python video_detection.py

Press `q` to quit video window.

---

## Limitations

- Sensitive to lighting variations
- No object classification
- Purely contour-based (not ML-based)
- May detect non-box rectangular objects

---

## Future Improvements

- Add contour shape filtering (aspect ratio constraints)
- Use Hough transform for stronger edge validation
- Integrate deep learning detector (YOLO / Faster R-CNN)