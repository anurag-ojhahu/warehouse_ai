import cv2
import numpy as np


def detect_boxes(image):
    """
    Detect rectangular box-like objects using classical CV.
    Returns image with bounding boxes drawn.
    """

    # 1️⃣ Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3️⃣ Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # 4️⃣ Morphological closing (fills small gaps in edges)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5️⃣ Find contours
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore very small noise only
        if area < 50:
            continue

        # Get bounding rect
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Filter unrealistic shapes (e.g. extremely thin lines)
        if 0.5 < aspect_ratio < 4.0:

            detected_boxes.append((x, y, w, h))

            # Calculate center
            cx = x + w // 2
            cy = y + h // 2

            # Draw rectangle
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw center
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

            # Draw dimensions and center text
            label = f"{w}x{h} ({cx},{cy})"
            cv2.putText(
                image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    return image, detected_boxes


if __name__ == "__main__":

    image_path = "section1_cv/box.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found.")
        exit()

    output, boxes = detect_boxes(image)

    print(f"Detected {len(boxes)} box(es)")

    cv2.imshow("Detected Boxes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()