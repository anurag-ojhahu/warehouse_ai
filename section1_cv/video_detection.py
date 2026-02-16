import cv2
from image_detection import detect_boxes

cap = cv2.VideoCapture("section1_cv/warehouse_video.mp4")

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    output, boxes = detect_boxes(frame)

    cv2.imshow("Video Box Detection", output)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()