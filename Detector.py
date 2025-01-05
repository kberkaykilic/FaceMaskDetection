import cv2
from ultralytics import YOLO
"""Please use Python 3.10 due to compatibility"""

model = YOLO(fr'train\weights\best.pt')

CLASS_NAMES = ['Incorrectly Weared Mask', 'With Mask', 'Without Mask']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    iou_threshold = 0.4
    conf_threshold = 0.5

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold=conf_threshold, nms_threshold=iou_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            conf = confidences[i]
            cls_id = class_ids[i]

            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "Unknown"

            color = (0, 0, 255) if class_name == "Without Mask" else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('YOLOv8 Object Detection', frame)

    try:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        print("Error: OpenCV is unable to display windows. Ensure OpenCV GUI backend is installed.")
        break

cap.release()
cv2.destroyAllWindows()
