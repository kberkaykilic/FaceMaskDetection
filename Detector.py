import cv2
import torch
from ultralytics import YOLO
from SSD.SSD import SSDLite
"""Please use Python 3.10 due to compatibility"""
"""My project is built on the YOLO model, whereas the SSD model shows poor performance in low-light conditions and performs well in well-lit environments."""
"""I couldn't make further improvements to the SSD model due to training time constraints, as each training session took approximately 5 hours."""

def run_yolo_model():
    model = YOLO(fr'YOLO/yolo_model.pt')
    categories = ['Incorrect Mask', 'Masked', 'No Mask']

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera Error")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        results = model(frame)

        for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box)
                label = categories[int(cls)]

                if label == 'No Mask':
                    box_color = (0, 0, 255)  # Kırmızı
                elif label == 'Masked':
                    box_color = (0, 255, 0)  # Yeşil
                elif label == 'Incorrect Mask':
                    box_color = (0, 255, 255)  # Sarı
                else:
                    box_color = (255, 255, 255)  # Varsayılan Beyaz

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()



def run_ssd_model(model_file='SSD/ssd_model.pth'):
    device = torch.device('cpu')
    model = SSDLite(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Unable to access the camera.")
        return

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Yazıyı eklemek için çerçeveye metin yerleştirme
        cv2.putText(frame, "Please try in a well-lit environment to see best performance.",
                    (10, 30),  # Sağ üst köşe için koordinatlar
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Sarı renk, kalınlık: 2

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if detected_faces is not None and len(detected_faces) > 0:
            detected_faces = sorted(detected_faces, key=lambda face: face[2] * face[3], reverse=True)
            x, y, w, h = detected_faces[0]
            face_region = frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_region, (320, 320))
            normalized_face = resized_face / 255.0
            face_tensor = torch.FloatTensor(normalized_face).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                location_predictions, confidence_predictions = model(face_tensor)

            location_predictions = location_predictions.view(-1, 4)
            confidence_predictions = confidence_predictions.view(-1, 3)

            highest_confidence = 0
            best_bounding_box = None
            predicted_label = None

            for location, confidence in zip(location_predictions, confidence_predictions):
                confidence_value = torch.max(confidence).item()
                if confidence_value > 0.7 and confidence_value > highest_confidence:
                    highest_confidence = confidence_value
                    best_bounding_box = location.tolist()
                    predicted_label = torch.argmax(confidence).item()

            if best_bounding_box:
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h

                mask_label = "Masked" if predicted_label == 0 else "No Mask"
                box_color = (0, 255, 0) if predicted_label == 0 else (0, 0, 255)
                text_color = (255, 255, 255)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(frame, f"{mask_label} ({highest_confidence:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        cv2.imshow("SSD Model Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



def main():
    choice = input("Select Model:\n1: YOLO\n2: SSD\nEnter your choice: ")
    if choice == '1':
        run_yolo_model()
    elif choice == '2':
        run_ssd_model()
        print("Please try well-lit environment to see best performance.")
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
