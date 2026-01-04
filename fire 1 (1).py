from ultralytics import YOLO
import cvzone
import cv2
import math

# Video input (FIXED)
cap = cv2.VideoCapture(
    r'C:\Users\GOKUL\Desktop\project1\Fire Detector Course\fire4.mp4'
)

# Load YOLO model
model = YOLO(
    r'C:\Users\GOKUL\Desktop\project1\Fire Detector Course\fire1.pt'
)

# Class names
classnames = ['fire']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)

    for info in results:
        for box in info.boxes:
            confidence = int(float(box.conf[0]) * 100)
            cls = int(box.cls[0])

            if confidence > 50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(
                    frame,
                    f'{classnames[cls]} {confidence}%',
                    (x1, y1 - 10),
                    scale=1.2,
                    thickness=2
                )

    cv2.imshow('Fire Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
