import cv2
import time
import pyttsx3
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Target objects
target_objects = ["person", "chair", "car", "dining table","couch","bus","motorcycle","bench","bed","refrigerator","stairs"]

# Reliable speak function 
def speak(text):
    engine = pyttsx3.init()   # reinitialize every time
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# Start camera
cap = cv2.VideoCapture(0)

last_detection_time = 0
DELAY = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run detection every 5 seconds
    if current_time - last_detection_time > DELAY:

        results = model(frame)
        h, w, _ = frame.shape

        detected_objects = []

        for r in results:
            for box in r.boxes:

                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label not in target_objects:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                area = (x2 - x1) * (y2 - y1)

                # Distance estimation
                if area < 15000:
                    distance = "far"
                elif area < 40000:
                    distance = "at medium distance"
                else:
                    distance = "very close"

                # Direction detection
                center_x = (x1 + x2) // 2

                if center_x < w / 3:
                    direction = "on your left"
                elif center_x > 2 * w / 3:
                    direction = "on your right"
                else:
                    direction = "in front of you"

                detected_objects.append(f"{label} {distance} {direction}")

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Speak all objects in ONE sentence
        if detected_objects:
            message = ", ".join(detected_objects[:3])  # limit to 3 objects
            print(message)
            speak(message)
        else:
            print("No important object detected")

        last_detection_time = current_time

    # Show camera feed
    cv2.imshow("Assistive Vision", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()