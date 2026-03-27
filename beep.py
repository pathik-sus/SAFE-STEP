import cv2
import numpy as np
import pygame
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Target objects
target_objects = ["person", "chair", "bed", "dining table", "car"]

# Initialize sound
pygame.mixer.init(frequency=44100, size=-16, channels=2)

def play_beep(side="left", count=1):

    freq = 900
    duration = 0.15
    sample_rate = 44100

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(freq * 2 * np.pi * t)
    tone = (tone * 32767).astype(np.int16)

    if side == "left":
        stereo = np.column_stack((tone, np.zeros_like(tone)))
    else:
        stereo = np.column_stack((np.zeros_like(tone), tone))

    sound = pygame.sndarray.make_sound(stereo)

    for i in range(count):
        sound.play()
        time.sleep(0.3)


cap = cv2.VideoCapture(0)

# timer variable
last_alert = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    h, w, _ = frame.shape

    for r in results:

        for box in r.boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label not in target_objects:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1
            area = width * height

            center_x = (x1 + x2) // 2

            # Detect side
            if center_x < w / 2:
                side = "left"
            else:
                side = "right"

            # Distance estimation
            if area < 15000:
                beep_count = 1
            elif area < 40000:
                beep_count = 2
            else:
                beep_count = 3

            # 5 SECOND DELAY
            if time.time() - last_alert > 5:
                play_beep(side, beep_count)
                last_alert = time.time()

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

    cv2.imshow("Assistive Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()