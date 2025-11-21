# Smart Assist System - Detects new objects + speaks name + distance instantly
# python smart_assist_distance.py

from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# ---------------- Speech engine ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

def speak(text):
    """Runs voice in background so loop never freezes"""
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()

# ---------------- Variables ----------------
last_seen_objects = {}       # {object_name: last_distance}
DISTANCE_THRESHOLD = 5       # speak if distance changes by 5 cm
KNOWN_DISTANCE = 30          # cm calibration
KNOWN_WIDTH = 7              # cm width of calibration object
base_width = None

print("\n🔹 Smart Assist Started")
print("👉 Place a bottle (or object ~7 cm wide) around 30 cm from webcam...\n")

# ---------------- Calibration ----------------
while base_width is None:
    ret, frame = cap.read()
    results = model(frame, conf=0.35, iou=0.4, verbose=False)

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        base_width = x2 - x1
        print(f"✅ Calibrated base width = {base_width}px at 30 cm\n")
        time.sleep(1)

def estimate_distance(width_px):
    if width_px == 0 or base_width is None:
        return None
    distance = (base_width / width_px) * KNOWN_DISTANCE
    return round(distance, 1)

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, iou=0.4, verbose=False)
    annotated = results[0].plot()
    current_objects = {}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        distance = estimate_distance(width)
        if distance is None:
            continue

        current_objects[name] = distance

        cv2.putText(annotated, f"{name} {distance}cm",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # ------- SPEAK LOGIC -------
        if (name not in last_seen_objects):  
            # NEW OBJECT APPEARED
            speak(f"{name} detected at {distance} centimeters")

        elif abs(distance - last_seen_objects[name]) >= DISTANCE_THRESHOLD:
            # SAME OBJECT BUT CLOSER/FARTHER
            speak(f"{name} now at {distance} centimeters")

    # Update last seen objects
    last_seen_objects = current_objects.copy()

    cv2.imshow("Smart Assist - Object + Distance", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
