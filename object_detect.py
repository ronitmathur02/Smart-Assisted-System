# Smart Assist System (Improved Version)
# python object_detect.py
from ultralytics import YOLO
import cv2
import pyttsx3
import time


# Load YOLOv8 model (lightweight version)
model = YOLO('yolov8n.pt')

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize offline TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 165)
engine.setProperty('volume', 1.0)

# Dictionary to store last time each object was spoken
last_spoken = {}
cooldown = 5  # seconds to wait before repeating the same object

print("\n🔹 Smart Assist System Started")
print("Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from camera.")
        break

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Get detected object names
    names = [model.names[int(box.cls[0])] for box in results[0].boxes]

    current_time = time.time()
    for name in names:
        # Speak only if cooldown time has passed
        if name not in last_spoken or (current_time - last_spoken[name] > cooldown):
            print(f"Detected: {name}")
            engine.say(f"{name} detected")
            engine.runAndWait()
            last_spoken[name] = current_time

    # Display camera feed with YOLO bounding boxes
    cv2.imshow("Smart Assist - YOLOv8 Object Detection", annotated_frame)

    # Quit with 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 Program stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
