import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera working!")
else:
    print("❌ Camera not detected.")
cap.release()
