import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return "Smart Assisted System (SAS) is Live"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files["image"])
    results = model(image)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        detections.append(results[0].names[cls])

    return jsonify({
        "objects_detected": detections
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
