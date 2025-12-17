import os
from flask import Flask
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return "Smart Assisted System (SAS) is running successfully"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
