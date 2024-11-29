from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime, timedelta
import os
import base64
import torch
import threading
import queue
import time
from collections import defaultdict
import random

app = Flask(__name__)

model_path = os.getenv('YOLO_MODEL_PATH', '.')
model_file= os.getenv('YOLO_MODEL_FILE', 'object-detector-hardhat-v1.pt')
model_threshold= float(os.getenv('YOLO_MODEL_THRESHOLD', '0.25'))
model = YOLO(f"{model_path}/{model_file}")

cap = None
selected_camera_index = None
confidence_thresholds = {}
class_colors = {}
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

current_detections_lock = threading.Lock()
current_detections = defaultdict(lambda: {"max_confidence": 0.0, "min_confidence": 0.0, "last_seen": datetime.min, "count": 0})

if torch.cuda.is_available():
    model.to('cuda')
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

def get_color_for_class(class_name):
    """
    Generate and return a unique color for the given class if it doesn't already exist.
    """
    if class_name not in class_colors:
        class_colors[class_name] = (
            random.randint(0, 255),  #  R
            random.randint(0, 255),  #  G
            random.randint(0, 255)   #  B
        )
    return class_colors[class_name]

def process_frame(frame, conf_dict):
    """
    Process a single frame and return detections
    """
    global current_detections
    detections_this_frame = defaultdict(lambda: {"max_confidence": 0.0, "min_confidence": float('inf'), "last_seen": datetime.min, "count": 0})

    results = model(frame)[0]

    for detection in results.boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        class_name = results.names[int(cls)]
        color = get_color_for_class(class_name)

        threshold = conf_dict.get(class_name, model_threshold )  

        if conf >= threshold:
            # Store the highest confidence for each object class
            detections_this_frame[class_name]["max_confidence"] = max(detections_this_frame[class_name]["max_confidence"], float(conf))
            detections_this_frame[class_name]["min_confidence"] = min(detections_this_frame[class_name]["min_confidence"], float(conf))
            detections_this_frame[class_name]["last_seen"] = datetime.now()
            detections_this_frame[class_name]["count"] += 1

            # box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # rectangle behind the text
            label = f"{class_name}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_background_start = (int(x1), int(y1) - text_height - baseline)
            text_background_end = (int(x1) + text_width, int(y1))
            cv2.rectangle(frame, text_background_start, text_background_end, color, -1)

            # white on dark colors, black on bright colors
            brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Add the class name text
            cv2.putText(frame, label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    with current_detections_lock:
        current_detections = {
            class_name: {
                "max_confidence": details["max_confidence"],
                "min_confidence": details["min_confidence"],
                "count": details["count"]
            }
            for class_name, details in detections_this_frame.items()
        }

    return frame, current_detections


def initialize_camera():
    """
    Initialize the camera
    """
    global cap, selected_camera_index
    if selected_camera_index is not None:
        print(f"Camera {selected_camera_index} already initialized. Using it for inference.")
        return True  # Camera already initialized

    camera_index = int(os.getenv('CAMERA_INDEX', -1))  # Default to -1 if not set
    print("Selecting Camera")

    if camera_index != -1:
        print(f"Testing Camera index {camera_index}")
        cap = cv2.VideoCapture(camera_index)

    if not cap or not cap.isOpened():
        print(f"Camera index {camera_index} not working, looking for any available camera...")

        available_cameras = []
        max_resolution = (0, 0)

        # Find the best available camera by resolution
        for camera_index in range(10):  # Adjust range if necessary
            temp_cap = cv2.VideoCapture(camera_index)

            if temp_cap.isOpened():
                print(f"Camera {camera_index} is available.")
                width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera {camera_index} resolution: {width}x{height}")

                if (width * height) > (max_resolution[0] * max_resolution[1]):
                    max_resolution = (width, height)
                    selected_camera_index = camera_index

                available_cameras.append(camera_index)
                temp_cap.release()

        if selected_camera_index is not None:
            cap = cv2.VideoCapture(selected_camera_index)
            print(f"Selected camera: {selected_camera_index} with resolution {max_resolution[0]}x{max_resolution[1]}")
        else:
            print("No available cameras found.")
            cap = None

    print(f"Camera index {selected_camera_index} will be used for inference")

    if not cap or not cap.isOpened():
        print("Unable to access any camera")
        return False

    return True

def continuous_inference_thread():
    """
    Continuously capture frames and process them.
    """
    global cap
    if not cap or not cap.isOpened():
        print("Camera not initialized.")
        return

    try:
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)  # Delay
                continue

            processed_frame, _ = process_frame(frame, confidence_thresholds)

            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

            frame_queue.put(processed_frame)
    finally:
        cap.release()

@app.route('/detect_image', methods=['POST'])
def detect_batch():
    """
    Process a batch of images
    """
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    files = request.files.getlist('images')

    for file in files:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_image, counts = process_frame(image, confidence_thresholds)

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.splitext(file.filename)[0]
        processed_filename = f"{original_filename}_processed_{timestamp}.jpg"

        results.append({
            "filename": file.filename,
            "processed_filename": processed_filename,
            "object_counts": counts,
            "image_base64": img_base64
        })

    return jsonify({"results": results})

def generate_frames():
    """
    Generator function for streaming webcam
    """
    while True:
        frame = frame_queue.get()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        with current_detections_lock:
            current_detections_serializable = {
                class_name: {
                    "max_confidence": details["max_confidence"],
                    "min_confidence": details["min_confidence"],
                    "count": details["count"]
                }
                for class_name, details in current_detections.items()
            }

        counts_json = json.dumps(current_detections_serializable)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
               b'Content-Type: application/json\r\n\r\n' + counts_json.encode() + b'\r\n')


@app.route('/video_stream')
def video_stream():
    """
    Route for webcam streaming
    """
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_detections', methods=['GET'])
def current_detections_endpoint():
    """
    Return the current object counts with confidence scores
    """
    with current_detections_lock:
        formatted_detections = current_detections.copy()

    return jsonify(formatted_detections)

inference_thread = None

def start_inference():
    """
    Start the continuous inference thread
    """
    global inference_thread
    stop_event.clear()
    inference_thread = threading.Thread(target=continuous_inference_thread, daemon=True)
    inference_thread.start()

def stop_inference():
    """
    Stop the continuous inference thread
    """
    global inference_thread
    if inference_thread:
        stop_event.set()
        inference_thread.join()

if __name__ == '__main__':
    initialize_camera()
    try:
        start_inference()
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        stop_inference()
