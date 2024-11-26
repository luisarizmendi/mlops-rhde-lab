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

app = Flask(__name__)

model_path = os.getenv('YOLO_MODEL_PATH', '../../../models/luisarizmendi/object-detector-safety/object-detector-safety-v1.pt')
model = YOLO(model_path)

confidence_thresholds = {}

# Create a thread-safe queue to store processed frames
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

current_detections = defaultdict(lambda: {"max_confidence": 0.0, "min_confidence": 0.0, "last_seen": datetime.min, "count": 0})

if torch.cuda.is_available():
    model.to('cuda') 
    print("Using GPU for inference")
else:
    print("Using CPU for inference")


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
        
        threshold = conf_dict.get(class_name, 0.6)  # Default threshold is 0.6
        
        if conf >= threshold:
            # Store the highest confidence for each object class
            detections_this_frame[class_name]["max_confidence"] = max(detections_this_frame[class_name]["max_confidence"], float(conf))
            detections_this_frame[class_name]["min_confidence"] = min(detections_this_frame[class_name]["min_confidence"], float(conf))
            detections_this_frame[class_name]["last_seen"] = datetime.now()
            detections_this_frame[class_name]["count"] += 1

            cv2.rectangle(frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 
                        2)
            
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, 
                       label, 
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       2)
    
    # Update the current detections, removing old ones (no detections in the last second)
    now = datetime.now()
    for class_name, detection in list(current_detections.items()):
        if (now - detection["last_seen"]).total_seconds() > 1:  
            del current_detections[class_name]  
    
    # Merge new detections with current detections
    for class_name, detection in detections_this_frame.items():
        current_detections[class_name] = detection

    return frame, {
        class_name: {
            "max_confidence": details["max_confidence"],
            "min_confidence": details["min_confidence"],
            "count": details["count"] 
        }
        for class_name, details in current_detections.items()
    }
    
    
def continuous_inference_thread():
    """
    Continuously capture frames and process them
    """
    camera_index = int(os.getenv('CAMERA_INDEX', -1))  # Default to -1 if not set
    
    cap = None
    if camera_index != -1:
        # Try to open the camera at the specified index
        cap = cv2.VideoCapture(camera_index)
    
    if not cap or not cap.isOpened():
        print(f"Camera index {camera_index} not working, looking for any available camera...")
        
        available_cameras = []
        max_resolution = (0, 0)
        selected_camera_index = None

        for camera_index in range(10):  # Adjust range if necessary
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                print(f"Camera {camera_index} is available.")
                
                # Check the resolution of the current camera
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera {camera_index} resolution: {width}x{height}")
                
                # Check if this camera has the highest resolution so far
                if (width * height) > (max_resolution[0] * max_resolution[1]):
                    max_resolution = (width, height)
                    selected_camera_index = camera_index
                    
                available_cameras.append(camera_index)           
        
        if available_cameras:
            print(f"Available camera indices: {available_cameras}")
            print(f"Selected camera: {selected_camera_index} with resolution {max_resolution[0]}x{max_resolution[1]}")
        else:
            print("No available cameras found.")
            return
       
        cap = cv2.VideoCapture(selected_camera_index)
           
    if not cap or not cap.isOpened():
        print("Unable to access any camera")
        return

    try:
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)  #  delay
                continue
            
            processed_frame, _ = process_frame(frame, confidence_thresholds)
            
            # Clear the queue if it's full to ensure we always have the latest frame
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Put the processed frame in the queue
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

        current_detections_serializable = {
            class_name: {
                "max_confidence": details["max_confidence"],
                "min_confidence": details["min_confidence"],
                "last_seen": details["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
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
    formatted_detections = {
        class_name: {
            "max_confidence": details["max_confidence"],
            "min_confidence": details["min_confidence"],
            "count": details["count"]
        } for class_name, details in current_detections.items()
    }
    
    return jsonify(formatted_detections)

# Continuous inference thread
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

start_inference()

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        stop_inference()
