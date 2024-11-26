from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import os
import base64
import torch
import subprocess

app = Flask(__name__)

model_path = os.getenv('YOLO_MODEL_PATH', '../../../models/luisarizmendi/object-detector-safety/object-detector-safety-v1.pt')
model = YOLO(model_path)

confidence_thresholds = {}

if torch.cuda.is_available():
    model.to('cuda') 
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

current_counts = {}

def process_frame(frame, conf_dict):
    """
    Process a single frame and return detections
    """
    global current_counts
    results = model(frame)[0]
    
    object_counts = {}
    
    for detection in results.boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        class_name = results.names[int(cls)]
        
        threshold = conf_dict.get(class_name, 0.25)  # Default threshold is 0.25
        
        if conf >= threshold:
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
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

    current_counts = object_counts
    
    return frame, object_counts

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
            exit -1
       
        cap = cv2.VideoCapture(selected_camera_index)
           
    if not cap or not cap.isOpened():
        raise RuntimeError("Unable to access any camera")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        processed_frame, counts = process_frame(frame, confidence_thresholds)
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        counts_json = json.dumps(counts)
  
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
               b'Content-Type: application/json\r\n\r\n' + counts_json.encode() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """
    Route for webcam streaming
    """
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_counts', methods=['GET'])
def current_counts_endpoint():
    """
    Return the current object counts
    """
    return jsonify(current_counts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
