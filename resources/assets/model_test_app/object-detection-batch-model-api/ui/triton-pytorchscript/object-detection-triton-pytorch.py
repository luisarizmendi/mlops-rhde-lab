import gradio as gr
import cv2
import numpy as np
import requests
import json
from PIL import Image

def preprocess_image(image):
    img = cv2.resize(np.array(image), (640, 640))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)

def non_max_suppression(detections, iou_threshold=0.5):
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    final_detections = []

    while detections:
        best = detections.pop(0)
        final_detections.append(best)

        detections = [
            det for det in detections
            if calculate_iou(best['bbox'], det['bbox']) < iou_threshold
        ]

    return final_detections

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def detect_objects_with_triton(url, files, class_names):
    if not files:
        return "No files uploaded.", []
    
    # Parse class names if provided
    class_names_list = [name.strip() for name in class_names.split(",")] if class_names else []
    
    results_images = []
    for file in files:
        try:
            image = Image.open(file).convert("RGB")
            input_data = preprocess_image(image)
            
            payload = {
                "inputs": [{
                    "name": "input__0",
                    "shape": input_data.shape,
                    "datatype": "FP32",
                    "data": input_data.flatten().tolist()
                }]
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, 
                                     headers=headers, 
                                     data=json.dumps(payload))
            
            if response.status_code != 200:
                return f"Error: {response.status_code} - {response.text}", []
            
            result = response.json()
            output = np.array(result['outputs'][0]['data']).reshape(result['outputs'][0]['shape'])
            
            processed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            detections = []
            confidence_threshold = 0.5
            
            for detection in output[0].T:
                confidence = detection[4]
                if confidence > confidence_threshold:
                    class_id = np.argmax(detection[5:])
                    x_center, y_center, width, height = detection[:4]

                    x = x_center - width/2
                    y = y_center - height/2

                    detections.append({
                        'class_name': class_names_list[class_id] if class_names_list and class_id < len(class_names_list) else f'class{class_id}',
                        'confidence': confidence,
                        'bbox': [x, y, width, height]
                    })

            filtered_detections = non_max_suppression(detections)

            h, w = processed_image.shape[:2]
            scale_x, scale_y = 640/w, 640/h

            for det in filtered_detections:
                x, y, width, height = det['bbox']
                x, width = x/scale_x, width/scale_x
                y, height = y/scale_y, height/scale_y

                cv2.rectangle(
                    processed_image,
                    (int(x), int(y)),
                    (int(x+width), int(y+height)),
                    (0, 255, 0),
                    2
                )

                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(
                    processed_image,
                    label,
                    (int(x), int(y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
            
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            results_images.append(processed_image_rgb)
        
        except Exception as e:
            return f"Error processing image: {str(e)}", []
    
    return "Processing completed.", results_images

interface = gr.Interface(
    fn=detect_objects_with_triton,
    inputs=[
        gr.Textbox(label="Triton Server Inference URL", value=""),
        gr.Files(file_types=["image"], label="Select Images"),
        gr.Textbox(label="Class Names (comma-separated)", value="")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Gallery(label="Results")
    ],
    title="YOLOv11 Object Detection with Triton",
    description="Upload multiple images for object detection."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8800)