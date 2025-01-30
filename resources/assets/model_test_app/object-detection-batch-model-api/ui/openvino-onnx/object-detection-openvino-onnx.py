import gradio as gr
import requests
import os
import cv2
import numpy as np

def load_model(model_url):
    return model_url

def detect_objects(model_url, image_paths, class_names):
    if not image_paths:
        return "No images uploaded.", []

    # Parse class names into a list
    class_names_list = [name.strip() for name in class_names.split(",")]

    results_images = []
    for image_path in image_paths:
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            payload = {
                "inputs": [
                    {
                        "name": "images",
                        "shape": [1, 3, 640, 640],
                        "datatype": "FP32",
                        "data": preprocess_image(image_data)
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(model_url, json=payload, headers=headers)

            if response.status_code != 200:
                return f"Error processing file: {os.path.basename(image_path)}. Exception: {response.text}", []

            result = response.json()
            output = np.array(result['outputs'][0]['data']).reshape(result['outputs'][0]['shape'])
            processed_image = cv2.imread(image_path)
            detections, annotated_img = postprocess_predictions(output, processed_image, class_names_list)

            # Convert annotated image to RGB
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            results_images.append(annotated_img_rgb)

        except Exception as e:
            return f"Error processing file: {os.path.basename(image_path)}. Exception: {str(e)}", []

    return "Processing completed.", results_images


def preprocess_image(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose((2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)
    return img_batched.tolist()

def postprocess_predictions(output, original_img, class_names, confidence_threshold=0.5):
    print("First detection full details:", output[0].T[0])
    
    output = output[0]
    detections = []

    for detection in output.T:
        x_center, y_center, width, height = detection[:4]
        confidence = detection[4]
        class_probs = detection[5:]
        
        if confidence > confidence_threshold:
            print("DETECT: ", detection)

            class_id = np.argmax(class_probs)
            class_name = class_names[class_id] if class_id < len(class_names) else f'class{class_id}'
            
            print(f"Detection: Class {class_id} ({class_name}), Confidence: {confidence}")
  


    for detection in output.T:
        confidence = detection[4]
        if confidence > confidence_threshold:
            class_id = np.argmax(detection[5:])
            x_center, y_center, width, height = detection[:4]

            x = x_center - width/2
            y = y_center - height/2

            detections.append({
                'class_name': class_names[class_id] if class_id < len(class_names) else f'class{class_id}',
                'confidence': confidence,
                'bbox': [x, y, width, height]
            })

    filtered_detections = non_max_suppression(detections)

    h, w = original_img.shape[:2]
    scale_x, scale_y = 640/w, 640/h

    for det in filtered_detections:
        x, y, width, height = det['bbox']
        x, width = x/scale_x, width/scale_x
        y, height = y/scale_y, height/scale_y

        cv2.rectangle(
            original_img,
            (int(x), int(y)),
            (int(x+width), int(y+height)),
            (0, 255, 0),
            2
        )

        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(
            original_img,
            label,
            (int(x), int(y-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    return filtered_detections, original_img

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

interface = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Text(label="Inference Endpoint URL"),
        gr.Files(file_types=["image"], label="Select Images"),
        gr.Textbox(label="Class Names (comma-separated)")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Gallery(label="Results")
    ],
    title="Object Detection",
    description="Upload images to perform object detection using the provided API endpoint."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8800)
