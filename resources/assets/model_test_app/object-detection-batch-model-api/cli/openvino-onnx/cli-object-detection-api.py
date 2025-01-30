import os
import sys
import argparse
import requests
import numpy as np
import cv2
from typing import List, Dict

def preprocess_image(image_path: str) -> tuple:
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose((2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)
    return img_batched, img

def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
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

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def postprocess_predictions(output: np.ndarray, original_img: np.ndarray, 
                             class_names: List[str], confidence_threshold: float = 0.5) -> tuple:
    output = output[0]
    detections = []
    
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

def perform_inference(image_path: str, url: str, class_names: List[str]) -> List[Dict]:
    payload = {
        "inputs": [{
            "name": "images",
            "shape": [1, 3, 640, 640],
            "datatype": "FP32",
            "data": preprocess_image(image_path)[0].tolist()
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Inference request failed: {response.text}")
    
    result = response.json()
    output = np.array(result['outputs'][0]['data']).reshape(result['outputs'][0]['shape'])
    
    processed_image = cv2.imread(image_path)
    detections, annotated_img = postprocess_predictions(output, processed_image, class_names)
    
    return detections, annotated_img

def process_images_in_directory(image_directory: str, url: str, class_names: List[str]):
    os.makedirs('annotated_images', exist_ok=True)
    
    for root, _, files in os.walk(image_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, image_directory)
                output_path = os.path.join('annotated_images', relative_path)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    detections, annotated_img = perform_inference(image_path, url, class_names)
                    cv2.imwrite(output_path, annotated_img)
                    
                    print(f"Processed {image_path}")
                    for det in detections:
                        print(f"  {det['class_name']}: {det['confidence']:.2f}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument("-u", "--url", required=True, help="Inference endpoint URL")
    parser.add_argument("-c", "--classes", 
                        type=str, 
                        help="Comma-separated list of class names (e.g., hardhat,no-hardhat)")
    parser.add_argument("-p", "--path", 
                        default=os.getcwd(), 
                        help="Path to image directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Parse the comma-separated class names into a list
    class_names = args.classes.split(',') if args.classes else []
    
    process_images_in_directory(args.path, args.url, class_names)



if __name__ == "__main__":
    main()