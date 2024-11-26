import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os
import cv2 
import torch 




def detect_objects_in_files(files):
    """
    Processes uploaded images for object detection.
    """
    if not files:
        return "No files uploaded.", []

    model = YOLO("runs/detect/train11/weights/best.pt")
    
    if torch.cuda.is_available():
        model.to('cuda') 
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    
    results_images = []
    for file in files:
        try:
            image = Image.open(file).convert("RGB")
            results = model(image) 
            result_img_bgr = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
            results_images.append(result_img_rgb)   
         
            # If you want that images appear one by one (slower)
            #yield "Processing image...", results_images  
                
        except Exception as e:
            return f"Error processing file: {file}. Exception: {str(e)}", []

    del model  
    torch.cuda.empty_cache()
    
    return "Processing completed.", results_images

interface = gr.Interface(
    fn=detect_objects_in_files,
    inputs=gr.Files(file_types=["image"], label="Select Images"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Gallery(label="Results")
    ],
    title="Object Detection on Images",
    description="Upload images to perform object detection. The model will process each image and display the results."
)

if __name__ == "__main__":
    interface.launch()
