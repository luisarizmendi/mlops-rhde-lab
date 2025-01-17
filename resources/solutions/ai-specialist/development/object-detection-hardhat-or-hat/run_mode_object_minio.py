import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import torch
from minio import Minio
from minio.error import S3Error

# Environment variables for Minio setup
AWS_S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
MODEL_KEY = os.getenv("MODEL_KEY")

# Initialize Minio client
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True
)

def download_model_from_s3(bucket_name, model_key):
    """
    Downloads a model from Minio object storage.
    """
    model_path = "/tmp/model.pt"
    try:
        client.fget_object(bucket_name, model_key, model_path)
        print(f"Model downloaded successfully to {model_path}")
    except S3Error as e:
        print(f"Error occurred while downloading the model: {e}")
        return None
    return model_path

def detect_objects_in_files(model_input, files):
    """
    Processes uploaded images for object detection.
    """
    if not files:
        return "No files uploaded.", []

    # If model path doesn't exist, download it from Minio
    model_path = model_input
    if not os.path.exists(model_path):
        if AWS_S3_BUCKET and MODEL_KEY:
            print("Downloading model from Minio...")
            model_path = download_model_from_s3(AWS_S3_BUCKET, MODEL_KEY)
        else:
            return "Model key or object storage credentials are missing.", []

    # Load the model
    model = YOLO(str(model_path))
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

        except Exception as e:
            return f"Error processing file: {file}. Exception: {str(e)}", []

    del model
    torch.cuda.empty_cache()

    return "Processing completed.", results_images

interface = gr.Interface(
    fn=detect_objects_in_files,
    inputs=[
        gr.Textbox(value="", label="Model Path", placeholder="Leave empty to fetch model from object storage"),
        gr.Files(file_types=["image"], label="Select Images"),
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Gallery(label="Results")
    ],
    title="Object Detection on Images",
    description="Upload images to perform object detection. The model will process each image and display the results."
)

if __name__ == "__main__":
    interface.launch()
