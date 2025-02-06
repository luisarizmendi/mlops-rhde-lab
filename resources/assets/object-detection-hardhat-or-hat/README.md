
# Model for detecting Hardhats and Hats


<div align="center">
  <img width="640" alt="luisarizmendi/hardhat-or-hat" src="example.png">
</div>

## Model binary

You can [download the model from here](https://github.com/luisarizmendi/workshop-object-detection-rhde/raw/refs/heads/main/resources/assets/object-detection-hardhat-or-hat/v2/model/pytorch/best.pt)


## Base Model

Ultralytics/YOLO11m


## Huggingface page

https://huggingface.co/luisarizmendi/hardhat-or-hat


## Model Dataset

[https://universe.roboflow.com/luisarizmendi/hardhat-or-hat](https://universe.roboflow.com/luisarizmendi/hardhat-or-hat)

## Labels

```
- hat
- helmet
- no_helmet
```


## Model metrics

<div align="center">
  <img width="640" alt="luisarizmendi/hardhat-or-hat" src="v2/train-val/results.png"> 
</div>



<div align="center">
  <img width="640" alt="luisarizmendi/hardhat-or-hat" src="v2/train-val/confusion_matrix_normalized.png"> 
</div>



## Model training

You can [review the Jupyter notebook here](https://github.com/luisarizmendi/workshop-object-detection-rhde/blob/main/resources/solutions/ai-specialist/development/prototyping.ipynb)

You can [review the Kubeflow Pipeline here](https://github.com/luisarizmendi/workshop-object-detection-rhde/blob/main/resources/solutions/ai-specialist/training/kubeflow/hardhat-kubeflow-pipeline.py)



### Hyperparameters

```
base model: yolov11x.pt
epochs: 150
batch: 16
imgsz: 640
patience: 15
optimizer: 'SGD'
lr0: 0.001
lrf: 0.01
momentum: 0.9
weight_decay: 0.0005
warmup_epochs: 3
warmup_bias_lr: 0.01
warmup_momentum: 0.8
```


## Model Usage

### Usage with Huggingface spaces

If you don't want to run it locally, you can use [this huggingface space](https://huggingface.co/spaces/luisarizmendi/object-detection-batch) that I've created with this code but be aware that this will be slow since I'm using a free instance, so it's better to run it locally with the python script below.

Remember to check that the Model URL is pointing to the model that you want to test.

<div align="center">
  <img width="640" alt="luisarizmendi/hardhat-or-hat" src="https://huggingface.co/luisarizmendi/hardhat-or-hat/resolve/main/spaces-example.png">
</div>



### Usage with Python script

Install the following PIP requirements

```
gradio
ultralytics
Pillow
opencv-python
torch
```

Then [run the python code below ](https://github.com/luisarizmendi/workshop-object-detection-rhde/raw/refs/heads/main/resources/assets/model_test_app/object-detection-batch-model-file/pytorch/object-detection-pytorch.py) and open `http://localhost:8800` in a browser to upload and scan the images.


```
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import torch

DEFAULT_MODEL_URL = "https://github.com/luisarizmendi/workshop-object-detection-rhde/raw/refs/heads/main/resources/assets/object-detection-hardhat-or-hat/v2/model/pytorch/best.pt"

def detect_objects_in_files(model_input, files):
    """
    Processes uploaded images for object detection.
    """
    if not files:
        return "No files uploaded.", []

    model = YOLO(str(model_input))
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
    inputs=[
        gr.Textbox(value=DEFAULT_MODEL_URL, label="Model URL", placeholder="Enter the model URL"),
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

```




