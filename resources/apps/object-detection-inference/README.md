# Object detection inference server

## Description

This Flask application provides object detection system using YOLO (You Only Look Once) for computer vision tasks.


## Features

- Object detection using YOLO
- Webcam streaming with object annotations
- Batch image processing (for testing)
- Configurable detection thresholds
- Endpoint giving predictions
- GPU/CPU support

## Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Flask

You can run `pip install` using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Environment Variables

The application supports the following environment variables for configuration:

| Variable             | Description                                     | Default Value                |
|----------------------|------------------------------------------------|------------------------------|
| `YOLO_MODEL_PATH`    | Directory path for YOLO model                   | Current directory (`.`)      |
| `YOLO_MODEL_FILE`    | Filename of the YOLO model                      | `object-detection-hardhat-v1-m.pt` |
| `YOLO_MODEL_THRESHOLD` | Global confidence threshold for object detection | `0.25`                       |
| `CAMERA_INDEX`       | Specific camera index to use                    | `-1` (auto-select)           |

## Endpoints

### 1. `http://<ip>:5000/video_stream`
- **Purpose**: Stream webcam with object detection annotations
- **Returns**: Multipart stream with annotated video frames and detection JSON

### 2. `http://<ip>:5000/current_detections` (GET)
- **Purpose**: Retrieve current object detection statistics
- **Returns**: JSON with object class counts and confidence levels

### 3. `http://<ip>:5000/detect_image` (POST)
- **Purpose**: Process batch image uploads
- **Accepts**: Multipart form-data with multiple image files
- **Returns**: JSON with processed images, object counts, and base64 encoded images

## Usage Examples

### Running the Application

```bash
# Set environment variables (optional)
export YOLO_MODEL_THRESHOLD=0.3
export CAMERA_INDEX=0

# Run the application
python object-detection-server.py
```

If you want to run it containerized, you will need to run with root user as a "privileged" container to have access to the video input device.

```bash
sudo podman run -d -p 5000:5000 --privileged <image name>
```
> **Note:**
> You can find an image in `quay.io/luisarizmendi/object-detection-webcam:x86`. It is a big image so it could take time to pull it.

> **Note:**
> You can select the device to be used by setting the environment variable `CAMERA_INDEX`.

If you want to run the Container with podman accessing and using the GPUs, be sure that you have installed the `nvidia-container-toolkit` and the NVIDIA drivers in your host and you [configured the Container Device Interface for Podman](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

The Container always tries to run the model on a GPU (fallbacks to CPU if it's not found) so you just need to bind it to the running Container to make use of it:


```bash
sudo podman run -d -p 5000:5000 --device nvidia.com/gpu=all --security-opt=label=disable --privileged <image name>
```


### Sending Batch Image Detection Request

You can use the script that you find under the `test` directory or run curl to send a test image:

```bash
curl -X POST -F "images=@example.jpg" http://localhost:5000/detect_batch > response.json
```

Yo can also use python to send images to the batch image detection endpoint:

```python
import requests

url = 'http://localhost:5000/detect_image'
files = [
    ('images', open('image1.jpg', 'rb')),
    ('images', open('image2.jpg', 'rb'))
]
response = requests.post(url, files=files)
```

## Considerations

- The application automatically detects and uses GPU if available
- The application will use the camera configured with `CAMERA_INDEX` environment variable, if it cannot be used the camera selection is then dynamic, choosing the highest resolution available
- Configurable confidence thresholds for fine-tuning detection sensitivity

