# Object detection using Webcam

## Description

Python server that uses a YOLO model to provide object detection on a directly connected webcam stream providing an output stream with the labeled images and an additional endpoint with the number of detected objects.

You can find more information about the [model in this Huggingface page](https://huggingface.co/luisarizmendi/yolo11-safety-equipment) but essentially it detects safety equiment. 


## Enpoints

`http://<ip>:5000/video_stream`

Provides the labeled stream from the webcam input stream.


`http://<ip>:5000/current_detections`

Provides the entities detected and the number of each one in that moment. This endpoint can be used by external services in order to take actions depending on the objects detected.


`http://<ip>:5000/detect_image`

Endpoint that can be used to test image labeling by uploading a specific image. You have an example script using `curl` under the `test` directory:

```bash
curl -X POST -F "images=@example.jpg" http://localhost:5000/detect_batch > response.json
```



## Run

Container will need to run with root user as a "privileged" container to have access to the video input device.

```bash
sudo podman run -d -p 5000:5000 --privileged <image name>
```
> **Note:**
> You have a running image in `quay.io/luisarizmendi/object-detection-webcam:x86`. It is a big image so it could take time to pull it.

> **Note:**
> You can select the device to be used by setting the environment variable `CAMERA_INDEX`.

If you want to run the Container with podman accessing and using the GPUs, be sure that you have installed the `nvidia-container-toolkit` and the NVIDIA drivers in your host and you [configured the Container Device Interface for Podman](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

The Container always tries to run the model on a GPU (fallbacks to CPU if it's not found) so you just need to bind it to the running Container to make use of it:


```bash
sudo podman run -d -p 5000:5000 --device nvidia.com/gpu=all --security-opt=label=disable --privileged <image name>
```

