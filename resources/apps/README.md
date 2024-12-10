# Object detection with Webcam

This is an example use case of how you can use a webcam to detect objects at the edge, and how those detections can trigger messages/alarms that can be visualized in a dashboard on the Core Datacenter/Cloud.

## Application architecture

![](images/object-detection-webcam.png)


[Inference server](object-detection-inference/README.md)

[Actuator service](object-detection-action/README.md)

[Dashboard backend](object-detection-dashboard/src/backend/README.md)

[Dashboard frontend](object-detection-dashboard/src/frontend/README.md)


## Application workflow

1. The inference server, that contains the AI model detecting objects, publish the predictions in an endpoint
2. The "action" service calls the inference endpoint and if detects certain objects it will trigger an alarm, that is sent to the database hosted in a remote site.
3. The information of the database is shown in the database


## Example screenshots

Live video stream detection from the inference server:

![](images/screenshot_video_stream.png)


Main Dashboard screen:

![](images/screenshot_dashboard_main.png)


Device details screen on the Dashboard:

![](images/screenshot_dashboard_detail.png)



## Quick local setup

If you want to run a quick test you could run all components locally. First start the inference server as root:

```bash
sudo podman run -d -p 5000:5000 --privileged quay.io/luisarizmendi/object-detection-webcam:x86
```

If you have an NVIDIA card in your device you can use it:

```bash
sudo podman run -d -p 5000:5000 --device nvidia.com/gpu=all --security-opt=label=disable  --privileged quay.io/luisarizmendi/object-detection-webcam:x86
```

After that, start the rest of the services:


```bash
podman run -d --network=host quay.io/luisarizmendi/object-detection-action:x86

podman run -d -p 5005:5005 quay.io/luisarizmendi/object-detection-dashboard-backend:v1

podman run -d -p 3000:3000 quay.io/luisarizmendi/object-detection-dashboard-frontend:v1
```


Then you can see the video stream in `http://localhost:5000/video_stream` and the dashboard in `http://localhost:3000`





