

backend: 

podman run -it -p 5005:5005 quay.io/luisarizmendi/object-detection-dashboard-backend:v1




frontend:

podman run -it -p 3000:3000 quay.io/luisarizmendi/object-detection-dashboard-frontend:v1
