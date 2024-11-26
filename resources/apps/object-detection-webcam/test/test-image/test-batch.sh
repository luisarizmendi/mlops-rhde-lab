#!/bin/bash

rm -f detection_results.json processed_image.jpg

curl -X POST -F "images=@example.jpg" http://localhost:5000/detect_batch > response.json

cat response.json | jq -r '.results[0].image_base64' | base64 -d > processed_image.jpg

cat response.json | jq '.results[0].object_counts' > detection_results.json

rm response.json