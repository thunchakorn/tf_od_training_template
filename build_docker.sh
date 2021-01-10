# bin/bash

cd models
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t tf_od .
cd ..
docker build -t tf_od_training .