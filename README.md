# tf_od_training_template
**Template for training tensorflow object detection API.**

This repo follow tutorial in this link: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

**This repo is design to run in docker environment**
## Steps
1. build tensorflow object detection images
```bash
cd models
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t tf_od_training .
```
2. run container
```bash
# using gpu
docker run --rm --gpus all \
-v $(pwd)/workspace/:/home/tensorflow/workspace \
-it tf_od_training bash
# not using gpu
docker run --rm \
-v $(pwd)/workspace/:/home/tensorflow/workspace \
-it tf_od_training 
```