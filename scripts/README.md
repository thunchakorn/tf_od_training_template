# Explanation for each python file

## create_coco_tf_record.py

This file is for create tfrecord file from coco annotation file.

### For keypoint detection task
Before running, change "_COCO_KEYPOINT_NAMES" in this file to match your task.
Running example for keypoint task:

```console
python scripts/create_coco_kpt_tf_record.py --logtostderr \
--train_image_dir=workspace/data/images/ \
--val_image_dir=workspace/data/images/ \
--test_image_dir=workspace/data/images/ \
--train_annotations_file=workspace/data/annotation/train.json \
--val_annotations_file=workspace/data/annotation/val.json \
--testdev_annotations_file=workspace/data/annotation/test.json \
--train_keypoint_annotations_file=workspace/data/annotation/train.json \
--output_dir=workspace/data/tfrecord/
```


### For object detection task
```
python scripts/create_coco_kpt_tf_record.py --logtostderr \
--train_image_dir=workspace/data/images/ \
--val_image_dir=workspace/data/images/ \
--test_image_dir=workspace/data/images/ \
--train_annotations_file=workspace/data/annotation/train.json \
--val_annotations_file=workspace/data/annotation/val.json \
--testdev_annotations_file=workspace/data/annotation/test.json \
--output_dir=workspace/data/tfrecord/
```
