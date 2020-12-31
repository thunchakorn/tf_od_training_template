# Instruction for training model.

For more detail click [this link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

## Steps
### 1. Prepare tfrecord and data/label_map.pbtxt
`data/label_map.pbtxt`
example of label_map.pbtxt for object detection task:
```
item {
    id: 1
    name: 'cat'
}

item {
    id: 2
    name: 'dog'
}
```
example of label_map.pbtxt for keypoint detection task:
```
item {
    id: 1
    name: "card"
    keypoints {
        id: 0
        label: 'top_left'
      }
      keypoints {
        id: 1
        label: 'top_right'
      }
      keypoints {
        id: 2
        label: 'bottom_right'
      }
      keypoints {
        id: 3
        label: 'bottom_left'
      }
}
```
### 2. Download Pre-Trained Model
- download TensorFlow 2 Detection Model Zoo from [this link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- this will down load `*.tar.gz` file.
- extract its contents inside the folder `pre-trained-models/` as such
```
training_project/
├─ ...
├─ pre-trained-models/
│  ├─ efficientdet_d1_coco17_tpu-32/
│  │  ├─ checkpoint/
│  │  ├─ saved_model/
│  │  └─ pipeline.config
│  └─ ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```
### 3. Configure the Training Pipeline
Now that we have downloaded and extracted our pre-trained model, let’s create a directory for our training job. Under the training_project/models create a new directory named my_ssd_resnet50_v1_fpn and copy the training_project/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config file inside the newly created directory. Our training_project/models directory should now look like this:
```
training_project/
├─ ...
├─ models/
│  └─ my_ssd_resnet50_v1_fpn/
│     └─ pipeline.config
└─ ...
```
then edit `pipeline.config` 
example for keypoint detection task from centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8:
```
# CenterNet meta-architecture from the "Objects as Points" [1] paper
# with the ResNet-v2-101 backbone. The ResNet backbone has a few differences
# as compared to the one mentioned in the paper, hence the performance is
# slightly worse. This config is TPU comptatible.
# [1]: https://arxiv.org/abs/1904.07850
#

model {
  center_net {
    num_classes:1
    feature_extractor {
      type: "resnet_v1_50_fpn"
    }
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 512
        max_dimension: 512
        pad_to_max_dimension: true
      }
    }
    object_detection_task {
      task_loss_weight: 1.0
      offset_loss_weight: 1.0
      scale_loss_weight: 0.1
      localization_loss {
        l1_localization_loss {
        }
      }
    }
    object_center_params {
      object_center_loss_weight: 1.0
      min_box_overlap_iou: 0.7
      max_box_predictions: 10
      classification_loss {
        penalty_reduced_logistic_focal_loss {
          alpha: 2.0
          beta: 4.0
        }
      }
    }
    keypoint_label_map_path: "data/label_map.pbtxt"
    keypoint_estimation_task {
      task_name: "card_keypoint"
      task_loss_weight: 1.0
      loss {
        localization_loss {
          l1_localization_loss {
          }
        }
        classification_loss {
          penalty_reduced_logistic_focal_loss {
            alpha: 2.0
            beta: 4.0
          }
        }
      }
      keypoint_class_name: "card"
      keypoint_regression_loss_weight: 0.1
      keypoint_heatmap_loss_weight: 1.0
      keypoint_offset_loss_weight: 1.0
      offset_peak_radius: 3
      per_keypoint_offset: true
    }
  }
}

train_config: {
  batch_size: 8
  num_steps: 18000
  data_augmentation_options {
    random_crop_image {
      min_aspect_ratio: 0.5
      max_aspect_ratio: 1.7
      random_coef: 0.25
    }
  }
  data_augmentation_options {
    random_distort_color {
      color_ordering: 0
    }
  }

  data_augmentation_options {
    random_crop_image {
    }
  }
  optimizer {
    adam_optimizer: {
      epsilon: 1e-7  # Match tf.keras.optimizers.Adam's default.
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 1e-3
          total_steps: 18000
          warmup_learning_rate: 2.5e-4
          warmup_steps: 3000
        }
      }
    }
    use_moving_average: false
  }
  max_number_of_boxes: 10
  unpad_groundtruth_tensors: false

  fine_tune_checkpoint_version: V2
  fine_tune_checkpoint: "pre-trained-models/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8/checkpoint/ckpt-0"
  fine_tune_checkpoint_type: "fine_tune"
}
train_input_reader: {
  label_map_path: "data/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "data/tfrecord/coco_train.record-?????-of-00100"
  }
  num_keypoints: 4
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  num_visualizations: 10
  max_num_boxes_to_visualize: 20
  min_score_threshold: 0.2
  batch_size: 8;
  parameterized_metric {
    coco_keypoint_metrics {
      class_label: "card"
    }
  }
  # Provide the edges to connect the keypoints. The setting is suitable for
  # COCO's 4 card keypoints.
  keypoint_edge {  
    start: 0
    end: 1
  }
  keypoint_edge {  
    start: 1
    end: 2
  }
  keypoint_edge {  
    start: 2
    end: 3
  }
  keypoint_edge {  
    start: 3
    end: 0
  }
}
eval_input_reader: {
  label_map_path: "data/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "data/tfrecord/coco_val.record-?????-of-00050"
  }
  num_keypoints: 4
}
```
### 4. Training the Model
 - (Optional) For validation loss monitoring, run `model_main_tf2.py` with --checkpoint_dir argument.
 for example:
 ```
 python model_main_tf2.py --model_dir=models/centernet_resnet50 \
 --pipeline_config_path=models/centernet_resnet50/pipeline.config \
 --checkpoint_dir=models/centernet_resnet50
 ```
 - For training, run `model_main_tf2.py`.
 for example:
 ```
 python model_main_tf2.py --model_dir=models/centernet_resnet50 \
 --pipeline_config_path=models/centernet_resnet50/pipeline.config \
 --checkpoint_every_n 500 --sample_1_of_n_eval_examples 1 \
 --alsologtostderr
 ```
 - (Optional) monitor via tensorboard
 ```
 tensorboard --logdir=models/centernet_resnet50
 ```
### 5. Export model
example:
```
python exporter_main_v2.py --input_type image_tensor \
--pipeline_config_path ./models/my_efficientdet_d1/pipeline.config \
--trained_checkpoint_dir ./models/my_efficientdet_d1/ \
--output_directory ./exported-models/my_model
```
