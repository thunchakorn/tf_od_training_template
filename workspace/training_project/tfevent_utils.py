from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import tensorflow as tf
import os
import numpy as np

def get_metrics_and_image_from_eval_tfevent(logdir):
    """ get metrics from specific tfevent object and retrun as dict
    args:
        logdir: path to directory contained tfevent object e.g. "models/centernet/eval"
        index: index of ckpt file to be getting from
    return:
        dict of metrics
    """
    INVALID_PUNC = '''!"#$%&'()*+, :;<=>?@[\]^`{|}~'''
    DEFAULT_SIZE_GUIDANCE = {
            "compressedHistograms": 1,
            "images": 1,
            "scalars": 0,  # 0 means load all
            "histograms": 1,
            "tensors":0
    }
    return_dict = dict()
    event_paths = sorted(glob.glob(os.path.join(logdir, "event*")))
    for i, path in enumerate(event_paths):
        
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()['tensors']
        _, step, _ = event_acc.Tensors(tags[0])[0] # get step
        return_dict.update({int(step):dict()})

        image_dir = os.path.join(logdir, f'images_{int(step)}')
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)
        for tag in tags:
            metric_name = 'eval_' + tag.translate(str.maketrans(INVALID_PUNC, '_'*len(INVALID_PUNC)))
            tensor_list = event_acc.Tensors(tag)
            _, _, tensor_content_value = tensor_list[0]
            array = tf.make_ndarray(tensor_content_value)
            if 'side_by_side' in tag:
                save_image_path = os.path.join(image_dir, tag + '.jpg')
                with open(save_image_path, 'wb+') as fp:
                    fp.write(array[2])
            else:
                return_dict[int(step)].update({metric_name:array.item()})
    return return_dict

def get_metrics_from_train_tfevent(logdir, every_n: int = None):
    """ get metrics from specific tfevent object and retrun as dict
    args:
        logdir: path to directory contained tfevent object e.g. "models/centernet/eval"
        every_n: get metric every n train step, if None then get all steps
    return:
        dict of list of metrics
    """
    INVALID_PUNC = '''!"#$%&'()*+, :;<=>?@[\]^`{|}~'''
    IGNORE_TAGS = ['train_input_images', 'steps_per_sec']
    DEFAULT_SIZE_GUIDANCE = {
            "compressedHistograms": 1,
            "images": 1,
            "scalars": 0,  # 0 means load all
            "histograms": 1,
            "tensors":0
        }
    # get data from tfevent file
    path = sorted(glob.glob(os.path.join(logdir, "event*")))[0]
    event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    tags = [x for x in event_acc.Tags()['tensors'] if x not in IGNORE_TAGS]
    all_step = len(event_acc.Tensors(tags[0]))
    if every_n is None:
        return_dict = {i:dict() for i in range(all_step)}
        for tag in tags:
            metric_name = 'train_' + tag.translate(str.maketrans(INVALID_PUNC, '_'*len(INVALID_PUNC)))
            tensor_list = event_acc.Tensors(tag)
            for i, (_, _, tensor_content_value) in enumerate(tensor_list):
                scalar_value = tf.make_ndarray(tensor_content_value)
                return_dict[i].update({metric_name:scalar_value.item()})
        return return_dict
    else:
        every_n_step = np.arange(0, all_step + 1, every_n)
        return_dict = {i:dict() for i in every_n_step}
        for tag in tags:
            metric_name = 'train_' + tag.translate(str.maketrans(INVALID_PUNC, '_'*len(INVALID_PUNC)))
            tensor_list = event_acc.Tensors(tag)
            for i, (_, _, tensor_content_value) in enumerate(tensor_list):
                if i in every_n_step:
                    scalar_value = tf.make_ndarray(tensor_content_value)
                    return_dict[i].update({metric_name:scalar_value.item()})
        return return_dict

def get_best_eval_train_step(logdir, metric_name = 'Loss/total_loss', obj_func = 'min'):
    """
    args:
        logdir: path to directory contained tfevent object e.g. "models/centernet/eval"
        obj_name: metric name that want to optimize
        obj_func: choose between "min" for minimum or "max" for maximum
    """
    assert obj_func in ('min', 'max'), 'obj_func must be either "min" or "max".'
    event_paths = sorted(glob.glob(os.path.join(logdir, "event*")))
    metric_record = list()
    for path in event_paths:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        list_ = event_acc.Tensors(metric_name)
        _, _, t = list_[0]
        arr = tf.make_ndarray(t)
        metric_record.append(arr.item())
    if obj_func == 'min':
        result = np.argmin(metric_record)
    else:
        result = np.argmax(metric_record)
    return result