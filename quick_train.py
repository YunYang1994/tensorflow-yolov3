#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : quick_train.py
#   Author      : YunYang1994
#   Created date: 2018-12-07 17:58:58
#   Description :
#
#================================================================

#=========================== for debug =========================#

import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils, yolov3

input_shape = [416, 416]
BATCH_SIZE = 1
EPOCHS = 700000
LR = 0.0001
SHUFFLE_SIZE = 1

sess = tf.Session()
num_classes = 80
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
file_pattern = "../COCO/tfrecords/coco*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt') # numpy [9,2]

dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser(anchors, num_classes).parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()

layer = 0
_ANCHORS = [anchors[6:9], anchors[3:6], anchors[0:3]]
images, *y_true = example
model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=False)
    loss = model.loss_layer(y_pred[layer], y_true[layer], _ANCHORS[layer], 0.5, 8)

sess.run(tf.global_variables_initializer())

# pretrained_weights = tf.global_variables(scope="yolov3/darknet-53")
# load_op = utils.load_weights(var_list=pretrained_weights,
                            # weights_file="../yolo_weghts/darknet53.conv.74")

pretrained_weights = tf.global_variables(scope="yolov3")
load_op = utils.load_weights(var_list=pretrained_weights,
                            weights_file="../yolo_weghts/yolov3.weights")
sess.run(load_op)

result = sess.run([y_pred, y_true, loss])
data = result[-1]
print(np.sum(data[0]))
print(np.sum(data[2]))




