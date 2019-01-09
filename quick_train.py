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
num_classes = 20
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
file_pattern = "../voc_tfrecords/voc*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt') # numpy [9,2]

dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser(anchors, num_classes).parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()

images, y_true = example
model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=False)
    loss = model.compute_loss(y_pred, y_true)


sess.run(tf.global_variables_initializer())

pretrained_weights = tf.global_variables(scope="yolov3/darknet-53")
load_op = utils.load_weights(var_list=pretrained_weights,
                            weights_file="../yolo_weghts/darknet53.conv.74")
sess.run(load_op)

result = sess.run([loss])





