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

import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils, yolov3

sess = tf.Session()
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
input_shape = [416, 416]
dataset = utils.read_image_box_from_text('./data/train_data/quick_train_data.txt')
anchors = utils.get_anchors('./data/yolo_anchors.txt')
# self._ANCHORS = [[10 ,13], [16 , 30], [33 , 23],
                    # [30 ,61], [62 , 45], [59 ,119],
                    # [116,90], [156,198], [373,326]]

inputs = tf.placeholder(tf.float32, shape=[1, 416, 416, 3])
y_true_13 = tf.placeholder(tf.float32, shape=[1,13,13,3,85])
y_true_26 = tf.placeholder(tf.float32, shape=[1,26,26,3,85])
y_true_52 = tf.placeholder(tf.float32, shape=[1,52,52,3,85])

model = yolov3.yolov3(80)
with tf.variable_scope('yolov3'):
    feature_maps = model.forward(inputs, is_training=True)
    load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), "./checkpoint/yolov3.weights")
    sess.run(load_ops)
    loss = model.compute_loss(feature_maps, [y_true_13, y_true_26, y_true_52])
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_op = optimizer.minimize(loss)
sess.run(tf.global_variables_initializer())

for image_path in dataset.keys():
    image = Image.open(image_path)
    true_boxes, true_labels = dataset[image_path]
    image, true_boxes = utils.resize_image_correct_bbox(image, true_boxes, input_shape)
    scores = np.ones(len(true_boxes))
    # utils.draw_boxes(image, boxes, scores, labels, classes)
    true_boxes = np.expand_dims(true_boxes, 0)
    true_labels = np.expand_dims(true_labels, 0)
    y_true = utils.preprocess_true_boxes(true_boxes, true_labels, input_shape, anchors, num_classes)

    image_data = np.expand_dims(np.array(image, dtype=np.float32) / 255., axis=0)

    _, result = sess.run([train_op,loss], feed_dict={inputs:image_data,
                                    y_true_13:y_true[0],
                                    y_true_26:y_true[1],
                                    y_true_52:y_true[2],})
    print(result)













