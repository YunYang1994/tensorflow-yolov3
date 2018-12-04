#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:47:45
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from core import yolov3, dataset
sess = tf.Session()

train_reader = dataset.Reader('train', './data/train_data/TFrecord',
                      './yolo_anchors.txt', 80, input_shape=416, max_boxes=20)
train_data = train_reader.build_dataset(1)

iterator = train_data.make_one_shot_iterator()
images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()

images.set_shape([None, 416, 416, 3])
bbox.set_shape([None, 20, 5])
grid_shapes = [13, 26, 52]

bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], 3, 5 + 80])
bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], 3, 5 + 80])
bbox_true_52.set_shape([None, grid_shapes[2], grid_shapes[2], 3, 5 + 80])
bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]


model = yolov3.yolov3(80)
with tf.variable_scope('yolov3'):
    feature_map = model.forward(images, is_training=True)

feature_map_1, feature_map_2, feature_map_3 = feature_map
saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))
saver.restore(sess, './checkpoint/yolov3.ckpt')

#=========================== compute_loss ========================#

images_ = sess.run(images)
images_ = sess.run(images)
cv2.imshow("hello", images_[0]*255)
cv2.waitKey(30)


# loss = model.compute_loss(feature_map, bbox_true)
# optimizer = tf.train.AdamOptimizer(0.001)
# train_op = optimizer.minimize(loss)
# sess.run(tf.global_variables_initializer())
# for i in range(100):
    # _, loss_val = sess.run([train_op, loss])
    # print(i, loss_val)

# saver.save(sess, './checkpoint/yolov3_100.ckpt')

