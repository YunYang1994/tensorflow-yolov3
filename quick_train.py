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

import tensorflow as tf
from core import utils, yolov3

INPUT_SIZE = 416
BATCH_SIZE = 1
EPOCHS = 20
LR = 0.001

sess = tf.Session()
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
file_pattern = "./data/train_data/tfrecords/quick_train_data*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt')

dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser, num_parallel_calls = 10)
dataset = dataset.repeat().batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
images, true_boxes, true_labels = iterator.get_next()
images, true_boxes = utils.resize_image_correct_bbox(images, true_boxes, [INPUT_SIZE, INPUT_SIZE])

y_true = tf.py_func(utils.preprocess_true_boxes,
                    inp=[true_boxes, true_labels, [INPUT_SIZE, INPUT_SIZE], anchors, num_classes],
                    Tout = [tf.float32, tf.float32, tf.float32])

y_true_13 = tf.placeholder(tf.float32, shape=[1,13,13,3,85])
y_true_26 = tf.placeholder(tf.float32, shape=[1,26,26,3,85])
y_true_52 = tf.placeholder(tf.float32, shape=[1,52,52,3,85])

model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    feature_maps = model.forward(images, is_training=True)
    # load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), "./checkpoint/yolov3.weights")
    # sess.run(load_ops)
    loss_items = model.compute_loss(feature_maps, y_true)
    loss = sum(loss_items)

optimizer = tf.train.AdamOptimizer(LR)
train_op = optimizer.minimize(loss)
sess.run(tf.global_variables_initializer())

for epoch in range(EPOCHS):
    result = sess.run([train_op, loss] + loss_items)
    _, total_loss, loss_coord, loss_sizes, loss_confs, loss_class = result
    print("=> EPOCH:%4d, total_loss:%9.4f\tloss_coord:%9.4f\tloss_sizes:%9.4f\tloss_confs:%9.4f\tloss_class:%9.4f\t"
          %(epoch, total_loss, loss_coord, loss_sizes, loss_confs, loss_class))




