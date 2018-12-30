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
SHUFFLE_SIZE = 1
weights_path = "/home/yang/test/yolov3.weights"

sess = tf.Session()
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
file_pattern = "./data/train_data/quick_train_data/tfrecords/quick_train_data*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt')

dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser(anchors, num_classes).parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()
images, *y_true = example
model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=False)
    loss = model.compute_loss(y_pred, y_true)
    y_pred = model.predict(y_pred)
    load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), weights_path)
    sess.run(load_ops)

for epoch in range(EPOCHS):
    run_items = sess.run([y_pred, y_true] + loss)
    rec, prec, mAP = utils.evaluate(run_items[0], run_items[1], num_classes, score_thresh=0.3, iou_thresh=0.5)

    print("=> EPOCH: %2d\ttotal_loss:%7.4f\tloss_coord:%7.4f\tloss_sizes:%7.4f\tloss_confs:%7.4f\tloss_class:%7.4f"
          "\trec:%7.4f\tprec:%7.4f\tmAP:%7.4f"
          %(epoch, run_items[2], run_items[3], run_items[4], run_items[5], run_items[6], rec, prec, mAP))



