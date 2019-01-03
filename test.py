#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2018-12-20 11:58:21
#   Description :
#
#================================================================

import tensorflow as tf
from core import utils, yolov3

INPUT_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 313
SHUFFLE_SIZE = 1000
WEIGHTS_PATH = "./checkpoint/yolov3.ckpt"

sess = tf.Session()
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
file_pattern = "../COCO/val_tfrecords/coco_val*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt')

dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser(anchors, num_classes).parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()
saver = tf.train.Saver()

images, *y_true = example
model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=False)
    y_pred = model.predict(y_pred)

saver.restore(sess, save_path=WEIGHTS_PATH)
for epoch in range(EPOCHS):
    run_items = sess.run([y_pred, y_true])
    rec, prec, mAP = utils.evaluate(run_items[1], run_items[2], num_classes)
    print("=> EPOCH:%10d\trec:%.2f\tprec:%.2f\tmAP:%.2f" %(rec, prec, mAP))







