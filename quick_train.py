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
file_pattern = "./data/train_data/quick_train_data/tfrecords/quick_train_data*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt')


dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser(anchors, num_classes).parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()
images, *y_true = example

model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=True)
    result = model.compute_loss(y_pred, y_true)

optimizer = tf.train.AdamOptimizer(LR)
train_op = optimizer.minimize(result[3])
sess.run(tf.global_variables_initializer())

for epoch in range(EPOCHS):
    run_items = sess.run([train_op] + result)
    print("=> EPOCH:%4d\t| prec_50:%.4f\trec_50:%.4f\tavg_iou:%.4f\t | total_loss:%7.4f\tloss_coord:%7.4f"
          "\tloss_sizes:%7.4f\tloss_confs:%7.4f\tloss_class:%7.4f" %(epoch, run_items[1], run_items[2],
                        run_items[3], run_items[4], run_items[5], run_items[6], run_items[7], run_items[8]))

#************************ test with yolov3.weights ****************************#

# model = yolov3.yolov3(num_classes)
# with tf.variable_scope('yolov3'):
    # y_pred = model.forward(images, is_training=False)
    # load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), "./checkpoint/yolov3.weights")
    # sess.run(load_ops)
    # result = model.compute_loss(y_pred, y_true)

# for epoch in range(EPOCHS):
    # run_items = sess.run(result)
    # print("=> EPOCH:%4d\t| prec_50:%.4f\trec_50:%.4f\tavg_iou:%.4f\t | total_loss:%7.4f\tloss_coord:%7.4f"
          # "\tloss_sizes:%7.4f\tloss_confs:%7.4f\tloss_class:%7.4f" %(epoch, run_items[0], run_items[1],
                        # run_items[2], run_items[3], run_items[4], run_items[5], run_items[6], run_items[7]))


