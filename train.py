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

import tensorflow as tf
from core import utils, yolov3
from core.dataset import dataset, Parser
sess = tf.Session()

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE       = 8
EPOCHS           = 2000*1000
LR               = 0.0001
SHUFFLE_SIZE     = 1000
CLASSES          = utils.read_coco_names('./data/voc.names')
ANCHORS          = utils.get_anchors('./data/voc_anchors.txt')
NUM_CLASSES      = len(CLASSES)

train_tfrecord   = "../VOC/train/voc_train*.tfrecords"
test_tfrecord    = "../VOC/test/voc_test*.tfrecords"

parser   = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
testset  = dataset(parser, test_tfrecord , BATCH_SIZE, shuffle=None)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

images, *y_true = example
model = yolov3.yolov3(NUM_CLASSES, ANCHORS)

with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=is_training)
    loss = model.compute_loss(y_pred, y_true)

optimizer = tf.train.AdamOptimizer(LR)
saver = tf.train.Saver(max_to_keep=2)

tf.summary.scalar("loss/coord_loss",   loss[1])
tf.summary.scalar("loss/sizes_loss",   loss[2])
tf.summary.scalar("loss/confs_loss",   loss[3])
tf.summary.scalar("loss/class_loss",   loss[4])

write_op = tf.summary.merge_all()
writer_train = tf.summary.FileWriter("./data/train")
writer_test  = tf.summary.FileWriter("./data/test")

update_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="yolov3/yolo-v3")
with tf.control_dependencies(update_var):
    train_op = optimizer.minimize(loss[0], var_list=update_var) # only update yolo layer

sess.run(tf.global_variables_initializer())
pretrained_weights = tf.global_variables(scope="yolov3/darknet-53")
load_op = utils.load_weights(var_list=pretrained_weights,
                            weights_file="./darknet53.conv.74")
sess.run(load_op)

for epoch in range(EPOCHS):
    run_items = sess.run([train_op, write_op] + loss, feed_dict={is_training:True})
    writer_train.add_summary(run_items[1], global_step=epoch)
    writer_train.flush() # Flushes the event file to disk
    if (epoch+1)%1000 == 0: saver.save(sess, save_path="./checkpoint/yolov3.ckpt", global_step=epoch)

    run_items = sess.run([write_op] + loss, feed_dict={is_training:False})
    writer_test.add_summary(run_items[0], global_step=epoch)
    writer_test.flush() # Flushes the event file to disk

    print("=> EPOCH:%10d \tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
          %(epoch, run_items[2], run_items[3], run_items[4], run_items[5]))







