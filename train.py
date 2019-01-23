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
LR               = 0.0005
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
    loss   = model.compute_loss(y_pred, y_true)
    y_pred = model.predict(y_pred)

rec_50  = tf.Variable(0., trainable=False)
prec_50 = tf.Variable(0., trainable=False)

tf.summary.scalar("loss/coord_loss",   loss[1])
tf.summary.scalar("loss/sizes_loss",   loss[2])
tf.summary.scalar("loss/confs_loss",   loss[3])
tf.summary.scalar("loss/class_loss",   loss[4])

tf.summary.scalar("yolov3/total_loss", loss[0])
tf.summary.scalar("yolov3/rec_50",     rec_50)
tf.summary.scalar("yolov3/prec_50",   prec_50)

write_op = tf.summary.merge_all()
writer_train = tf.summary.FileWriter("./data/train")
writer_test  = tf.summary.FileWriter("./data/test")
saver = tf.train.Saver(max_to_keep=2)

pretrained_weights = tf.global_variables(scope="yolov3/darknet-53")
load_op = utils.load_weights(var_list=pretrained_weights, weights_file="./darknet53.conv.74")

optimizer = tf.train.AdamOptimizer(LR)
update_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="yolov3/yolo-v3")
with tf.control_dependencies(update_var):
    train_op = optimizer.minimize(loss[0], var_list=update_var) # only update yolo layer

sess.run(tf.global_variables_initializer())
sess.run(load_op)
for epoch in range(EPOCHS):

    run_items = sess.run([train_op, y_pred, y_true] + loss, feed_dict={is_training:True})
    rec_value, prec_value = utils.evaluate(run_items[1], run_items[2])
    _, _, summary = sess.run([tf.assign(rec_50,  rec_value),
                              tf.assign(prec_50, prec_value), write_op], {is_training:True})

    writer_train.add_summary(summary, global_step=epoch)
    writer_train.flush() # Flushes the event file to disk
    if (epoch+1)%1000 == 0: saver.save(sess, save_path="./checkpoint/yolov3.ckpt", global_step=epoch)

    print("=> EPOCH:%10d \tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
          "\ttotal_loss:%7.4f\t rec_50:%.2f\t prec_50:%.2f"
          %(epoch, run_items[4], run_items[5], run_items[6], run_items[7], run_items[3], rec_value, prec_value))

    run_items = sess.run([y_pred, y_true] + loss, feed_dict={is_training:False})
    rec_value, prec_value = utils.evaluate(run_items[0], run_items[1])

    _, _, summary = sess.run([tf.assign(rec_50,  rec_value),
                              tf.assign(prec_50, prec_value), write_op], {is_training:False})

    writer_test.add_summary(summary, global_step=epoch)
    writer_test.flush() # Flushes the event file to disk

