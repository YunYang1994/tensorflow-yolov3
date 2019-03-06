#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : quick_train.py
#   Author      : YunYang1994
#   Created date: 2019-01-21 14:46:26
#   Description :
#
#================================================================

import tensorflow as tf
from core import utils, yolov3
from core.dataset import dataset, Parser
sess = tf.Session()

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE       = 8
STEPS            = 2500
LR               = 0.001 # if Nan, set 0.0005, 0.0001
DECAY_STEPS      = 100
DECAY_RATE       = 0.9
SHUFFLE_SIZE     = 200
CLASSES          = utils.read_coco_names('./data/raccoon.names')
ANCHORS          = utils.get_anchors('./data/raccoon_anchors.txt', IMAGE_H, IMAGE_W)
NUM_CLASSES      = len(CLASSES)
EVAL_INTERNAL    = 100
SAVE_INTERNAL    = 500

train_tfrecord   = "./raccoon_dataset/raccoon_train.tfrecords"
test_tfrecord    = "./raccoon_dataset/raccoon_test.tfrecords"

parser   = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
testset  = dataset(parser, test_tfrecord , BATCH_SIZE, shuffle=None)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

images, *y_true = example
model = yolov3.yolov3(NUM_CLASSES, ANCHORS)

with tf.variable_scope('yolov3'):
    pred_feature_map = model.forward(images, is_training=is_training)
    loss             = model.compute_loss(pred_feature_map, y_true)
    y_pred           = model.predict(pred_feature_map)

tf.summary.scalar("loss/coord_loss",   loss[1])
tf.summary.scalar("loss/sizes_loss",   loss[2])
tf.summary.scalar("loss/confs_loss",   loss[3])
tf.summary.scalar("loss/class_loss",   loss[4])

global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
write_op = tf.summary.merge_all()
writer_train = tf.summary.FileWriter("./data/train")
writer_test  = tf.summary.FileWriter("./data/test")

saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3/darknet-53"]))
update_vars = tf.contrib.framework.get_variables_to_restore(include=["yolov3/yolo-v3"])
learning_rate = tf.train.exponential_decay(LR, global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
saver_to_restore.restore(sess, "./checkpoint/yolov3.ckpt")
saver = tf.train.Saver(max_to_keep=2)

for step in range(STEPS):
    run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training:True})

    if (step+1) % EVAL_INTERNAL == 0:
        train_rec_value, train_prec_value = utils.evaluate(run_items[2], run_items[3])

    writer_train.add_summary(run_items[1], global_step=step)
    writer_train.flush() # Flushes the event file to disk
    if (step+1) % SAVE_INTERNAL == 0: saver.save(sess, save_path="./checkpoint/yolov3.ckpt", global_step=step+1)

    print("=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
        %(step+1, run_items[5], run_items[6], run_items[7], run_items[8]))

    run_items = sess.run([write_op, y_pred, y_true] + loss, feed_dict={is_training:False})
    if (step+1) % EVAL_INTERNAL == 0:
        test_rec_value, test_prec_value = utils.evaluate(run_items[1], run_items[2])
        print("\n=======================> evaluation result <================================\n")
        print("=> STEP %10d [TRAIN]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, train_rec_value, train_prec_value))
        print("=> STEP %10d [VALID]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, test_rec_value,  test_prec_value))
        print("\n=======================> evaluation result <================================\n")

    writer_test.add_summary(run_items[0], global_step=step)
    writer_test.flush() # Flushes the event file to disk

