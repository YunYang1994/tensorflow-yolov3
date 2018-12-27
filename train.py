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

INPUT_SIZE = 416
BATCH_SIZE = 1
EPOCHS = 20
LR = 0.001
# SHUFFLE_SIZE = 10000
SHUFFLE_SIZE = 10

sess = tf.Session()
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
file_pattern = "/home/yang/test/COCO/tfrecords/coco*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt')

is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser(anchors, num_classes).parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()

images, *y_true = example
model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=is_training)
    result = model.compute_loss(y_pred, y_true)

optimizer = tf.train.AdamOptimizer(LR)
train_op = optimizer.minimize(result[3])
saver = tf.train.Saver(max_to_keep=2)
sess.run(tf.global_variables_initializer())

tf.summary.scalar("recall_50" , result[1])
tf.summary.scalar("avg_iou"   , result[2])
tf.summary.scalar("total_loss", result[3])
tf.summary.scalar("coord_loss", result[4])
tf.summary.scalar("sizes_loss", result[5])
tf.summary.scalar("confs_loss", result[6])
tf.summary.scalar("class_loss", result[7])
write_op = tf.summary.merge_all()
writer_train = tf.summary.FileWriter("./data/log/train", graph=sess.graph)

for epoch in range(EPOCHS):
    run_items = sess.run([train_op, write_op] + result, feed_dict={is_training:True})
    writer_train.add_summary(summary=run_items[1], global_step=epoch)
    writer_train.flush() # Flushes the event file to disk
    if epoch%10 == 0: saver.save(sess, save_path="./checkpoint/yolov3.ckpt", global_step=epoch)

    print("=> EPOCH:%4d\t| prec_50:%.4f\trec_50:%.4f\tavg_iou:%.4f\t | total_loss:%7.4f\tloss_coord:%7.4f"
          "\tloss_sizes:%7.4f\tloss_confs:%7.4f\tloss_class:%7.4f" %(epoch, run_items[2], run_items[3],
                        run_items[4], run_items[5], run_items[6], run_items[7], run_items[8], run_items[9]))





