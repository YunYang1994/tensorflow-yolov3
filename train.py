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
EPOCHS = 10
LR = 0.001
SHUFFLE_SIZE = 1

sess = tf.Session()
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
# file_pattern = "../COCO/tfrecords/coco*.tfrecords"
file_pattern = "./data/train_data/quick_train_data/tfrecords/quick_train_data*.tfrecords"
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
    loss = model.compute_loss(y_pred, y_true)

    y_pred = model.predict(y_pred)


optimizer = tf.train.AdamOptimizer(LR)
train_op = optimizer.minimize(loss[0])
saver = tf.train.Saver(max_to_keep=2)
sess.run(tf.global_variables_initializer())

tf.summary.scalar("yolov3/total_loss", loss[0])


tf.summary.scalar("loss/coord_loss", loss[1])
tf.summary.scalar("loss/sizes_loss", loss[2])
tf.summary.scalar("loss/confs_loss", loss[3])
tf.summary.scalar("loss/class_loss", loss[4])
write_op = tf.summary.merge_all()
writer_train = tf.summary.FileWriter("./data/log/train", graph=sess.graph)

for epoch in range(EPOCHS):
    run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training:True})

    data = utils.evaluate(run_items[2], run_items[3], num_classes)

    writer_train.add_summary(summary=run_items[1], global_step=epoch)
    writer_train.flush() # Flushes the event file to disk
    if epoch%1000 == 0: saver.save(sess, save_path="./checkpoint/yolov3.ckpt", global_step=epoch)

    print("=> EPOCH:%10d | total_loss:%7.4f\tloss_coord:%7.4f\tloss_sizes:%7.4f\tloss_confs:%7.4f\tloss_class:%7.4f"
          %(epoch, run_items[4], run_items[5], run_items[6], run_items[7], run_items[8]))
    break





