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

import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils

sess = tf.Session()
input_shape = [416, 416]
classes = utils.read_coco_names('./data/coco.names')
file_pattern = "./data/train_data/train_*.tfrecords"

def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image' : tf.FixedLenFeature([], dtype = tf.string),
            'bboxes': tf.FixedLenFeature([], dtype = tf.string),
            'labels': tf.VarLenFeature(dtype = tf.int64),
        }
    )

    image = tf.image.decode_jpeg(features['image'], channels = 3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    bboxes = tf.decode_raw(features['bboxes'], tf.float32)
    bboxes = tf.reshape(bboxes, shape=[-1,4])

    labels = features['labels'].values
    return image, bboxes, labels




dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(parser, num_parallel_calls = 10)
dataset = dataset.repeat().batch(1).prefetch(1)
iterator = dataset.make_one_shot_iterator()
image, bboxes, labels = iterator.get_next()
image, bboxes, labels = iterator.get_next()

# image, bboxes = utils.resize_image_correct_bbox(image, bboxes, input_shape)
result = sess.run([image, bboxes, labels])
result = sess.run([image, bboxes, labels])


result = sess.run([image, bboxes, labels])
image = Image.fromarray(np.uint8(result[0][0]))
boxes = result[1][0]
labels = result[2][0]
scores = np.ones_like(labels)

utils.draw_boxes(image, boxes, scores, labels, classes)
