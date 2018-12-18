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
from core import utils

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


sess = tf.Session()

result = sess.run([image, bboxes, labels])




