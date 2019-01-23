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

