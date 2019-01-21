#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : debug.py
#   Author      : YunYang1994
#   Created date: 2019-01-21 15:02:05
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from core import utils
from PIL import Image
from core.dataset import Parser, dataset
sess = tf.Session()

INPUT_SIZE = 416
BATCH_SIZE = 1
SHUFFLE_SIZE = 1

train_tfrecord = "/home/yang/VOC/train/voc_train*.tfrecords"
test_tfrecord  = "/home/yang/VOC/test/voc_test*.tfrecords"
anchors        = utils.get_anchors('./data/yolo_anchors.txt')
classes = utils.read_coco_names('./data/voc.names')
num_classes = len(classes)

parser   = Parser(416, 416, anchors, num_classes, debug=True)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
testset  = dataset(parser, test_tfrecord , BATCH_SIZE, shuffle=None)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

for l in range(20):
    image, boxes = sess.run(example, feed_dict={is_training:False})
    image, boxes = image[0], boxes[0]

    n_box = len(boxes)
    for i in range(n_box):
        image = cv2.rectangle(image,(int(float(boxes[i][0])),
                                     int(float(boxes[i][1]))),
                                    (int(float(boxes[i][2])),
                                     int(float(boxes[i][3]))), (255,0,0), 1)
        label = classes[boxes[i][4]]
        image = cv2.putText(image, label, (int(float(boxes[i][0])),int(float(boxes[i][1]))),
                            cv2.FONT_HERSHEY_SIMPLEX,  .6, (0, 255, 0), 1, 2)

    image = Image.fromarray(np.uint8(image))
    image.show()
