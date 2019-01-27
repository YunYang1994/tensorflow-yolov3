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

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE = 1
SHUFFLE_SIZE = 1

train_tfrecord = "./raccoon_dataset/raccoon_*.tfrecords"
anchors        = utils.get_anchors('./data/raccoon_anchors.txt', IMAGE_H, IMAGE_W)
classes = utils.read_coco_names('./data/raccoon.names')
num_classes = len(classes)

parser   = Parser(IMAGE_H, IMAGE_W, anchors, num_classes, debug=True)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)

is_training = tf.placeholder(tf.bool)
example = trainset.get_next()

for l in range(10):
    image, boxes = sess.run(example)
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
