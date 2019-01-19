#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2018-12-20 11:58:21
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from core.dataset import dataset, Parser
from core import utils

INPUT_SIZE = 416
BATCH_SIZE = 1
EPOCHS = 313
SHUFFLE_SIZE = 1000
WEIGHTS_PATH = "./checkpoint/yolov3.ckpt"

sess = tf.Session()
classes = utils.read_coco_names('./data/raccoon.names')
num_classes = len(classes)
train_tfrecord = "./raccoon_dataset/raccoon*.tfrecords"
anchors = utils.get_anchors('./data/raccoon_anchors.txt')

parser   = Parser(416, 416, anchors, num_classes, debug=True)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=100)
example  = trainset.get_next()

# for l in range(20):
    # image, boxes = sess.run(example)
    # image, boxes = image[0], boxes[0]

    # n_box = len(boxes)
    # for i in range(n_box):
        # image = cv2.rectangle(image,(int(float(boxes[i][0])),
                                    # int(float(boxes[i][1]))),
                                    # (int(float(boxes[i][2])),
                                    # int(float(boxes[i][3]))), (255,0,0), 2)

    # image = Image.fromarray(np.uint8(image))
    # image.show()

parser   = Parser(416, 416, anchors, num_classes, debug=False)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=100)

image ,y_true_13, y_true_26, y_true_52 = sess.run(trainset.get_next())







