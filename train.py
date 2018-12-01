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


import time
import numpy as np
from PIL import Image
import tensorflow as tf
from core import utils, yolov3


img = Image.open('./data/demo_data/dog.jpg')
img_resized = img.resize(size=(416, 416))

classes = utils.get_classes('./data/coco.names')
num_classes = len(classes)
model = yolov3.yolov3(num_classes)

with tf.Graph().as_default() as graph:

    sess = tf.Session(graph=graph)
    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])

    with tf.variable_scope('detector'):
        detections = model.forward(inputs)

    boxes, scores = utils.get_boxes_scores(detections)
    # boxes, scores, labels = utils.gpu_nms(boxes, scores, num_classes, 20, 0.3, 0.5)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
    saver.restore(sess, './checkpoint/yolov3.ckpt')

    # output_tensors = [boxes, confs, probs]
    output_tensors = [boxes, scores]
    # for i in range(10):
        # start = time.time()
        # boxes, scores, labels = sess.run(output_tensors, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
        # print(boxes, time.time()-start)

    for i in range(10):
        start = time.time()
        boxes, scores = sess.run(output_tensors, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes)
        print(boxes, time.time()-start)

image = utils.draw_boxes(boxes, scores, labels, img, classes, show=True)
