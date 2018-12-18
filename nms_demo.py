#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : nms_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-27 13:02:17
#   Description :
#
#================================================================

import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils


SIZE = [416, 416]
# SIZE = [608, 608]
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
img = Image.open('./data/demo_data/611.jpg')
img_resized = np.array(img.resize(size=tuple(SIZE)), dtype=np.float32)
img_resized = img_resized / 255.
cpu_nms_graph, gpu_nms_graph = tf.Graph(), tf.Graph()

# nms on GPU
input_tensor, output_tensors = utils.read_pb_return_tensors(gpu_nms_graph, "./checkpoint/yolov3_gpu_nms.pb",
                                           ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])
with tf.Session(graph=gpu_nms_graph) as sess:
    for i in range(5):
        start = time.time()
        boxes, scores, labels = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        print("=> nms on gpu the number of boxes= %d  time=%.2f ms" %(len(boxes), 1000*(time.time()-start)))
    image = utils.draw_boxes(img, boxes, scores, labels, classes, SIZE, show=True)
# nms on CPU
input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_9:0"])
with tf.Session(graph=cpu_nms_graph) as sess:
    for i in range(5):
        start = time.time()
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.2, iou_thresh=0.3)
        print("=> nms on cpu the number of boxes= %d  time=%.2f ms" %(len(boxes), 1000*(time.time()-start)))
    image = utils.draw_boxes(img, boxes, scores, labels, classes, SIZE, show=True)

image.save('./docs/images/611_result.jpg')


