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

import utils
import time
import numpy as np
import tensorflow as tf
from PIL import Image


classes = utils.get_classes('./data/coco.names')
num_classes = len(classes)
img = Image.open('./data/demo_data/road.jpeg')
# img = Image.open('./data/demo_data/dog.jpg')
img_resized = img.resize(size=(416, 416))
cpu_nms_graph, gpu_nms_graph = tf.Graph(), tf.Graph()

input_tensor, output_tensors = utils.read_pb_return_tensors(gpu_nms_graph, "./checkpoint/yolov3_gpu_nms.pb",
                                           ["Placeholder:0", "concat_1:0", "concat_2:0", "concat_3:0"])

with tf.Session(graph=gpu_nms_graph) as sess:
    for i in range(40):
        start = time.time()
        boxes, scores, labels = sess.run(output_tensors, feed_dict={input_tensor: [np.array(img_resized, dtype=np.float32)]})
        print("=> nms on gpu: the number of boxes= %d  time=%.2f ms" %(len(boxes), 1000*(time.time()-start)))
    image = utils.draw_boxes(boxes, scores, labels, img, classes, show=True)


input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3_cpu_nms.pb",
                                           ["Placeholder:0", "concat:0", "mul:0"])

with tf.Session(graph=cpu_nms_graph) as sess:
    for i in range(40):
        start = time.time()
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: [np.array(img_resized, dtype=np.float32)]})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
        print("=> nms on cpu: the number of boxes= %d  time=%.2f ms" %(len(boxes), 1000*(time.time()-start)))
    image = utils.draw_boxes(boxes, scores, labels, img, classes, show=True)










