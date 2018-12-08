#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : quick_train.py
#   Author      : YunYang1994
#   Created date: 2018-12-07 17:58:58
#   Description :
#
#================================================================


import numpy as np
from PIL import Image
from core import utils

classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
input_shape = [416, 416]
dataset = utils.read_image_box_from_text('./data/train_data/quick_train_data.txt')
anchors = utils.get_anchors('./data/yolo_anchors.txt')

for image_path in dataset.keys():
    image = Image.open(image_path)
    true_boxes, true_labels = dataset[image_path]
    image, true_boxes = utils.resize_image_correct_bbox(image, true_boxes, input_shape)
    scores = np.ones(len(true_boxes))
    # utils.draw_boxes(image, boxes, scores, labels, classes)
    true_boxes = np.expand_dims(true_boxes, 0)
    true_labels = np.expand_dims(true_labels, 0)
    result = utils.preprocess_true_boxes(true_boxes, true_labels, input_shape, anchors, num_classes)
    break



