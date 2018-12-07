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


from PIL import Image
import numpy as np
from core import utils

classes = utils.read_coco_names('./data/coco.names')

data = open('./data/train_data/quick_train_data.txt', 'r').readlines()

example = data[5].split(' ')

image_path = example[0]
boxes_num = len(example[1:]) // 5

bboxes = np.zeros([boxes_num, 4], dtype=np.float64)
labels = np.zeros([boxes_num,  ], dtype=np.int32)

for i in range(boxes_num):
    labels[i] = example[1+i*5]
    bboxes[i] = [float(x) for x in example[2+i*5:6+i*5]]

scores = np.array([1]*boxes_num)

image = Image.open(image_path)

utils.draw_boxes(bboxes, scores, labels, image, classes, image.size)
