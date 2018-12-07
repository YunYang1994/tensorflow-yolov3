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


import cv2
from PIL import Image
import numpy as np
from core import utils

classes = utils.read_coco_names('./data/coco.names')

data = open('./data/train_data/quick_train_data.txt', 'r').readlines()

example = data[0].split(' ')




# image_path, category_id, x_min, y_min, x_max, y_max = example

# id = int(category_id)
# x_min = float(x_min)
# x_max = float(x_max)
# y_min = float(y_min)
# y_max = float(y_max)


# boxes = np.array([x_min, y_min, x_max, y_max])
# boxes = boxes.reshape([1, 4])
# scores = np.array([1])
# labels = np.array([id])

# image = Image.open(image_path)

# utils.draw_boxes(boxes, scores, labels, image, classes, image.size)
