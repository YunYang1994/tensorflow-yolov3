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
SHUFFLE_SIZE = 1

sess = tf.Session()
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
train_tfrecord = "../COCO/tfrecords/coco_train0000.tfrecords"
anchors = utils.get_anchors('./data/coco_anchors.txt')

# 检查图片的resize是否正确
parser   = Parser(416, 416, anchors, num_classes, debug=True)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=None)
example  = trainset.get_next()
# for l in range(100):
image, gt_boxes = sess.run(example)
image, gt_boxes = image[0], gt_boxes[0]

n_box = len(gt_boxes)
for i in range(n_box):
    image = cv2.rectangle(image,(int(float(gt_boxes[i][0])),
                                int(float(gt_boxes[i][1]))),
                                (int(float(gt_boxes[i][2])),
                                int(float(gt_boxes[i][3]))), (255,0,0), 2)

image = Image.fromarray(np.uint8(image))

anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
grid_sizes = [[INPUT_SIZE//x, INPUT_SIZE//x] for x in (32, 16, 8)]

box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2 # the center of box
box_sizes =    gt_boxes[:, 2:4] - gt_boxes[:, 0:2] # the height and width of box

# ================================ 分割线============================== #

gt_boxes[:, 0:2] = box_centers
gt_boxes[:, 2:4] = box_sizes

y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5+num_classes], dtype=np.float32)
y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+num_classes], dtype=np.float32)
y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+num_classes], dtype=np.float32)

y_true = [y_true_13, y_true_26, y_true_52]
anchors_max =  anchors / 2.
anchors_min = -anchors_max
valid_mask = box_sizes[:, 0] > 0

# Discard zero rows.
wh = box_sizes[valid_mask]
# set the center of all boxes as the origin of their coordinates
# and correct their coordinates
wh = np.expand_dims(wh, -2)
boxes_max = wh / 2.
boxes_min = -boxes_max

intersect_mins = np.maximum(boxes_min, anchors_min)
intersect_maxs = np.minimum(boxes_max, anchors_max)
intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
box_area       = wh[..., 0] * wh[..., 1]

anchor_area = anchors[:, 0] * anchors[:, 1]
iou = intersect_area / (box_area + anchor_area - intersect_area)
# Find best anchor for each true box
best_anchor = np.argmax(iou, axis=-1)

for t, n in enumerate(best_anchor):
    for l in range(3):
        if n not in anchor_mask[l]: continue

        i = np.floor(gt_boxes[t,0]/INPUT_SIZE*grid_sizes[l][1]).astype('int32')
        j = np.floor(gt_boxes[t,1]/INPUT_SIZE*grid_sizes[l][0]).astype('int32')

        k = anchor_mask[l].index(n)
        c = gt_boxes[t, 4].astype('int32')
        print(j, i, k)
        y_true[l][j, i, k, 0:4] = gt_boxes[t, 0:4]
        y_true[l][j, i, k,   4] = 1.
        y_true[l][j, i, k, 5+c] = 1.
        print(y_true[l][j,i,k])




# box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2 # the center of box
# box_sizes =    gt_boxes[:, 2:4] - gt_boxes[:, 0:2] # the height and width of box


# y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5+num_classes], dtype=np.float32)
# y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+num_classes], dtype=np.float32)
# y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+num_classes], dtype=np.float32)

# y_true = [y_true_13, y_true_26, y_true_52]
# anchors_max =  anchors / 2.
# anchors_min = -anchors_max
# valid_mask = box_sizes[:, 0] > 0

# box_sizes = np.expand_dims(box_sizes, 1)
# mins = np.maximum(- box_sizes / 2, - anchors / 2)
# maxs = np.minimum(box_sizes / 2, anchors / 2)
# # [N, 9, 2]
# whs = maxs - mins
# # [N, 9]
# iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1])
# # [N]
# best_match_idx = np.argmax(iou, axis=1)

# ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
# for i, idx in enumerate(best_match_idx):
    # # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 2
    # feature_map_group = 2 - idx // 3
    # # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
    # ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
    # print("=>ratio", ratio)
    # x = int(np.floor(box_centers[i, 0] / ratio))
    # y = int(np.floor(box_centers[i, 1] / ratio))
    # k = anchor_mask[feature_map_group].index(idx)
    # c = int(gt_boxes[i, 4])
    # # print feature_map_group, '|', y,x,k,c
    # print( y, x, k )
    # y_true[feature_map_group][y, x, k, :2] = box_centers[i]
    # y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
    # y_true[feature_map_group][y, x, k, 4] = 1.
    # y_true[feature_map_group][y, x, k, 5+c] = 1.
    # print(y_true[feature_map_group][y,x,k, :4])


