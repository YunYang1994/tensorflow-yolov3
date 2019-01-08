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
import tensorflow as tf
from PIL import Image
from core import utils, yolov3

input_shape = [416, 416]
BATCH_SIZE = 1
EPOCHS = 700000
LR = 0.0001
SHUFFLE_SIZE = 1

sess = tf.Session()
num_classes = 20
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
file_pattern = "../voc_tfrecords/voc*.tfrecords"
anchors = utils.get_anchors('./data/yolo_anchors.txt') # numpy [9,2]

is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(file_pattern))
dataset = dataset.map(utils.parser(anchors, num_classes).parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()

images, true_boxes, true_labels = example
model = yolov3.yolov3(num_classes)
with tf.variable_scope('yolov3'):
    y_pred = model.forward(images, is_training=is_training)


sess.run(tf.global_variables_initializer())

pretrained_weights = tf.global_variables(scope="yolov3/darknet-53")
load_op = utils.load_weights(var_list=pretrained_weights,
                            weights_file="../yolo_weghts/darknet53.conv.74")
sess.run(load_op)

data = sess.run([images, true_boxes, true_labels])
# # 展现训练图片，看看bounding box是否对？
# img = Image.fromarray(data[0][0].astype('uint8')).convert('RGB')
# utils.draw_boxes(img, data[1].reshape(-1,4), data[2].reshape(-1), data[2].reshape(-1), classes, input_shape)

image = data[0][0] # numpy [416, 416, 3]
true_boxes = data[1][0] # numpy [?, 4]
true_labels = data[2][0] # [?]

# 下面对单张图片开始做处理
input_shape = np.array(input_shape, dtype=np.int32) #　h, w
num_anchors = len(anchors)
num_layers  = num_anchors // 3
anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
grid_sizes = [input_shape//32, input_shape//16, input_shape//8]

box_centers = (true_boxes[:, 0:2] + true_boxes[:, 2:4]) / 2 # the center of box
box_sizes =  true_boxes[:, 2:4] - true_boxes[:, 0:2] # the height and width of box

true_boxes[:, 0:2] = box_centers # 绝对中心位置
true_boxes[:, 2:4] = box_sizes   # 绝对尺寸

y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5+num_classes], dtype=np.float32)
y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+num_classes], dtype=np.float32)
y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+num_classes], dtype=np.float32)

y_true = [y_true_13, y_true_26, y_true_52]

# 因为anchor的中心位于图片上网格的中心，所以可以计算出它们的左上角和右下角坐标
anchors_max =  anchors / 2. # 此时的坐标原点位于网格的中心
anchors_min = -anchors_max


# set the center of all boxes as the origin of their coordinates
# and correct their coordinates

for t in range(len(true_boxes)):
    grid_xy = []
    box_min = np.zeros([num_anchors, 2])
    box_max = np.zeros([num_anchors, 2])
    for l in range(num_layers):
        # 计算该boundingbox在第几排第几列的网格里
        x = np.floor(true_boxes[t,0]/input_shape[0]*grid_sizes[l][0]).astype('int32')
        y = np.floor(true_boxes[t,1]/input_shape[1]*grid_sizes[l][1]).astype('int32')
        grid_xy.append([x,y])
        # 计算该网格的中心相对位置
        box_center_x = true_boxes[t,0] - (x+0.5)*(input_shape[0]//grid_sizes[l][0])
        box_center_y = true_boxes[t,1] - (y+0.5)*(input_shape[1]//grid_sizes[l][1])
        # 相对于网格中心来说，计算左上和右下角坐标
        box_min[l*3: (l+1)*3] = [np.array([box_center_x, box_center_y]) - true_boxes[t,2:4] / 2.] * 3
        box_max[l*3: (l+1)*3] = [np.array([box_center_x, box_center_y]) + true_boxes[t,2:4] / 2.] * 3
    # 然后计算与９个boundibox的iou值
    intersect_mins = np.maximum(box_min, anchors_min)
    intersect_maxs = np.minimum(box_max, anchors_max)
    intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    box_area = true_boxes[t,2] * true_boxes[t,3]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    # 找出每个iou最大的所对应的那个layer，以及anchor
    max_iou__idx    = np.argmax(iou)
    true_layer_idx  = max_iou__idx // 3 # 是第几层layer
    true_anchor_idx = max_iou__idx % 3  # 是第几层anchor
    true_grid_xy    = grid_xy[true_layer_idx] # 是第几个cell

    print(true_layer_idx, true_grid_xy, true_anchor_idx)
    y_true[true_layer_idx][true_grid_xy[0]][true_grid_xy[1]][true_anchor_idx][0:4] = true_boxes[t]
    y_true[true_layer_idx][true_grid_xy[0]][true_grid_xy[1]][true_anchor_idx][  4] = 1.
    y_true[true_layer_idx][true_grid_xy[0]][true_grid_xy[1]][true_anchor_idx][true_labels[t]] = 1.







