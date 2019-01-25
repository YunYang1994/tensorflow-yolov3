#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2018-12-20 11:58:21
#   Description : compute mAP
#
#================================================================

import numpy as np
import tensorflow as tf
from core import utils, yolov3
from core.dataset import dataset, Parser
sess = tf.Session()


IMAGE_H, IMAGE_W = 608, 608
BATCH_SIZE       = 8
CLASSES          = utils.read_coco_names('./data/coco.names')
NUM_CLASSES      = len(CLASSES)
ANCHORS          = utils.get_anchors('./data/coco_anchors.txt')
CKPT_FILE        = "./checkpoint/yolov3.ckpt"
IOU_THRESH       = 0.5
SCORE_THRESH     = 0.3

all_detections   = []
all_annotations  = []
test_tfrecord    = "./raccoon_dataset/raccoon_test*.tfrecords"
parser           = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
testset          = dataset(parser, test_tfrecord , BATCH_SIZE, shuffle=None, repeat=False)


images, *y_true  = testset.get_next()
model = yolov3.yolov3(NUM_CLASSES, ANCHORS)
with tf.variable_scope('yolov3'):
    pred_feature_map    = model.forward(images, is_training=False)
    y_pred              = model.predict(pred_feature_map)

saver = tf.train.Saver()
saver.restore(CKPT_FILE)

try:
    while True:
        run_items  = sess.run([y_pred, y_true])
        pred_boxes = y_pred[0][0]
        pred_confs = y_pred[1][0]
        pred_probs = y_pred[2][0]

        true_labels_list, true_boxes_list = [], []
        for i in range(3):
            true_probs_temp = y_true[i][..., 5: ]
            true_boxes_temp = y_true[i][..., 0:4]
            object_mask     = true_probs_temp.sum(axis=-1) > 0

            true_probs_temp = true_probs_temp[object_mask]
            true_boxes_temp = true_boxes_temp[object_mask]

            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            true_boxes_list  += true_boxes_temp.tolist()

        pred_boxes, pred_confs, pred_labels = utils.cpu_nms(pred_boxes, pred_confs*pred_probs, NUM_CLASSES,
                                                      score_thresh=SCORE_THRESH, iou_thresh=IOU_THRESH)
        break



except tf.errors.OutOfRangeError:
    print("=> finished")





