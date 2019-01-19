#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-01-19 22:43:26
#   Description :
#
#================================================================

import numpy as np
from core import utils
import tensorflow as tf

class Parser(object):
    def __init__(self, image_h, image_w, anchors, num_classes, debug=False):
        self.anchors     = anchors
        self.num_classes = num_classes
        self.image_h     = image_h
        self.image_w     = image_w
        self.debug       = debug

    def preprocess(self, image, gt_boxes):
        # resize_image_correct_bbox
        image, gt_boxes = utils.resize_image_correct_bbox(image, gt_boxes, self.image_h, self.image_w)
        if self.debug: return image, gt_boxes

        image = image / 255
        y_true_13, y_true_26, y_true_52 = tf.py_func(self.preprocess_true_boxes, inp=[gt_boxes],
                            Tout = [tf.float32, tf.float32, tf.float32])
        # data augmentation
        # pass

        return image, y_true_13, y_true_26, y_true_52

    def preprocess_true_boxes(self, gt_boxes):
        """
        Preprocess true boxes to training input format
        Parameters:
        -----------
        :param true_boxes: numpy.ndarray of shape [T, 4]
                            T: the number of boxes in each image.
                            4: coordinate => x_min, y_min, x_max, y_max
        :param true_labels: class id
        :param input_shape: the shape of input image to the yolov3 network, [416, 416]
        :param anchors: array, shape=[9,2], 9: the number of anchors, 2: width, height
        :param num_classes: integer, for coco dataset, it is 80
        Returns:
        ----------
        y_true: list(3 array), shape like yolo_outputs, [13, 13, 3, 85]
                            13:cell szie, 3:number of anchors
                            85: box_centers, box_sizes, confidence, probability
        """
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
        grid_sizes = [[self.image_h//x, self.image_w//x] for x in (32, 16, 8)]

        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2 # the center of box
        box_sizes =    gt_boxes[:, 2:4] - gt_boxes[:, 0:2] # the height and width of box

        gt_boxes[:, 0:2] = box_centers
        gt_boxes[:, 2:4] = box_sizes

        y_true_13 = np.zeros(shape=[grid_sizes[0][1], grid_sizes[0][0], 3, 5+self.num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[grid_sizes[1][1], grid_sizes[1][0], 3, 5+self.num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[grid_sizes[2][1], grid_sizes[2][0], 3, 5+self.num_classes], dtype=np.float32)

        y_true = [y_true_13, y_true_26, y_true_52]
        anchors_max =  self.anchors / 2.
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
        box_area = wh[..., 0] * wh[..., 1]

        anchor_area = self.anchors[:, 0] * self.anchors[:, 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n not in anchor_mask[l]: continue

                i = np.floor(gt_boxes[t,0]/self.image_w*grid_sizes[l][1]).astype('int32')
                j = np.floor(gt_boxes[t,1]/self.image_h*grid_sizes[l][0]).astype('int32')

                k = anchor_mask[l].index(n)
                c = gt_boxes[t, 4].astype('int32')

                y_true[l][i, j, k, 0:4] = gt_boxes[t, 0:4]
                y_true[l][i, j, k,   4] = 1
                y_true[l][i, j, k, 5+c] = 1

        return y_true_13, y_true_26, y_true_52

    def parser_example(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image' : tf.FixedLenFeature([], dtype = tf.string),
                'boxes' : tf.FixedLenFeature([], dtype = tf.string),
            }
        )

        image = tf.image.decode_jpeg(features['image'], channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        gt_boxes = tf.decode_raw(features['boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, shape=[-1,5])

        return self.preprocess(image, gt_boxes)

class dataset(object):
    def __init__(self, parser, tfrecords_path, batch_size, shuffle=None):
        self.parser = parser
        self.filenames = tf.gfile.Glob(tfrecords_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._buildup()

    def _buildup(self):
        try:
            self._TFRecordDataset = tf.data.TFRecordDataset(self.filenames)
        except:
            raise NotImplementedError("No tfrecords found!")

        self._TFRecordDataset = self._TFRecordDataset.map(map_func = self.parser.parser_example,
                                                        num_parallel_calls = 10).repeat()
        if self.shuffle is not None:
            self._TFRecordDataset = self._TFRecordDataset.shuffle(self.shuffle)
        self._TFRecordDataset = self._TFRecordDataset.batch(self.batch_size).prefetch(self.batch_size)
        self._iterator = self._TFRecordDataset.make_one_shot_iterator()

    def get_next(self):
        return self._iterator.get_next()
