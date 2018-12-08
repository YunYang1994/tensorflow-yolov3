#! /usr/bin/env python3
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2018-11-21 18:41:35
#   Description : YOLOv3: An Incremental Improvement
#
#================================================================

import tensorflow as tf
from core import common, utils
slim = tf.contrib.slim

class darknet53(object):
    """network for performing feature extraction"""

    def __init__(self, inputs):
        self.outputs = self.forward(inputs)

    def _darknet53_block(self, inputs, filters):
        """
        implement residuals block in darknet53
        """
        shortcut = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    def forward(self, inputs):

        inputs = common._conv2d_fixed_padding(inputs, 32,  3, strides=1)
        inputs = common._conv2d_fixed_padding(inputs, 64,  3, strides=2)
        inputs = self._darknet53_block(inputs, 32)
        inputs = common._conv2d_fixed_padding(inputs, 128, 3, strides=2)

        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)

        inputs = common._conv2d_fixed_padding(inputs, 256, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)

        route_1 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 512, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        route_2 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)

        return route_1, route_2, inputs



class yolov3(object):

    def __init__(self, num_classes=80,
                 batch_norm_decay=0.9, leaky_relu=0.1, anchors_path='./data/yolo_anchors.txt'):

        # self._ANCHORS = [[10 ,13], [16 , 30], [33 , 23],
                         # [30 ,61], [62 , 45], [59 ,119],
                         # [116,90], [156,198], [373,326]]
        self._ANCHORS = utils.get_anchors(anchors_path)
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self._NUM_CLASSES = num_classes
        self.feature_maps = [] # [[None, 13, 13, 255], [None, 26, 26, 255], [None, 52, 52, 255]]

    def _yolo_block(self, inputs, filters):
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)
        feature_map = slim.conv2d(inputs, num_anchors * (5 + self._NUM_CLASSES), 1,
                                stride=1, normalizer_fn=None,
                                activation_fn=None,
                                biases_initializer=tf.zeros_initializer())
        return feature_map

    def get_boxes_confs_scores(self, feature_map, anchors):

        num_anchors = len(anchors) # num_anchors=3
        grid_size = tf.shape(feature_map)[1:3]

        stride = (self.img_size[0] // grid_size[0], self.img_size[1] // grid_size[1])
        anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])

        box_centers, box_sizes, confs, probs = tf.split(
            feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)
        confs = tf.nn.sigmoid(confs)
        probs = tf.nn.sigmoid(probs)

        grid_x = tf.range(grid_size[0], dtype=tf.int32)
        grid_y = tf.range(grid_size[1], dtype=tf.int32)
        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride

        box_sizes = tf.exp(box_sizes) * anchors
        box_sizes = box_sizes * stride

        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, confs, probs

    @staticmethod
    def _upsample(inputs, out_shape):

        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        inputs = tf.identity(inputs, name='upsampled')

        return inputs

    # @staticmethod
    # def _upsample(inputs, out_shape):
        # """
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT 5 optimization
        # """
        # new_height, new_width = out_shape[1], out_shape[2]
        # filters = 256 if (new_height == 26 and new_width==26) else 128
        # inputs = tf.layers.conv2d_transpose(inputs, filters, kernel_size=3, padding='same',
                                            # strides=(2,2), kernel_initializer=tf.ones_initializer())
        # return inputs

    def forward(self, inputs, is_training=False, reuse=False):
        """
        Creates YOLO v3 model.

        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
               Dimension batch_size may be undefined. The channel order is RGB.
        :param is_training: whether is training or not.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # it will be needed later on
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding],reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, inputs = darknet53(inputs).outputs

                with tf.variable_scope('yolo-v3'):
                    route, inputs = self._yolo_block(inputs, 512)
                    feature_map_1 = self._detection_layer(inputs, self._ANCHORS[6:9])
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inputs = common._conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route_2.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)

                    route, inputs = self._yolo_block(inputs, 256)
                    feature_map_2 = self._detection_layer(inputs, self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inputs = common._conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3)

                    route, inputs = self._yolo_block(inputs, 128)
                    feature_map_3 = self._detection_layer(inputs, self._ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def _reshape(self, x_y_offset, boxes, confs, probs):

        grid_size = x_y_offset.shape.as_list()[:2]
        boxes = tf.reshape(boxes, [-1, grid_size[0]*grid_size[1]*3, 4])
        confs = tf.reshape(confs, [-1, grid_size[0]*grid_size[1]*3, 1])
        probs = tf.reshape(probs, [-1, grid_size[0]*grid_size[1]*3, self._NUM_CLASSES])

        return boxes, confs, probs

    def predict(self, feature_maps):
        """
        Note: given by feature_maps, compute the receptive field
              and get boxes, confs and class_probs
        input_argument: feature_maps -> [None, 13, 13, 255],
                                        [None, 26, 26, 255],
                                        [None, 52, 52, 255],
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                               (feature_map_2, self._ANCHORS[3:6]),
                               (feature_map_3, self._ANCHORS[0:3]),]

        results = [self.get_boxes_confs_scores(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            boxes, confs, probs = self._reshape(*result)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1,1,1,1], axis=-1)
        x0 = center_x - width  / 2
        y0 = center_y - height / 2
        x1 = center_x + width  / 2
        y1 = center_y + height / 2

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs

    def compute_loss(self, feature_map, boxes_true, ignore_thresh=0.5):
        """
        Note: compute the loss
        Arguments: feature_map, list -> [feature_map_1, feature_map_2, feature_map_3]
                                        the shape of [None, 13, 13, 3*85]. etc
        """
        loss = 0.
        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        for i in range(3):
            grid_size = tf.shape(feature_map[i])[1:3]
            object_mask = boxes_true[i][..., 4:5]
            class_probs = boxes_true[i][..., 5:]
            grid, boxes_pred, confs_pred, probs_pred = self.get_boxes_confs_scores(
                                                        feature_map=feature_map[i],
                                                        anchors=_ANCHORS[i])
            grid = tf.cast(grid, tf.float32)
            pred_xy = boxes_pred[...,  :2] / tf.cast(self.img_size[::-1], tf.float32)
            pred_wh = boxes_pred[..., 2:4] / tf.cast(self.img_size[::-1], tf.float32)
            predictions = tf.reshape(feature_map[i],
                                     [-1, grid_size[0], grid_size[1], 3, 5 + self._NUM_CLASSES])
            pred_box = tf.concat([pred_xy, pred_wh], axis = -1)
            raw_true_xy = boxes_true[i][..., :2] * tf.cast(grid_size[::-1], tf.float32) - grid
            object_mask_bool = tf.cast(object_mask, dtype = tf.bool)
            raw_true_wh = tf.log(tf.where(tf.equal(boxes_true[i][..., 2:4] / _ANCHORS[i] * tf.cast(self.img_size[::-1], tf.float32), 0),
                                        tf.ones_like(boxes_true[i][..., 2:4]), boxes_true[i][..., 2:4] / _ANCHORS[i] * tf.cast(self.img_size[::-1], tf.float32)))

            box_loss_scale = 2 - boxes_true[i][..., 2:3] * boxes_true[i][..., 3:4]
            ignore_mask = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)

            def loop_body(internal_index, ignore_mask):

                true_box = tf.boolean_mask(boxes_true[i][internal_index, ..., 0:4], object_mask_bool[internal_index, ..., 0])
                iou = utils.box_iou(pred_box[internal_index], true_box)

                best_iou = tf.reduce_max(iou, axis = -1)
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
                return internal_index + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda internal_index, ignore_mask : internal_index < tf.shape(feature_map[0])[0], loop_body, [0, ignore_mask])

            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis = -1)

            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels = raw_true_xy, logits = predictions[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) * ignore_mask
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels =  class_probs, logits = predictions[..., 5:])
            xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(feature_map[0])[0], tf.float32)
            wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(feature_map[0])[0], tf.float32)
            confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(feature_map[0])[0], tf.float32)
            class_loss = tf.reduce_sum(class_loss) / tf.cast(tf.shape(feature_map[0])[0], tf.float32)

            loss += xy_loss + wh_loss + confidence_loss + class_loss

        return loss
