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


import common
import tensorflow as tf
slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]



class Darknet53(object):
    """network for performing feature extraction"""

    def __init__(self, inputs):
        self.route_1, self.route_2, self.route_3 = self.network(inputs)
        self.outputs = [self.route_1, self.route_2, self.route_3]

    def _darknet53_block(self, inputs, filters):
        shortcut = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    def network(self, inputs):

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

        route_3 = inputs
        return route_1, route_2, route_3



class Yolov3(object):

    def __init__(self):
        pass


    def _yolo_block(self, inputs, filters):
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    @staticmethod
    def _get_size(shape, data_format):
        if len(shape) == 4: shape = shape[1:]
        return shape[1:3] if data_format == 'NCHW' else shape[0:2]

    def _detection_layer(self, inputs, num_classes, anchors, img_size, data_format):
        num_anchors = len(anchors)
        predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                                stride=1, normalizer_fn=None,
                                activation_fn=None,
                                biases_initializer=tf.zeros_initializer())

        shape = predictions.get_shape().as_list()
        grid_size = self._get_size(shape, data_format)
        dim = grid_size[0] * grid_size[1]
        bbox_attrs = 5 + num_classes

        if data_format == 'NCHW':
            predictions = tf.reshape(
                predictions, [-1, num_anchors * bbox_attrs, dim])
            predictions = tf.transpose(predictions, [0, 2, 1])

        predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

        stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
        anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

        box_centers, box_sizes, confidence, classes = tf.split(
            predictions, [2, 2, 1, num_classes], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)
        confidence = tf.nn.sigmoid(confidence)

        grid_x = tf.range(grid_size[0], dtype=tf.float32)
        grid_y = tf.range(grid_size[1], dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride

        anchors = tf.tile(anchors, [dim, 1])
        box_sizes = tf.exp(box_sizes) * anchors
        box_sizes = box_sizes * stride

        classes_prob = tf.nn.sigmoid(classes)
        feature_map = tf.concat([box_centers, box_sizes, confidence, classes_prob], axis=-1)

        return feature_map

    @staticmethod
    def _upsample(inputs, out_shape, data_format='NCHW'):
        # tf.image.resize_nearest_neighbor accepts input in format NHWC
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        if data_format == 'NCHW':
            new_height = out_shape[3]
            new_width = out_shape[2]
        else:
            new_height = out_shape[2]
            new_width = out_shape[1]

        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

        # back to NCHW if needed
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = tf.identity(inputs, name='upsampled')
        return inputs

    def forward(self, inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
        """
        Creates YOLO v3 model.

        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
            Dimension batch_size may be undefined. The channel order is RGB.
        :param num_classes: number of predicted classes.
        :param is_training: whether is training or not.
        :param data_format: data format NCHW or NHWC.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # it will be needed later on
        img_size = inputs.get_shape().as_list()[1:3]

        # transpose the inputs to NCHW
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # normalize values to range [0..1]
        inputs = inputs / 255

        # set batch norm params
        batch_norm_params = {
            'decay': _BATCH_NORM_DECAY,
            'epsilon': _BATCH_NORM_EPSILON,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding], data_format=data_format, reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, inputs = Darknet53(inputs).outputs

                with tf.variable_scope('yolo-v3'):
                    route, inputs = self._yolo_block(inputs, 512)
                    feature_map_1 = self._detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inputs = common._conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route_2.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size, data_format)
                    inputs = tf.concat([inputs, route_2], axis=1 if data_format == 'NCHW' else 3)

                    route, inputs = self._yolo_block(inputs, 256)
                    feature_map_2 = self._detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inputs = common._conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size, data_format)
                    inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)

                    route, inputs = self._yolo_block(inputs, 128)
                    feature_map_3 = self._detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

                    feature_map = tf.concat([feature_map_1, feature_map_2, feature_map_3], axis=1)
                    feature_map = tf.identity(feature_map, name='feature_map')
                    return feature_map
