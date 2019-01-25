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
from core import common
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

    def __init__(self, num_classes, anchors,
                 batch_norm_decay=0.9, leaky_relu=0.1):

        # self._ANCHORS = [[10 ,13], [16 , 30], [33 , 23],
                         # [30 ,61], [62 , 45], [59 ,119],
                         # [116,90], [156,198], [373,326]]
        self._ANCHORS = anchors
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

    def _reorg_layer(self, feature_map, anchors):

        num_anchors = len(anchors) # num_anchors=3
        grid_size = feature_map.shape.as_list()[1:3]
        # the downscale image in height and weight
        stride = tf.cast(self.img_size // grid_size, tf.float32) # [h,w] -> [y,x]
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset   = tf.reshape(a, (-1, 1))
        y_offset   = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride[::-1]

        box_sizes = tf.exp(box_sizes) * anchors # anchors -> [w, h]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

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

        results = [self._reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            boxes, conf_logits, prob_logits = self._reshape(*result)

            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1,1,1,1], axis=-1)
        x0 = center_x - width   / 2.
        y0 = center_y - height  / 2.
        x1 = center_x + width   / 2.
        y1 = center_y + height  / 2.

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs

    def compute_loss(self, pred_feature_map, y_true, ignore_thresh=0.5, max_box_per_image=8):
        """
        Note: compute the loss
        Arguments: y_pred, list -> [feature_map_1, feature_map_2, feature_map_3]
                                        the shape of [None, 13, 13, 3*85]. etc
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss = 0.
        # total_loss, rec_50, rec_75,  avg_iou    = 0., 0., 0., 0.
        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        for i in range(len(pred_feature_map)):
            result = self.loss_layer(pred_feature_map[i], y_true[i], _ANCHORS[i])
            loss_xy    += result[0]
            loss_wh    += result[1]
            loss_conf  += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]


    def loss_layer(self, feature_map_i, y_true, anchors):
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        grid_size_ = feature_map_i.shape.as_list()[1:3]

        y_true = tf.reshape(y_true, [-1, grid_size_[0], grid_size_[1], 3, 5+self._NUM_CLASSES])

        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self._reorg_layer(feature_map_i, anchors)
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N, 13, 13, 3, V]
        iou = self._broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        best_iou = tf.reduce_max(iou, axis=-1)
        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy      / ratio[::-1] - x_y_offset

        # get_tw_th, numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh      / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)

        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy    - pred_xy) * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # shape: [N, 13, 13, 3, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area  = pred_box_wh[..., 0]  * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area  = true_box_wh[..., 0]  * true_box_wh[..., 1]
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou



