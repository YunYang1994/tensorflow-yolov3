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

import numpy as np
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

    def _reorg_layer(self, feature_map, anchors):

        num_anchors = len(anchors) # num_anchors=3
        grid_size = feature_map.shape.as_list()[1:3]

        stride = tf.cast(self.img_size // grid_size, tf.float32)

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)

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

        center_x, center_y, height, width = tf.split(boxes, [1,1,1,1], axis=-1)
        x0 = center_x - height / 2
        y0 = center_y - width  / 2
        x1 = center_x + height / 2
        y1 = center_y + width  / 2

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs

    def compute_loss(self, y_pred, y_true, ignore_thresh=0.5, max_box_per_image=8):
        """
        Note: compute the loss
        Arguments: y_pred, list -> [feature_map_1, feature_map_2, feature_map_3]
                                        the shape of [None, 13, 13, 3*85]. etc
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss, rec_50, rec_75,  avg_iou    = 0., 0., 0., 0.
        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        for i in range(len( y_pred )):
            result = self.loss_layer(y_pred[i], y_true[i], _ANCHORS[i], ignore_thresh, max_box_per_image)
            loss_xy    += result[0]
            loss_wh    += result[1]
            loss_conf  += result[2]
            loss_class += result[3]
            rec_50     += result[4]
            rec_75     += result[5]
            avg_iou    += result[6]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class, rec_50, rec_75, avg_iou]


    def loss_layer(self, feature_map_i, y_true, anchors, ignore_thresh, max_box_per_image):

        NO_OBJECT_SCALE  = 1.0
        OBJECT_SCALE     = 5.0
        COORD_SCALE      = 1.0
        CLASS_SCALE      = 1.0

        grid_size = tf.shape(feature_map_i)[1:3] # [13, 13]
        stride = tf.cast(self.img_size//grid_size, dtype=tf.float32) # [32, 32]

        pred_result = self._reorg_layer(feature_map_i, anchors)

        xy_offset,  pred_boxes, pred_box_conf_logits, pred_box_class_logits = pred_result
        # (13, 13, 1, 2), (1, 13, 13, 3, 4), (1, 13, 13, 3, 1), (1, 13, 13, 3, 80)
        # pred_boxes 前面两个坐标是左上角，后面两个是右下角

        """
        Adjust prediction
        """
        pred_box_conf  = tf.nn.sigmoid(pred_box_conf_logits)                                    # adjust confidence
        pred_box_class = tf.argmax(tf.nn.softmax(pred_box_class_logits), -1)                    # adjust class probabilities
        pred_box_xy = (pred_boxes[..., 0:2] + pred_boxes[..., 2:4]) / 2.                        # absolute coordinate
        pred_box_wh =  pred_boxes[..., 2:4] - pred_boxes[..., 0:2]                               # absolute size

        # 每个cell里都会预测一个boundingbox，y_true里面每个cell里也对应一个
        # boundingbox，那么怎么计算iou呢?

        """
        Adjust ground truth
        """
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        true_box_conf  = y_true[..., 4:5]
        true_box_xy = y_true[..., 0:2]                                                           # absolute coordinate
        true_box_wh = y_true[..., 2:4]                                                           # absolute size
        object_mask = y_true[..., 4:5]

        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0

        """
        Compute some online statistics
        """
        true_mins = true_box_xy - true_box_wh / 2.
        true_maxs = true_box_xy + true_box_wh / 2.
        pred_mins = pred_box_xy - pred_box_wh / 2.
        pred_maxs = pred_box_xy + pred_box_wh / 2.

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxs = tf.minimum(pred_maxs, true_maxs)

        intersect_wh    = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_area  = pred_area + true_area - intersect_area
        iou_scores  = tf.truediv(intersect_area, union_area)

        return object_mask, intersect_area, iou_scores

        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)

        count       = tf.reduce_sum(object_mask)
        detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
        class_mask  = tf.expand_dims(tf.to_float(tf.equal(pred_box_class, true_box_class)), 4)
        recall50    = tf.reduce_mean(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
        recall75    = tf.reduce_mean(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)
        avg_iou     = tf.reduce_mean(iou_scores) / (count + 1e-3)

        """
        Compare each predicted box to all true boxes
        """
        def pick_out_gt_box(y_true):
            y_true = y_true.copy()
            bs = y_true.shape[0]
            # print("=>y_true", y_true.shape)
            true_boxes_batch = np.zeros([bs, 1, 1, 1, max_box_per_image, 4], dtype=np.float32)
            # print("=>true_boxes_batch", true_boxes_batch.shape)
            for i in range(bs):
                y_true_per_layer = y_true[i]
                true_boxes_per_layer = y_true_per_layer[y_true_per_layer[..., 4] > 0][:, 0:4]
                if len(true_boxes_per_layer) == 0: continue
                true_boxes_batch[i][0][0][0][0:len(true_boxes_per_layer)] = true_boxes_per_layer

            return true_boxes_batch

        true_boxes = tf.py_func(pick_out_gt_box, [y_true], [tf.float32] )[0]

        true_xy = true_boxes[..., 0:2]  # absolute location
        true_wh = true_boxes[..., 2:4]  # absolute size


        true_mins = true_xy - true_wh / 2.
        true_maxs = true_xy + true_wh / 2.
        pred_mins = tf.expand_dims(pred_boxes[..., 0:2], axis=4)
        pred_maxs = tf.expand_dims(pred_boxes[..., 2:4], axis=4)
        pred_wh   = pred_maxs - pred_mins

        intersect_mins  = tf.maximum(pred_mins, true_mins)
        intersect_maxs  = tf.minimum(pred_maxs, true_maxs)

        intersect_wh    = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_area = true_wh[..., 0] * true_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]

        union_area = pred_area + true_area - intersect_area
        iou_scores  = tf.truediv(intersect_area, union_area)
        best_ious   = tf.reduce_max(iou_scores, axis=4)
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)

        """
        Compare each true box to all anchor boxes
        """
        ### adjust x and y => relative position to the containing cell
        true_box_xy = true_box_xy / stride  - xy_offset      # t_xy  in `sigma(t_xy) + c_xy`
        pred_box_xy = pred_box_xy / stride  - xy_offset

        ### adjust w and h => relative size to the containing cell
        true_box_wh_logit = true_box_wh / anchors
        pred_box_wh_logit = pred_box_wh / anchors

        true_box_wh_logit = tf.where(condition=tf.equal(true_box_wh_logit,0),
                                     x=tf.ones_like(true_box_wh_logit), y=true_box_wh_logit)
        pred_box_wh_logit = tf.where(condition=tf.equal(pred_box_wh_logit,0),
                                     x=tf.ones_like(pred_box_wh_logit), y=pred_box_wh_logit)

        true_box_wh = tf.log(true_box_wh_logit)              # t_wh in `p_wh*exp(t_wh)`
        pred_box_wh = tf.log(pred_box_wh_logit)

        wh_scale = tf.exp(true_box_wh) * anchors / tf.to_float(self.img_size)
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

        xy_delta    = object_mask   * (pred_box_xy-true_box_xy) * wh_scale * COORD_SCALE
        wh_delta    = object_mask   * (pred_box_wh-true_box_wh) * wh_scale * COORD_SCALE
        conf_delta  = object_mask   * (pred_box_conf-true_box_conf) * OBJECT_SCALE + (1-object_mask) * conf_delta * NO_OBJECT_SCALE
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class_logits), 4) * CLASS_SCALE

        loss_xy    = tf.reduce_mean(tf.square(xy_delta))
        loss_wh    = tf.reduce_mean(tf.square(wh_delta))
        loss_conf  = tf.reduce_mean(tf.square(conf_delta))
        loss_class = tf.reduce_mean(class_delta)

        return  loss_xy, loss_wh, loss_conf, loss_class, recall50, recall75, avg_iou




