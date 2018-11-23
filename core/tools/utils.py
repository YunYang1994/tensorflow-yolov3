#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2018-11-22 12:02:52
#   Description :
#
#================================================================

import colorsys
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw


def get_boxes_confs_probs(feature_map):
    """
    :param feature_map: outputs of YOLOv3 network of shape [?, 10647, num_classes+5]
                        prediction in three scale -> (52×52+26×26+ 13×13)×3 = 10647
    :return
            boxes -- tensor of shape [None, 10647, 4], containing (x0, y0, x1, y1)
                    coordinates of selected boxes
            confidence -- tensor of shape [None, 10647, 1]
            probability -- tensor of shape [None, 10647, num_classes]
    """

    box_info, confidence, probability = tf.split(
                                             feature_map, [4, 1, -1], axis=-1)
    center_x, center_y, width, height = tf.split(box_info, [1,1,1,1], axis=-1)
    x0 = center_x - width  / 2
    y0 = center_y - height / 2
    x1 = center_x + width  / 2
    y1 = center_y + height / 2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    return boxes, confidence, probability

# Discard all boxes with low scores
def filter_boxes(boxes, confs, probs, score_threshold=0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    Arguments:
    box_confidence -- tensor of shape [None, 10647, 1]
    boxes -- tensor of shape [None, 10647, 4]
    box_class_probs -- tensor of shape [None, 10647, 4]
    threshold -- real value, if [ highest class probability score < threshold]
                             then get rid of the corresponding box
    Returns:
    scores -- tensor of shape [None, num_classes],
                containing the class probability score for selected boxes
    boxes --  tensor of shape [None, 4],
                containing coordinates of selected boxes
    probs --  tensor of shape [None, num_classes],
                containing the index of the class detected by the selected boxes
    Note: "None" is here because you don't know the exact number of how many boxes
           are selected
    """

    # Step 1: Compute box scores
    scores = confs * probs #multiply box probability(p) with class probability

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of
    # the corresponding score
    max_box_scores = tf.reduce_max(scores, axis=-1) # box_scores of shape [None, 10647, num_classes]

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold".
    # The mask should have the same dimension as box_class_scores, and be True for the
    # boxes you want to keep (with probability >= threshold)
    mask = tf.greater_equal(max_box_scores, tf.constant(score_threshold))

    # Step 4: Apply the mask to scores, boxes and probs and pick them out
    boxes = tf.boolean_mask(boxes, mask)
    probs = tf.boolean_mask(probs, mask)
    scores = tf.boolean_mask(scores, mask)

    return boxes, scores, probs

# Non-max suppression
# TODO queding weidu
def nms(boxes, scores, probs, max_boxes=20, iou_threshold=0.5):
    """
    Note:
    Applies Non-max suppression (NMS) to set of boxes. Prunes away boxes that have high
    intersection-over-union (IOU) overlap with previously selected boxes.

    Arguments:
            scores -- tensor of shape [None, num_classes]
            boxes  -- tensor of shape [None, 4]
            probs  -- tensor of shape [None, num_classes]
            max_boxes -- integer, maximum number of predicted boxes you'd like, default is 20
            iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
            bboxes -- tensor of shape [None, 4], predicted bbox coordinates
            scores -- tensor of shape [None, 1], predicted score for each box
            class_ -- tensor of shape [None, 1], predicted class for each box
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes.
    """

    max_boxes = tf.constant(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes=boxes,
                                            scores=tf.reduce_max(scores, axis=-1),
                                            max_output_size=max_boxes,
                                            iou_threshold=iou_threshold, name='nms_indices')

    boxes = tf.gather(boxes, nms_indices)
    probs = tf.gather(probs, nms_indices)
    scores = tf.gather(scores, nms_indices)
    scores = tf.reduce_max(scores, axis=-1)
    labels = tf.argmax(probs, axis=-1)

    return boxes, scores, labels

##### draw bounding box #####
def draw_boxes(boxes, scores, labels, image, classes, show=True):

    detection_size = [416, 416]
    draw = ImageDraw.Draw(image)

    # draw settings
    font = ImageFont.truetype(font = '../data/font/FiraMono-Medium.otf',
                              size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    hsv_tuples = [( x / len(classes), 0.8, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # thickness = (image.size[0] + image.size[1]) // 300

    for i in range(len(labels)): # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" %(label, score)
        text_size = draw.textsize(bbox_text, font)
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2,2) * ratio).reshape(-1))

        draw.rectangle(bbox, outline=colors[labels[i]], width=3)
        draw.rectangle([tuple(bbox[:2]), tuple(np.array(bbox[:2])+text_size)], fill=colors[labels[i]])

        # # draw bbox
        draw.text(bbox[:2], bbox_text, fill=(0,0,0), font=font)

    del draw
    if show: image.show()
    return image

