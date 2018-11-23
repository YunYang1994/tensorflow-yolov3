# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import yolov3
from utils import load_coco_names, draw_boxes, detections_boxes, non_max_suppression

# model = yolo_v3.yolo_v3
model = yolov3.Yolov3()

img = Image.open('../data/dog.jpg')
img_resized = img.resize(size=(416, 416))

classes = load_coco_names('../data/coco.names')

# placeholder for detector inputs
inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])

with tf.variable_scope('detector'):
    feature_map = model.forward(inputs, len(classes),
                        data_format='NCHW')

saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

boxes = detections_boxes(feature_map)
# boxes_gt, confidence, probability = get_boxes_conf_probs(feature_map)

with tf.Session() as sess:
    saver.restore(sess, './saved_model/yolov3.ckpt')
    print('Model restored.')

    detected_boxes = sess.run(
        boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
print(detected_boxes)
filtered_boxes = non_max_suppression(detected_boxes,
                                        confidence_threshold=0.5,
                                        iou_threshold=0.4)

draw_boxes(filtered_boxes, img, classes, (416, 416))
# draw_boxes(detected_boxes, img, classes, (416, 416))

img.save('./dog_result.jpg')




