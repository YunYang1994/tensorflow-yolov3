#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_weight.py
#   Author      : YunYang1994
#   Created date: 2018-11-27 12:37:22
#   Description :
#
#================================================================

import os
import sys
import wget
import time
import argparse
import tensorflow as tf
from core import yolov3, utils


class parser(argparse.ArgumentParser):

    def __init__(self,description):
        super(parser, self).__init__(description)

        self.add_argument(
            "--ckpt_file", "-cf", default='./checkpoint/yolov3.ckpt', type=str,
            help="[default: %(default)s] The checkpoint file ...",
            metavar="<CF>",
        )

        self.add_argument(
            "--weights_path", "-wp", default='./checkpoint/yolov3.weights', type=str,
            help="[default: %(default)s] Download binary file with desired weights",
            metavar="<WP>",
        )

        self.add_argument(
            "--convert", "-cv", action='store_true',
            help="[default: %(default)s] Downloading yolov3 weights and convert them",
        )

        self.add_argument(
            "--freeze", "-fz", action='store_true',
            help="[default: %(default)s] freeze the yolov3 graph to pb ...",
        )

        self.add_argument(
            "--image_size", "-is", default=416, type=int,
            help="[default: %(default)s] The image size, 416 or 608",
            metavar="<IS>",
        )

        self.add_argument(
            "--iou_threshold", "-it", default=0.5, type=float,
            help="[default: %(default)s] The iou_threshold for gpu nms",
            metavar="<IT>",
        )

        self.add_argument(
            "--score_threshold", "-st", default=0.5, type=float,
            help="[default: %(default)s] The score_threshold for gpu nms",
            metavar="<ST>",
        )


def main(argv):

    flags = parser(description="freeze yolov3 graph from checkpoint file").parse_args()
    classes = utils.read_coco_names("./data/coco.names")
    num_classes = len(classes)
    SIZE = flags.image_size
    print("=> the input image size is [%d, %d]" %(SIZE, SIZE))
    model = yolov3.yolov3(num_classes)

    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        inputs = tf.placeholder(tf.float32, [1, SIZE, SIZE, 3]) # placeholder for detector inputs

        with tf.variable_scope('yolov3'):
            feature_map = model.forward(inputs, is_training=False)

        boxes, confs, probs = model.predict(feature_map)
        scores = confs * probs
        print("=>", boxes, scores)
        boxes, scores, labels = utils.gpu_nms(boxes, scores, num_classes,
                                              score_thresh=flags.score_threshold,
                                              iou_thresh=flags.iou_threshold)
        print("=>", boxes, scores, labels)
        feature_map_1, feature_map_2, feature_map_3 = feature_map
        print("=>", feature_map_1, feature_map_2, feature_map_3)
        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

        if flags.convert:
            if not os.path.exists(flags.weights_path):
                url = 'https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3.weights'
                for i in range(3):
                    time.sleep(1)
                    print("=> %s does not exists ! " %flags.weights_path)
                print("=> It will take a while to download it from %s" %url)
                print('=> Downloading yolov3 weights ... ')
                wget.download(url, flags.weights_path)

            load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), flags.weights_path)
            sess.run(load_ops)
            save_path = saver.save(sess, save_path=flags.ckpt_file)
            print('=> model saved in path: {}'.format(save_path))

        if flags.freeze:
            saver.restore(sess, flags.ckpt_file)
            print('=> checkpoint file restored from ', flags.ckpt_file)
            utils.freeze_graph(sess, './checkpoint/yolov3_cpu_nms.pb', ["concat_9", "mul_6"])
            utils.freeze_graph(sess, './checkpoint/yolov3_gpu_nms.pb', ["concat_10", "concat_11", "concat_12"])
            utils.freeze_graph(sess, './checkpoint/yolov3_feature.pb', ["yolov3/yolo-v3/feature_map_1",
                                                                        "yolov3/yolo-v3/feature_map_2",
                                                                        "yolov3/yolo-v3/feature_map_3",])


if __name__ == "__main__": main(sys.argv)



