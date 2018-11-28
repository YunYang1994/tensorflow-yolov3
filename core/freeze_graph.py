#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2018-11-27 12:37:22
#   Description :
#
#================================================================

import sys
import utils
import yolov3
import argparse
import tensorflow as tf


class parser(argparse.ArgumentParser):

    def __init__(self,description):
        super(parser, self).__init__(description)

        self.add_argument(
            "--gpu_nms_pb", "-gnp", default='./checkpoint/yolov3_gpu_nms.pb',
            help="[default: %(default)s] The location of a Frozen Graph (non_max_suppression on GPU),"
                                         "ie ../checkpoint/yolov3_gpu_nms.pb ",
            metavar="<GNP>",
        )

        self.add_argument(
            "--cpu_nms_pb", "-cnp", default='./checkpoint/yolov3_cpu_nms.pb',
            help="[default: %(default)s] The location of a Frozen Graph (non_max_suppression on CPU),"
                                         "ie ../checkpoint/yolov3_cpu_nms.pb ",
            metavar="<CNP>",
        )

        self.add_argument(
            "--labels_file", "-lf", default="./data/coco.names",
            help="[default: %(default)s] The location of a labels_file ",
            metavar="<LF>",
        )
        
        self.add_argument(
            "--gpu_output_node_names", "-gn", default=["concat_1", "concat_2", "concat_3"],
            help="[default: %(default)s] The output node names, list",
            metavar="<GN>",
        )

        self.add_argument(
            "--cpu_output_node_names", "-cn", default=["concat", "mul"],
            help="[default: %(default)s] The output node names, list",
            metavar="<CN>",
        )

        self.add_argument(
            "--ckpt_file", "-cf", default='./checkpoint/yolov3.ckpt',
            help="[default: %(default)s] The checkpoint path, ie ../checkpoint/yolov3.ckpt",
            metavar="<CF>",
        )


        self.add_argument(
            "--iou_threshold",   "-it", default=0.5, type=float,
            help="[default: %(default)s] The iou_threshold for gpu nms",
            metavar="<IT>",
        )

        self.add_argument(
            "--score_threshold", "-st", default=0.4, type=float,
            help="[default: %(default)s] The iou_threshold for gpu nms",
            metavar="<ST>",
        )


def main(argv):

    flags = parser(description="freeze yolov3 graph from checkpoint file").parse_args()
    classes = utils.get_classes(flags.labels_file)
    num_classes = len(classes)
    model = yolov3.yolov3(num_classes)

    with tf.Graph().as_default() as graph:

        sess = tf.Session(graph=graph)
        # placeholder for detector inputs
        inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])

        with tf.variable_scope('detector'):
            feature_map = model.forward(inputs)

        boxes, scores = utils.get_boxes_scores(feature_map)
        print("=>", boxes, scores)

        boxes, scores, labels = utils.gpu_nms(boxes, scores, num_classes, 20, flags.score_threshold, flags.iou_threshold)
        print("=>", boxes, scores, labels)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
        saver.restore(sess, flags.ckpt_file)
        print('=> checkpoint file restored.')

        utils.freeze_graph(sess, flags.gpu_nms_pb, flags.gpu_output_node_names)
        utils.freeze_graph(sess, flags.cpu_nms_pb, flags.cpu_output_node_names)
if __name__ == "__main__": main(sys.argv)


