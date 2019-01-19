#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_tfrecord.py
#   Author      : YunYang1994
#   Created date: 2018-12-18 12:34:23
#   Description :
#
#================================================================

import sys
import argparse
import numpy as np
import tensorflow as tf

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_txt", default='./data/train_data/quick_train_data/quick_train_data.txt')
    parser.add_argument("--tfrecord_path_prefix", default='./data/train_data/quick_train_data/tfrecords/quick_train_data')
    parser.add_argument("--num_tfrecords", default=3, type=int)
    flags = parser.parse_args()

    dataset = {}
    with open(flags.dataset_txt,'r') as f:
        for line in f.readlines():
            example = line.split(' ')
            image_path = example[0]
            boxes_num = len(example[1:]) // 5
            boxes = np.zeros([boxes_num, 5], dtype=np.float32)
            for i in range(boxes_num):
                boxes[i] = example[1+i*5:6+i*5]
            dataset[image_path] = boxes

    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print(">> Processing %d images" %images_num)
    per_tfrecord_images = images_num // flags.num_tfrecords

    n = 0
    while n <= flags.num_tfrecords:
        tfrecord_file = flags.tfrecord_path_prefix+"%04d.tfrecords" % n
        with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
            st = n*per_tfrecord_images
            en = (n+1)*per_tfrecord_images if n < flags.num_tfrecords else len(image_paths)
            for i in range(st, en):
                image = tf.gfile.FastGFile(image_paths[i], 'rb').read()
                boxes = dataset[image_paths[i]]
                boxes = boxes.tostring()

                example = tf.train.Example(features = tf.train.Features(
                    feature={
                        'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                        'boxes' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [boxes])),
                    }
                ))

                record_writer.write(example.SerializeToString())
            print(">> Saving %5d images in %s" %(en-st, tfrecord_file))
            n += 1


if __name__ == "__main__":main(sys.argv[1:])


