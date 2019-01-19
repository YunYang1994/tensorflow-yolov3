#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : extract_coco.py
#   Author      : YunYang1994
#   Created date: 2018-12-07 15:12:25
#   Description :
#
#================================================================

import os
import sys
import json
import argparse
from collections import defaultdict

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default='/home/yang/test/COCO/annotations/instances_train2017.json')
    parser.add_argument("--image_path", default="/home/yang/test/COCO/train2017")
    parser.add_argument("--dataset_info_path", default="./data/train2017.txt")
    flags = parser.parse_args()

    dataset = defaultdict(list)
    with open(os.path.realpath(flags.dataset_info_path), 'w') as f:
        labels = json.load(open(flags.json_path, encoding='utf-8'))
        annotations = labels['annotations']

        for annotation in annotations:
            image_id = annotation['image_id']
            image_folder_path = os.path.realpath(flags.image_path)
            single_image_path = os.path.join(image_folder_path, '%012d.jpg' %image_id)
            category_id = annotation['category_id']

            if category_id >=  1 and category_id <= 11: category_id = category_id - 1
            if category_id >= 13 and category_id <= 25: category_id = category_id - 2
            if category_id >= 27 and category_id <= 28: category_id = category_id - 3
            if category_id >= 31 and category_id <= 44: category_id = category_id - 5
            if category_id >= 46 and category_id <= 65: category_id = category_id - 6
            if category_id == 67: category_id = category_id - 7
            if category_id == 70: category_id = category_id - 9
            if category_id >= 72 and category_id <= 82: category_id = category_id - 10
            if category_id >= 84 and category_id <= 90: category_id = category_id - 11

            x_min, y_min, width, height = annotation['bbox']
            x_max = x_min+width
            y_max = y_min+height
            box = [x_min, y_min, x_max, y_max]
            dataset[single_image_path].append([category_id, box])

        for single_image_path in dataset.keys():
            write_content = [single_image_path]
            for category_id, box in dataset[single_image_path]:
                x_min, y_min, x_max, y_max = box
                write_content.append(str(x_min))
                write_content.append(str(y_min))
                write_content.append(str(x_max))
                write_content.append(str(y_max))
                write_content.append(str(category_id))
            write_content = " ".join(write_content)
            print(write_content)
            f.write(write_content+'\n')



if __name__ == "__main__": main(sys.argv[1:])



