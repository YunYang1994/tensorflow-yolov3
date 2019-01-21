#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : extract_voc.py
#   Author      : YunYang1994
#   Created date: 2019-01-05 00:14:10
#   Description :
#
#================================================================

import os
import argparse
import xml.etree.ElementTree as ET


LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

train_image_folder = "/home/yang/test/VOCdevkit/VOC2012/JPEGImages/"
train_annot_folder = "/home/yang/test/VOCdevkit/VOC2012/Annotations/"


def parse_annotation(ann_dir, img_dir, labels=[]):
    '''
    output:
    - Each element of the train_image is a dictionary containing the annoation infomation of an image.
    - seen_train_labels is the dictionary containing
            (key, value) = (the object class, the number of objects found in the images)
    '''
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object':[]}
        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                path_to_image = img_dir + elem.text
                img['filename'] = path_to_image
                ## make sure that the image exists:
                if not os.path.exists(path_to_image):
                    assert False, "file does not exist!\n{}".format(path_to_image)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:

                        obj['name'] = attr.text

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']]  = 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0: all_imgs += [img]

    return all_imgs, seen_labels

dataset = parse_annotation(train_annot_folder, train_image_folder)
all_imgs = dataset[0]
write_lines = []

with open('VOCdevkit_2012.txt', 'w') as f:
    for all_img in all_imgs:
        image_path = all_img['filename']
        objects    = all_img['object']
        line       = [image_path]
        for obj in objects:
            if obj['name'] not in LABELS: continue

            line.append(str(obj['xmin']))
            line.append(str(obj['ymin']))
            line.append(str(obj['xmax']))
            line.append(str(obj['ymax']))
            line.append(str(LABELS.index(obj['name'])))

        line = " ".join(line) + "\n"
        f.writelines(line)
