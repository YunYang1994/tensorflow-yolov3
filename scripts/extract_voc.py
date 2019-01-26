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

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2012', 'train'), ('2012', 'val')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument("--voc_path", default="/home/yang/test/VOCdevkit/train/")
parser.add_argument("--dataset_info_path", default="./")
flags = parser.parse_args()

def convert_annotation(year, image_id, list_file):
    xml_path = os.path.join(flags.voc_path, 'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open(xml_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " +  " ".join([str(a) for a in b]) + " " + str(cls_id))


for year, image_set in sets:
    text_path = os.path.join(flags.voc_path, 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set))
    if not os.path.exists(text_path): continue
    image_ids = open(text_path).read().strip().split()
    list_file_path = os.path.join(flags.dataset_info_path, '%s_%s.txt'%(year, image_set))
    list_file = open(list_file_path, 'w')
    for image_id in image_ids:
        image_path = os.path.join(flags.voc_path, 'VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
        print("=>", image_path)
        list_file.write(image_path)
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()


