#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : kangaroo.py
#   Author      : YunYang1994
#   Created date: 2019-01-07 13:36:29
#   Description :
#
#================================================================

import os
import pickle
import xml.etree.ElementTree as ET

def parse_voc_annotation(ann_dir, img_dir, dataset_txt, labels=[]):

    all_insts = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}

        try:
            tree = ET.parse(ann_dir + ann)
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + ann_dir + ann)
            continue

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

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

        if len(img['object']) > 0:
            all_insts += [img]

    # cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
    # with open(cache_name, 'wb') as handle:
        # pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dataset_txt, 'w') as f:
        for all_inst in all_insts:
            image_path = all_inst['filename']
            write_content = [image_path]
            for image_info in all_inst['object']:
                # ID = seen_labels[image_info['name']]
                xmin = image_info['xmin']
                ymin = image_info['ymin']
                xmax = image_info['xmax']
                ymax = image_info['ymax']

                write_content += ['0', str(xmin), str(ymin), str(xmax), str(ymax)]

            write_content = " ".join(write_content) + '\n'
            f.writelines(write_content)

    return all_insts, seen_labels

ann_dir = "/home/yang/test/kangaroo/annots/"
img_dir = "/home/yang/test/kangaroo/images/"

data = parse_voc_annotation(ann_dir, img_dir, "./kangaroo.txt")


