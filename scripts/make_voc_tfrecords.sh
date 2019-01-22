#! /bin/bash

#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   
#   Editor      : VIM 
#   File name   : make_voc_tfrecords.sh
#   Author      : YunYang1994
#   Created date: 2019-01-21 11:40:10
#   Description : 
#
#================================================================

python scripts/extract_voc.py --voc_path /home/yang/test/VOC/train/ --dataset_info_path ./
cat ./2007_train.txt ./2007_val.txt > voc_train.txt
python scripts/extract_voc.py --voc_path /home/yang/test/VOC/test/ --dataset_info_path ./
cat ./2007_test.txt > voc_test.txt
python core/convert_tfrecord.py --dataset_txt ./voc_train.txt --tfrecord_path_prefix /home/yang/test/VOC/train/voc_train
python core/convert_tfrecord.py --dataset_txt ./voc_test.txt  --tfrecord_path_prefix /home/yang/test/VOC/test/voc_test
