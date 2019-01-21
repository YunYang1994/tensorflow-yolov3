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

python scripts/extract_voc.py --voc_data_path /home/yang/VOC/train/ --dataset_txt scripts/voc_train.txt
python scripts/extract_voc.py --voc_data_path /home/yang/VOC/test/  --dataset_txt scripts/voc_test.txt
python core/convert_tfrecord.py --dataset_txt scripts/voc_train.txt --tfrecord_path_prefix /home/yang/VOC/train/voc_train
python core/convert_tfrecord.py --dataset_txt scripts/voc_test.txt  --tfrecord_path_prefix /home/yang/VOC/test/voc_test
