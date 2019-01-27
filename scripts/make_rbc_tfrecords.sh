#! /bin/bash

#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   
#   Editor      : VIM 
#   File name   : make_rbc_tfrecords.sh
#   Author      : YunYang1994
#   Created date: 2019-01-28 00:42:52
#   Description : 
#
#================================================================

git clone https://github.com/YunYang1994/red_blood_cell_dataset.git
cat ./red_blood_cell_dataset/labels.txt | head -n  300 > ./red_blood_cell_dataset/train.txt
cat ./red_blood_cell_dataset/labels.txt | tail -n +301 > ./red_blood_cell_dataset/test.txt
python core/convert_tfrecord.py --dataset_txt ./red_blood_cell_dataset/train.txt --tfrecord_path_prefix ./red_blood_cell_dataset/rbc_train
python core/convert_tfrecord.py --dataset_txt ./red_blood_cell_dataset/test.txt  --tfrecord_path_prefix ./red_blood_cell_dataset/rbc_test
