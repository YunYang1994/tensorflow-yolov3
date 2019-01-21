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

python scripts/extract_voc.py
cat scripts/VOCdevkit_2012.txt | head -n   10000 > ./scripts/train.txt
cat scripts/VOCdevkit_2012.txt | tail -n  +10000 > ./scripts/test.txt
