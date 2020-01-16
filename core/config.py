#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description : configurations
#
#================================================================

from easydict import EasyDict as edict
import os
import settings
settings.RDT_GIT_ROOT
YOLO_MODEL_VER=1
__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
if YOLO_MODEL_VER==1:
    __C.YOLO.CLASSES                = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/data/classes/rdt.names"
    __C.YOLO.ANCHORS                = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/data/anchors/rdt_anchors.txt"
elif YOLO_MODEL_VER>=2:
    __C.YOLO.CLASSES                = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/data/classes/rdt_rot.names"
    __C.YOLO.ANCHORS                = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/data/anchors/rdt_anchors_rot.txt"

__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.7
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/checkpoint/yolov3_coco_demo.ckpt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/data/dataset/rdt_train.txt"
__C.TRAIN.BATCH_SIZE            = 1
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-8
__C.TRAIN.LEARN_RATE_END        = 1e-10
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 1
__C.TRAIN.SECOND_STAGE_EPOCHS   = 10
__C.TRAIN.INITIAL_WEIGHT        = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/checkpoint/rdt_model"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/data/dataset/rdt_train.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 512
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/data/detection"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = settings.RDT_GIT_ROOT+"/tensorflow-yolov3/checkpoint/rdt_model"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.1
__C.TEST.IOU_THRESHOLD          = 0.1











