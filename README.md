# [TensorFlow2.0-Examples/4-Object_Detection/YOLOV3](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3)

## Please install tensorflow-gpu 1.11.0 !  Since Tensorflow is fucking ridiculous !

## part 1. Introduction [[代码剖析]](https://github.com/YunYang1994/ai-notebooks/blob/master/YOLOv3.md)

Implementation of YOLO v3 object detector in Tensorflow. The full details are in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).  In this project we cover several segments as follows:<br>
- [x] [YOLO v3 architecture](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py)
- [x] [Training tensorflow-yolov3 with GIOU loss function](https://giou.stanford.edu/)
- [x] Basic working demo
- [x] Training pipeline
- [x] Multi-scale training method
- [x] Compute VOC mAP

YOLO paper is quick hard to understand, along side that paper. This repo enables you to have a quick understanding of YOLO Algorithmn.


## part 2. Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/YunYang1994/tensorflow-yolov3.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd tensorflow-yolov3
$ pip install -r ./docs/requirements.txt
```
3. Exporting loaded COCO weights as TF checkpoint(`yolov3_coco.ckpt`)【[BaiduCloud](https://pan.baidu.com/s/11mwiUy8KotjUVQXqkGGPFQ&shfl=sharepset)】
```bashrc
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py
$ python freeze_graph.py
```
4. Then you will get some `.pb` files in the root path.,  and run the demo script
```bashrc
$ python image_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```
![image](./docs/images/611_result.jpg)
## part 3. Train your own dataset
Two files are required as follows:

- [`dataset.txt`](https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/master/data/dataset/voc_train.txt): 

```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```

- [`class.names`](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/data/classes/coco.names):

```
person
bicycle
car
...
toothbrush
```

### 3.1 Train VOC dataset
To help you understand my training process, I made this demo of training VOC PASCAL dataset
#### how to train it ?
Download VOC PASCAL trainval  and test data
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, which should have the following basic structure.

```bashrc

VOC           # path:  /home/yang/dataset/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
                     
$ python scripts/voc_annotation.py --data_path /home/yang/test/VOC
```
Then edit your `./core/config.py` to make some necessary configurations

```bashrc
__C.YOLO.CLASSES                = "./data/classes/voc.names"
__C.TRAIN.ANNOT_PATH            = "./data/dataset/voc_train.txt"
__C.TEST.ANNOT_PATH             = "./data/dataset/voc_test.txt"
```
Here are two kinds of training method: 

##### (1) train from scratch:

```bashrc
$ python train.py
$ tensorboard --logdir ./data
```
##### (2) train from COCO weights(recommend):

```bashrc
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py --train_from_coco
$ python train.py
```

#### how to test and evaluate it ?
```
$ python evaluate.py
$ cd mAP
$ python main.py -na
```
if you are still unfamiliar with training pipline, you can join [here](https://github.com/YunYang1994/tensorflow-yolov3/issues/39) to discuss with us.

### 3.2 Train other dataset
Download COCO trainval  and test data
```
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/test2017.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2017.zip 
```

## part 4. Other Implementations

[-**`YOLOv3目标检测有了TensorFlow实现，可用自己的数据来训练`**](https://mp.weixin.qq.com/s/cq7g1-4oFTftLbmKcpi_aQ)<br>

[-**`Stronger-yolo`**](https://github.com/Stinky-Tofu/Stronger-yolo)<br>

[- **`Implementing YOLO v3 in Tensorflow (TF-Slim)`**](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)

[- **`YOLOv3_TensorFlow`**](https://github.com/wizyoung/YOLOv3_TensorFlow)

[- **`Object Detection using YOLOv2 on Pascal VOC2012`**](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html)

[-**`Understanding YOLO`**](https://hackernoon.com/understanding-yolo-f5a74bbc7967)

