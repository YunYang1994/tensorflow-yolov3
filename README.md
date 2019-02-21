## part 1. Introduction

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). This repository  is inspired by [Paweł Kapica](https://github.com/mystic123). The full details are in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).  In this project we cover several segments as follows:<br>
- [x] [YOLO v3 architecture](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py)
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [x] Basic working demo
- [x] Non max suppression on the both `GPU` and `CPU` is supported
- [x] Training pipeline
- [x] Compute COCO mAP

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
3. Exporting loaded COCO weights as TF checkpoint(`yolov3.ckpt`) and frozen graph (`yolov3_gpu_nms.pb`) . If you don't have [yolov3.weights](https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3.weights). Download and put it in the dir `./checkpoint`
```bashrc
$ python convert_weight.py --convert --freeze
```
4. Then you will get some `.pb` files in the dir `./checkpoint`,  and run the demo script
```bashrc
$ python nms_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```
![image](./docs/images/611_result.jpg)
## part 3. Train on your own dataset
Three files are required as follows:

- `dataset.txt`: 

```
xxx/xxx.jpg 18.19 6.32 424.13 421.83 20 323.86 2.65 640.0 421.94 20 
xxx/xxx.jpg 55.38 132.63 519.84 380.4 16
# image_path x_min y_min x_max y_max class_id  x_min y_min ... class_id 
```
- `anchors.txt`

```
0.10,0.13, 0.16,0.30, 0.33,0.23, 0.40,0.61, 0.62,0.45, 0.69,0.59, 0.76,0.60,  0.86,0.68,  0.91,0.76
```

- `class.names`

```
person
bicycle
car
...
toothbrush
```

### 3.1 Train raccoon dataset
To help you understand my training process, I made this training-pipline demo. [raccoon dataset](https://github.com/YunYang1994/raccoon_dataset) has only one class with 200 images (180 for train, 20 for test), I have prepared a shell script in the `./scripts` which enables you to get data and train it !
#### how to train it ?
```
$ sh scripts/make_raccoon_tfrecords.sh
$ python show_input_image.py               # show your input image (optional)
$ python kmeans.py                         # get prior anchors and rescale the values to the range [0,1]
$ python convert_weight.py --convert       # get pretrained weights
$ python quick_train.py
$ tensorboard --logdir ./data
```
As you can see in the tensorboard, if your dataset is too small or you train for too long, the model starts to overfit and learn patterns from training data that does not generalize to the test data.

#### how to test and evaluate it ?
```
$ python convert_weight.py -cf ./checkpoint/yolov3.ckpt-2500 -nc 1 -ap ./data/raccoon_anchors.txt --freeze
$ python quick_test.py
$ python evaluate.py
```
if you are still unfamiliar with training pipline, you can join [here](https://github.com/YunYang1994/tensorflow-yolov3/issues/39) to discuss with us.

|raccoon-181.jpg|raccoon-55.jpg|
|---|:---:|
|![weibo-logo](./docs/images/raccoon1.jpg)|![weibo-logo](./docs/images/raccoon2.jpg)|

### 3.2 Train other dataset
Download VOC PASCAL trainval  and test data
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Download COCO trainval  and test data
```
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/test2017.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2017.zip 
```

## part 4. Why it is so magical ?
YOLO stands for You Only Look Once. It's an object detector that uses features learned by a deep convolutional neural network to detect an object. Although we has successfully run these codes, we must understand how YOLO works. 
### 4.1 Anchors clustering
The paper suggests to use clustering on bounding box shape to find the good anchor box specialization suited for the data. more details see [here](https://nbviewer.jupyter.org/github/YunYang1994/tensorflow-yolov3/blob/master/docs/Box-Clustering.ipynb)
![image](./docs/images/K-means.png)

### 4.2 Architercutre details
In this project, I use the pretrained weights, where we have 80 trained yolo classes (COCO dataset), for recognition. And the class [label](./data/coco.names) is represented as  `c`  and it's integer from 1 to 80, each number represents the class label accordingly. If `c=3`, then the classified object is a  `car`.  The image features learned by the deep convolutional layers are passed onto a classifier and regressor which makes the detection prediction.(coordinates of the bounding boxes, the class label.. etc).details also see in the below picture. (thanks [Levio](https://blog.csdn.net/leviopku/article/details/82660381) for your great image!)

![image](./docs/images/levio.jpeg)

### 4.3 Neural network io:
-  **input** : [None, 416, 416, 3]
-  **output** : confidece of an object being present in the rectangle, list of rectangles position and sizes and classes of the objects begin detected. Each bounding box is represented by 6 numbers `(Rx, Ry, Rw, Rh, Pc, C1..Cn)` as explained above. In this case n=80, which means we have `c` as 80-dimensional vector, and the final size of representing the bounding box is 85.The first number `Pc` is the confidence of an project, The second four number `bx, by, bw, bh` represents the information of bounding boxes. The last 80 number each is the output probability of corresponding-index class.

### 4.4 Filtering with score threshold

The output result may contain several rectangles that are false positives or overlap,  if your input image size of `[416, 416, 3]`, you will get `(52X52+26X26+13X13)x3=10647` boxes since YOLO v3 totally uses 9 anchor boxes. (Three for each scale). So It is time to find a way to reduce them. The first attempt to reduce these rectangles is to filter them by score threshold.

**Input arguments**: 

- `boxes`: tensor of shape [10647, 4] 
- `scores`: tensor of shape `[10647, 80]` containing the detection scores for 80 classes. 
- `score_thresh`: float value , then get rid of whose boxes with low score

```
# Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
score_thresh=0.4
mask = tf.greater_equal(scores, tf.constant(score_thresh))
```

### 4.5 Do non-maximum suppression

Even after yolo filtering by thresholding over, we still have a lot of overlapping boxes. Second approach and filtering is Non-Maximum suppression algorithm.

* Discard all boxes with `Pc <= 0.4`  
* While there are any remaining boxes : 
    * Pick the box with the largest `Pc`
    * Output that as a prediction
    * Discard any remaining boxes with `IOU>=0.5` with the box output in the previous step

In tensorflow, we can simply implement non maximum suppression algorithm like this. more details see [here](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/utils.py)
```
for i in range(num_classes):
    tf.image.non_max_suppression(boxes, score[:,i], iou_threshold=0.5) 
 ```
 
Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU. Here is an exmaple of non maximum suppression algorithm: on input the aglorithm receive 4 overlapping bounding boxes, and the output returns only one

![image](./docs/images/iou.png)

Welecome to discuss with me.

## part 5. Other Implementations

[- **`YOLOv3_TensorFlow`**](https://github.com/wizyoung/YOLOv3_TensorFlow)

[- **`Implementing YOLO v3 in Tensorflow (TF-Slim)`**](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)

[- **`Object Detection using YOLOv2 on Pascal VOC2012`**](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html)

[-**`Understanding YOLO`**](https://hackernoon.com/understanding-yolo-f5a74bbc7967)

[-**`YOLOv3目标检测有了TensorFlow实现，可用自己的数据来训练`**](https://mp.weixin.qq.com/s/cq7g1-4oFTftLbmKcpi_aQ)<br>

[- `学员分享 | 小哥哥和用YOLOv3做目标检测的故事「文末送课」`](https://mp.weixin.qq.com/s/dFiOkUsal62EoME52Iw-uQ)

[-`目标检测|YOLOv2原理与实现(附YOLOv3)`](https://zhuanlan.zhihu.com/p/35325884)

[-` YOLOv2は、2016年12月25日時点の、速度、精度ともに世界最高のリアルタイム物体検出手法です。`](https://github.com/leetenki/YOLOv2/blob/master/YOLOv2.md)

[-` 知乎专栏-目标检测yolov2`](https://zhuanlan.zhihu.com/p/35325884)
