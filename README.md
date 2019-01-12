## part 1. Introduction

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). This repository  is inspired by [Paweł Kapica](https://github.com/mystic123) and [Kiril Cvetkov
](https://github.com/kirilcvetkov92). The full details are in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).  In this project we cover several segments as follows:<br>
- [x] [YOLO v3 architecture](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py)
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [x] Basic working demo
- [x] Non max suppression on the both `GPU` and `CPU` is supported
- [x] Training pipeline
- [x] Compute COCO mAP

YOLO paper is quick hard to understand, along side that paper. This [tutorial](https://github.com/YunYang1994/tensorflow-yolov2_from_scratch) enables you to have a quick understanding of YOLO Algorithmn.


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
### 3.1 quick train
The purpose of this demo is to give you a glimpse of yolov3 training process. `python core/convert_tfrecord.py` to convert your imageset into tfrecords
```
$ python core/convert_tfrecord.py --dataset /data/train_data/quick_train_data/quick_train_data.txt  --tfrecord_path_prefix /data/train_data/quick_train_data/tfrecords/quick_train_data
$ python quick_train.py  # start training
```
### 3.2 train coco dataset
Firstly, you need to download the COCO2017 dataset from the [website](http://cocodataset.org/)　and put them in the `./data/train_data/COCO`
```bashrc
$ cd data/train_data/COCO
$ wget http://images.cocodataset.org/zips/train2017.zip
$ unzip train2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
```
Then you are supposed to extract some useful information such as bounding box, category id .etc from COCO dataset and generate your own `.txt` file.
```
$ python core/extract_coco.py --dataset_info_path ./data/train_data/COCO/train2017.txt
```
As a result, you will get  `./data/train_data/COCO/train2017.txt`.  Here is an example row for one image:<br>
```
/home/yang/test/tensorflow-yolov3/data/train_data/train2017/000000458533.jpg 20 18.19 6.32 424.13 421.83 20 323.86 2.65 640.0 421.94
/home/yang/test/tensorflow-yolov3/data/train_data/train2017/000000514915.jpg 16 55.38 132.63 519.84 380.4
# image_path, category_id, x_min, y_min, x_max, y_max, category_id, x_min, y_min, ...
```
In this step, you will convert image dataset into some `.tfrecord`  which are a kind of recommended file format for Tensorflow to store your data as  binary file. Finally, you can train it now!
```
$ python core/convert_tfrecord.py --dataset ./data/train_data/COCO/train2017.txt  --tfrecord_path_prefix ./data/train_data/COCO/tfrecords/coco --num_tfrecords 100
$ python train.py
```
### 3.3 evaluate coco dataset (continue to work)
```
$ cd data/train_data/COCO
$ wget http://images.cocodataset.org/zips/test2017.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2017.zip 
$ unzip test2017.zip
$ unzip image_info_test2017.zip
```

## part 4. Why it is so magical ?
YOLO stands for You Only Look Once. It's an object detector that uses features learned by a deep convolutional neural network to detect an object. Although we has successfully run these codes, we must understand how YOLO works. 
### 4.1 Architercutre details
In this project, I use the pretrained weights, where we have 80 trained yolo classes (COCO dataset), for recognition. And the class [label](./data/coco.names) is represented as  `c`  and it's integer from 1 to 80, each number represents the class label accordingly. If `c=3`, then the classified object is a  `car`.  The image features learned by the deep convolutional layers are passed onto a classifier and regressor which makes the detection prediction.(coordinates of the bounding boxes, the class label.. etc).details also see in the below picture. (thanks [Levio](https://blog.csdn.net/leviopku/article/details/82660381) for your great image!)

![image](./docs/images/levio.jpeg)

### 4.2 Neural network io:
-  **input** : [None, 416, 416, 3]
-  **output** : confidece of an object being present in the rectangle, list of rectangles position and sizes and classes of the objects begin detected. Each bounding box is represented by 6 numbers `(Rx, Ry, Rh, Rw, Pc, C1..Cn)` as explained above. In this case n=80, which means we have `c` as 80-dimensional vector, and the final size of representing the bounding box is 85.  why is 85? see also in the below picture
![image](./docs/images/probability_extraction.png)
As you can see in the above picture, The first number `Pc` is the confidence of an project, The second four number `bx, by, bh, bw` represents the information of bounding boxes. The last 80 number each is the output probability of corresponding-index class.

### 4.3 Filtering with score threshold

The output result may contain several rectangles that are false positives or overlap,  if your input image size of `[416, 416, 3]`, you will get `(52X52+26X26+13X13)x3=10647` boxes since YOLO v3 totally uses 9 anchor boxes. (Three for each scale). So It is time to find a way to reduce them. The first attempt to reduce these rectangles is to filter them by score threshold.

**Input arguments**: 

- `boxes`: tensor of shape [10647, 4)] 
- `scores`: tensor of shape `[10647, 80]` containing the detection scores for 80 classes. 
- `score_thresh`: float value , then get rid of whose boxes with low score

```
# Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
score_thresh=0.4
mask = tf.greater_equal(scores, tf.constant(score_thresh))
```

### 4.4 Do non-maximum suppression

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

Hope it helps you, Start your tensorflow-yolv3 journey here now!
