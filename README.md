## part1. Introduction

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). This repository  is inspired by [Paweł Kapica](https://github.com/mystic123) and [Kiril Cvetkov
](https://github.com/kirilcvetkov92). The full details are in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).  In this project we cover several segments as follows:<br>
- [x] [YOLO v3 architecture](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py)
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [x] Basic working demo
- [x] Non max suppression on the both `GPU` and `CPU` is supported
- [x] Training pipeline

## part2. Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/YunYang1994/tensorflow-yolov3.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd tensorflow-yolov3
$ pip install -r ./docs/requirements.txt
```
3. Exporting loaded COCO weights as TF checkpoint and frozen graph
```bashrc
$ python convert_weight.py --convert --freeze
```
4. The you will get some `.pb` files in the dir `./checkpoint`,  run the demo script
```bashrc
$ python nms_demo.py
```
![image](./docs/images/611_result.jpg)
## part3. Train for your own dataset

continue to work ...

## part4. How does yolov3 works
YOLO stands for You Only Look Once. It's an object detector that uses features learned by a deep convolutional neural network to detect an object. Before we get out hands dirty with code, we must understand how YOLO works.
### Architercutre details
In this project, I use pretrained weights, where we have 80 trained yolo classes, for recognition. The class label is represented as  `c`  and it's integer from 1 to 80, each number represent the class label accordingly. If  `c=3` , then the classified object is a  `car`.  The image features learned by the `Darknet-53` convolutional layers are passed onto a classifier/regressor which makes the detection prediction.(coordinates of the bounding boxes, the class label.. etc).Thanks [Levio](https://blog.csdn.net/leviopku/article/details/82660381) for your great image!
![image](./docs/images/levio.jpeg)

### Neural network io:
-  **input** : [None, 416, 416, 3]
-  **output** : confidece of an object being present in the rectangle, list of rectangles position and sizes and classes of the objects begin detected. Each bounding box is represented by 6 numbers `(Rx, Ry, Rh, Rw, Pc, C1..Cn)` as explained above. In this case n=80, which means we have `c` as 80-dimensional vector, and the final size of representing the bounding box is 85.  why is 85? see also in the below picture
![image](./docs/images/probability_extraction.png)
As you can see in the above picture, The first number `Pc` is the confidence of an project, The second four number `bx, by, bh, bw` represents the information of bounding boxes. The last 80 number each is the output probability of corresponding-index class.

### Filtering with score threshold

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

### Do non-max suppression

Even after yolo filtering by thresholding over, we still have a lot of overlapping boxes. Second approach and filtering is Non-Max suppression algorithm.

* Discard all boxes with `Pc <= 0.4`  
* While tehre are any remaining boxes : 
    * Pick the box with the largest `Pc`
    * Output that as a prediction
    * Discard any remaining boxes with `IOU>=0.5` with the box output in the previous step

```
for i in range(num_classes):
	tf.image.non_max_suppression(boxes, score[:,i], iou_threshold=0.5) 
 ```
 
Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU. Here is an exmaple of non max suppression algorithm: on input the aglorithm receive 4 overlapping bounding boxes, and the output returns only one

![image](./docs/images/iou.png)











