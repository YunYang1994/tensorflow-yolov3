## introduction

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). This repository  is inspired by [Paweł Kapica](https://github.com/mystic123) and [Kiril Cvetkov
](https://github.com/kirilcvetkov92). The full details are in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).  In this project we cover several segments as follows:<br>
- [x] YOLO v3 architecture
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [x] Basic working demo
- [x] Non max suppression on the both `GPU` and `CPU` is supported
- [x] Training pipeline

## quick start
1. Clone this file
```bashrc
$ git clone https://github.com/YunYang1994/tf-yolov3.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd tf-yolov3
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
<img src="./docs/images/611_result.jpg" width=512 height=128 />
![image](./docs/images/611_result.jpg)
## Yolov3

