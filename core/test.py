import numpy as np
from tools.utils import get_boxes_confs_probs, filter_boxes, nms, draw_boxes
from utils import load_coco_names
from PIL import Image, ImageFont, ImageDraw
import yolov3
import tensorflow as tf


# model = yolo_v3.yolo_v3
model = yolov3.Yolov3()
iou_threshold = 0.5
# img = Image.open('../data/dog.jpg')
img = Image.open('../data/road.jpeg')
img_resized = img.resize(size=(416, 416))

classes = load_coco_names('../data/coco.names')

# placeholder for detector inputs
inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])

with tf.variable_scope('detector'):
    feature_map = model.forward(inputs, len(classes),
                        data_format='NCHW')

saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

boxes, confs,  probs = get_boxes_confs_probs(feature_map)
boxes, scores, probs = filter_boxes(boxes, confs, probs, 0.5)
print("==>", boxes, scores, probs)
boxes, scores, labels = nms(boxes, scores, probs, iou_threshold=0.4)
# print("==>", boxes, scores, labels)

with tf.Session() as sess:
    saver.restore(sess, './saved_model/yolov3.ckpt')
    print('model restored.')

    boxes, scores, labels = sess.run(
        [boxes, scores, labels], feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
        # [boxes, scores, probs], feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})

image = draw_boxes(boxes, scores, labels, img, classes, show=True)

# ##### draw bounding box #####
# detection_size = [416, 416]
# image = img
# draw = ImageDraw.Draw(image)

# # draw settings
# font = ImageFont.truetype(font = '../data/font/FiraMono-Medium.otf',
                              # size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
# hsv_tuples = [( x / len(classes), 0.8, 1.0) for x in range(len(classes))]
# colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
# colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

# # thickness = (image.size[0] + image.size[1]) // 300

# for i in range(len(labels)): # for each bounding box, do:
    # bbox, score, label = boxes[i], scores[i], classes[labels[i]]
    # bbox_text = "%s %.2f" %(label, score)
    # text_size = draw.textsize(bbox_text, font)
    # # convert_to_original_size
    # detection_size, original_size = np.array(detection_size), np.array(image.size)
    # ratio = original_size / detection_size
    # bbox = list((bbox.reshape(2,2) * ratio).reshape(-1))

    # draw.rectangle(bbox, outline=colors[labels[i]], width=3)
    # draw.rectangle([tuple(bbox[:2]), tuple(np.array(bbox[:2])+text_size)], fill=colors[labels[i]])

    # # # draw bbox
    # draw.text(bbox[:2], bbox_text, fill=(0,0,0), font=font)

# image.show()







