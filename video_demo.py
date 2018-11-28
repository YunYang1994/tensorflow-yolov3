import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import yolov3
import time
# import yolo_v3
from utils import load_coco_names, draw_boxes, detections_boxes, non_max_suppression


def main():
    # model = yolo_v3.yolo_v3
    model = yolov3.Yolov3()
    classes = load_coco_names('../data/coco.names')
    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])

    with tf.variable_scope('detector'):
        detections = model.forward(inputs, len(classes),
                           data_format='NCHW')

    saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

    boxes = detections_boxes(detections)


    vid = cv2.VideoCapture('../data/project_video.mp4')
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('../data/test_video_result.mp4', video_FourCC, video_fps, video_size)
    with tf.Session() as sess:
        saver.restore(sess, './saved_model/yolov3.ckpt')
        print('Model restored.')

        while True:
            return_value, frame = vid.read()
            if return_value:
                image = Image.fromarray(frame)
            else:
                break
            image_resized = image.resize(size=[416,416])

            prev_time = time.time()
            detected_boxes = sess.run(
            boxes, feed_dict={inputs: [np.array(image_resized, dtype=np.float32)]})
            curr_time = time.time()
            exec_time = curr_time - prev_time
            filtered_boxes = non_max_suppression(detected_boxes,
                                                 confidence_threshold=0.5,
                                                 iou_threshold=0.4)
            draw_boxes(filtered_boxes, image, classes, (416, 416))
            result = np.asarray(image)
            fps = "FPS: %d time: %.2f ms" %(int(1/exec_time), 1000*exec_time)
            cv2.putText(result, text=fps, org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5, color=(255, 0, 0), thickness=5)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == '__main__': main()
