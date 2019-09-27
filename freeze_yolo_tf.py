# -*- coding: utf-8 -*-
"""Freeze tf-yolo model to use for inference. Change ckpt_file variable and model_export_path as needed

Example:

        $ python freeze_yolo_tf.py

"""

import tensorflow as tf
from core.yolov3 import YOLOV3
import tempfile
import os, shutil



ckpt_file = "./checkpoint/rdt_model" # Model to freeze
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
model_export_path = "./models/Flu_audere" # export directory
version = 2
export_path = os.path.join(model_export_path, str(version))




if __name__ == "__main__":

    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data',shape=(1,512,512,3))

    model = YOLOV3(input_data, trainable=False)

    with tf.Session() as sess:


        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)

        trainable = tf.placeholder(tf.bool, name='trainable')

        pred_sbbox, pred_mbbox, pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        classification_inputs = tf.saved_model.utils.build_tensor_info(input_data)
        classification_train = tf.saved_model.utils.build_tensor_info(trainable)

        classification_pred_sbbox = tf.saved_model.utils.build_tensor_info(pred_sbbox)
        classification_pred_mbbox = tf.saved_model.utils.build_tensor_info(pred_mbbox)
        classification_pred_lbbox = tf.saved_model.utils.build_tensor_info(pred_lbbox)



        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={"input":classification_inputs},

                outputs={

                    "pred_sbbox":
                        classification_pred_sbbox,

                    "pred_mbbox":

                        classification_pred_mbbox,

                    "pred_lbbox":

                        classification_pred_lbbox


                },

                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(

            sess, [tf.saved_model.tag_constants.SERVING],

            signature_def_map={

                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:

                    classification_signature,

            },

            main_op=tf.tables_initializer(),

            strip_default_attrs=True,clear_devices=True)



        builder.save()


