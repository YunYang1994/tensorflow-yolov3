import tensorflow as tf
from core.yolov3 import YOLOV3
from from_darknet_weights_to_ckpt import load_weights

input_size = 416
darknet_weights = '<your darknet weights file path>'
pb_file = './yolov3.pb'
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, input_size, input_size, 3), name='input_data')
model = YOLOV3(input_data, trainable=False)
load_ops = load_weights(tf.global_variables(), darknet_weights)

with tf.Session() as sess:
    sess.run(load_ops)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names=output_node_names
    )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("{} ops written to {}.".format(len(output_graph_def.node), output_graph))
