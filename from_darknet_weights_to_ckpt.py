import tensorflow as tf
from core.yolov3 import YOLOV3

iput_size = 416
darknet_weights = '<your yolov3.weights' path>'
ckpt_file = './checkpoint/yolov3_coco.ckpt'

def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)  # np.ndarray
    print('weights_num:', weights.shape[0])
    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for vari in batch_norm_vars:
                    shape = vari.shape.as_list()
                    num_params = np.prod(shape)
                    vari_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(
                        tf.assign(vari, vari_weights, validate_shape=True))
                i += 4
            elif 'conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                           bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(
                    tf.assign(bias, bias_weights, validate_shape=True))
                i += 1
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1
    print('ptr:', ptr)
    return assign_ops
    
with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32,shape=(None, iput_size, iput_size, 3), name='input_data')
model = YOLOV3(input_data, trainable=False)
load_ops = load_weights(tf.global_variables(), darknet_weights)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(load_ops)
    save_path = saver.save(sess, save_path=ckpt_file)
    print('Model saved in path: {}'.format(save_path))
