from Net.tensor import net3, inputdata
from options import AMTOptions as opt
from tensorflow.python.framework import ops
import numpy as np
import cv2

def get_conv1_variables(net_class, path):
    """
        Warning: this method will create and load a graph, please reset after using.
        returns the conv1 weights froma conv net given
        net class and path
    """
    net = net_class()
    sess = net.load(var_path = path)
    weights = sess.run(net.w_conv1)
    biases = sess.run(net.b_conv1)
    sess.close()
    return weights, biases

def assign_variables(sess, target_layer_weights, target_layer_biases, source_weights, source_biases):
    """
    Ensure that the layer's weights dimensions match that of the source weights
    ;param session of current target net
    :param target_layer_weights/biases: tf layer that will have its weights replaced
    :param source_weights/biases: np array of weights to replace with.
    :return:
    """
    transfer_w_op = target_layer_weights.assign(source_weights)
    transfer_b_op = target_layer_biases.assign(source_biases)
    sess.run(transfer_w_op)
    sess.run(transfer_b_op)


if __name__ == '__main__':
    test_image_path = opt.data_dir + 'color_images/dataset1_img_0.jpg'
    test_image = cv2.imread(test_image_path)

    source_path = '/Users/JonathanLee/Desktop/sandbox/vision/Net/tensor/./net3/net3_02-14-2016_21h52m23s.ckpt'
    target_path = '/Users/JonathanLee/Desktop/sandbox/vision/Net/tensor/net3/net3_02-14-2016_21h47m16s.ckpt'

    weights, biases = get_conv1_variables(net3.NetThree, source_path)
    ops.reset_default_graph() # required because get_conv1_variables loads new vars into instance

    new_net = net3.NetThree()
    sess = new_net.load(var_path = target_path)
    print new_net.output(sess, test_image)

    assign_variables(sess, new_net.w_conv1, new_net.b_conv1, weights, biases)
    print new_net.output(sess, test_image)

    sess.close()





