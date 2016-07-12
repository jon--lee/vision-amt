from Net.tensor import net3, net6, inputdata
from options import AMTOptions as opt
from tensorflow.python.framework import ops
import numpy as np
import cv2
import time

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
    # sess.run(transfer_w_op)
    # sess.run(transfer_b_op)

def get_layer_variables(sess, layer):
    """
        Warning: this method will create and load a graph, please reset after using.
        returns the conv1 weights froma conv net given
        net class and path
    """
    weights = sess.run(layer)
    return weights

def all_weights(net, sess, source_path):
    """
    Can only trasfter all the weights from netsix architectures
    returns a dictionary of name to weight matrix
    """
    net.saver.restore(sess, source_path)
    weights = {}
    weights['w_conv1'] = get_layer_variables(sess, net.w_conv1)
    weights['b_conv1'] = get_layer_variables(sess, net.b_conv1)
    weights['w_fc1'] = get_layer_variables(sess, net.w_fc1)
    # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
    weights['b_fc1'] = get_layer_variables(sess, net.b_fc1)
    weights['w_fc2'] = get_layer_variables(sess, net.w_fc2)
    weights['b_fc2'] = get_layer_variables(sess, net.b_fc2)
    ops.reset_default_graph()
    return weights

def assign_all_variables(sess, new_net, weights):
    """
    Assigns the variables in a type six net given a particular session
    """
    start = time.time()
    assign_variables(sess, new_net.w_conv1, new_net.b_conv1, weights['w_conv1'], weights['b_conv1'])
    assign_variables(sess, new_net.w_fc1, new_net.b_fc1, weights['w_fc1'], weights['b_fc1'])
    assign_variables(sess, new_net.w_fc2, new_net.b_fc2, weights['w_fc2'], weights['b_fc2'])
    print time.time()-start

if __name__ == '__main__':
    # test_image_path = opt.data_dir + 'color_images/dataset1_img_0.jpg'
    # test_image = cv2.imread(test_image_path)

    # source_path = '/Users/JonathanLee/Desktop/sandbox/vision/Net/tensor/./net3/net3_02-14-2016_21h52m23s.ckpt'
    # target_path = '/Users/JonathanLee/Desktop/sandbox/vision/Net/tensor/net3/net3_02-14-2016_21h47m16s.ckpt'

    # weights, biases = get_conv1_variables(net3.NetThree, source_path)
    # ops.reset_default_graph() # required because get_conv1_variables loads new vars into instance

    # new_net = net3.NetThree()
    # sess = new_net.load(var_path = target_path)
    # print new_net.output(sess, test_image)

    # assign_variables(sess, new_net.w_conv1, new_net.b_conv1, weights, biases)
    # print new_net.output(sess, test_image)

    # sess.close()
    target_path = '/media/1tb/Izzy/nets/net6_06-16-2016_10h43m14s.ckpt'
    new_net = net6.NetSix()
    sess = new_net.load(var_path = target_path)
    nets_file = open(opt.amt_dir + 'net_cluster.txt')
    paths = []
    for net_path in nets_file:
        paths.append(net_path.split(' ')[0])
    weight_dicts = []
    for path in paths:
        weight_dicts.append(all_weights(new_net, sess, path))
    for weight_dict in weight_dicts:
        for layer in weight_dict.keys():
            print layer, weight_dict[layer].shape
        print weight_dict['w_conv1']

    start = time.time()
    for weight_dict in weight_dicts:
        assign_all_variables(sess, new_net, weight_dict)
        print get_layer_variables(sess, new_net.w_conv1)


    sess.close()



