"""
    Model for net3
        conv
        relu
        fc
        relu
        fc
        tanh


"""


import tensorflow as tf
import inputdata
import random
from tensornet import TensorNet
import time
import datetime

class NetThree(TensorNet):

    def __init__(self):
        self.dir = "./net3/"
        self.name = "net3"
        channels = 1

        self.x = tf.placeholder('float', shape=[None, 250, 250, channels])
        self.y_ = tf.placeholder("float", shape=[None, 4])


        self.w_conv1 = self.weight_variable([5, 5, channels, 15])
        self.b_conv1 = self.bias_variable([15])

        """ Modifying net3 to contain max pooling layer"""        
       

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

        self.h_conv1 = self.max_pool(self.h_conv1, 4)

        # print self.h_conv1.get_shape()
        conv1_num_nodes = NetThree.reduce_shape(self.h_conv1.get_shape())
        fc1_num_nodes = 128
        
        self.w_fc1 = self.weight_variable([conv1_num_nodes, fc1_num_nodes])
        # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_conv1_flat = tf.reshape(self.h_conv1, [-1, conv1_num_nodes])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv1_flat, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([fc1_num_nodes, 4])
        self.b_fc2 = self.bias_variable([4])

        self.y_out = tf.tanh(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))
        self.train_step = tf.train.MomentumOptimizer(.03, .9)
        self.train = self.train_step.minimize(self.loss)

        
    def max_pool(self, conv, k):
        return tf.nn.max_pool(conv, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


if __name__ == '__main__':
    var_path = "/home/annal/Izzy/vision_amt/Net/tensor/net3/net3_02-11-2016_17h03m16s.ckpt"
    train_path = "home/annal/Izzy/vision_amt/data/amt/train.txt"
    test_path = "home/annal/Izzy/vision_amt/data/amt/test.txt"
    input_data = inputdata.AMTData(train_path, test_path)
    net = NetThree()


    net.optimize(100, input_data, path='net3_02-06-2016_16h38m28s.ckpt')    
    


