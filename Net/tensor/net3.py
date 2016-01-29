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

        self.x = tf.placeholder('float', shape=[None, 125, 125, 1])
        self.y_ = tf.placeholder("float", shape=[None, 4])

        self.w_conv1 = self.weight_variable([5, 5, 1, 32])
        self.b_conv1 = self.bias_variable([32])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

        conv1_num_nodes = NetThree.reduce_shape(self.h_conv1.get_shape())
        fc1_num_nodes = 128
        
        self.w_fc1 = self.weight_variable([conv1_num_nodes, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_conv1_flat = tf.reshape(self.h_conv1, [-1, conv1_num_nodes])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv1_flat, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([fc1_num_nodes, 4])
        self.b_fc2 = self.bias_variable([4])

        self.y_out = tf.tanh(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.loss)

        
        


    
    


net = NetThree()
net.optimize(100, path='net3/net3_01-19-2016_00h47m49s.ckpt')
