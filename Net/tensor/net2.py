import tensorflow as tf
from tensornet import TensorNet


class NetTwo(TensorNet):
    def __init__(self):
        self.dir = './net2/'
        self.name = 'net2'

        self.x = tf.placeholder('float', shape=[None, 125, 125, 1])
        self.y_ = tf.placeholder("float", shape=[None, 4])

        self.w_conv1 = self.weight_variable([5, 5, 1, 32])
        self.b_conv1 = self.bias_variable([32])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

        self.w_conv2 = self.weight_variable([5, 5, 32, 32])
        self.b_conv2 = self.bias_variable([32])
        
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.w_conv2) + self.b_conv2)

        conv2_num_nodes = TensorNet.reduce_shape(self.h_conv2.get_shape())
        fc1_num_nodes = 128

        self.w_fc1 = self.weight_variable([conv2_num_nodes, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_conv2_flat = tf.reshape(self.h_conv2, [-1, conv2_num_nodes])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv2_flat, self.w_fc1) + self.b_fc1)
        
        self.w_fc2 = self.weight_variable([fc1_num_nodes, 4])
        self.b_fc2 = self.bias_variable([4])
        
        self.y_out = tf.tanh(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.loss)

 
        
if __name == '__main__':
    # these lines are broken now
    #net = NetTwo()
    #net.optimize(100, batch_size=300)

