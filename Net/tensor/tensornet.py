"""
    DO NOT INSTANTIATE THIS CLASS!!!!
    Instead use subclasses that actually contain the net architectures

    General information about TensorNet
        - Save will save the current sessions variables to the given path.
          If no path given, it saves to 'self.dir/[timestamped net name].ckpt'
        - Load takes a path and returns a session with those tf variables
        - Optimize will load variables from path if given. Otherwise it will initialize new ones.
          Will save to new timestamp rather than overwriting given path
        - Output takes a session (that was ideally loaded from TensorNet.load) and image and returns
        - the net output in a list. Try not to edit the binary image. output will automatically reformat normal cv2.imread
          or BinaryCamera.read_binary_frame images.

    Try to close sessions after using them (i.e. sess.close()). If more than one is open at a time, exceptions are thrown

    CHANGED save_path directory from self.dir to /media/1tb/Izzy/nets/
"""


import tensorflow as tf
import time
import datetime
import inputdata
import logging
import numpy as np
try:
    import options
except:
    options = None
    pass

import os

class TensorNet():

    def __init__(self):
        raise NotImplementedError

    def save(self, sess, save_path=None):
        
        self.log( "Saving..." )
        saver = tf.train.Saver()
        if not save_path:
            model_name = self.name + "_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + ".ckpt"
            save_path = '/media/1tb/Izzy/nets/' + model_name
        save_path = saver.save(sess, save_path)
        self.log( "Saved model to " + save_path )
        self.recent = save_path
        return save_path



    def load(self, var_path=None):
        """
            load net's variables from absolute path or relative
            to the current working directory. Returns the session
            with those weights/biases restored.
        """
        if not var_path:
            raise Exception("No path to model variables specified")
        print "Restoring existing net from " + var_path + "..."
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            saver.restore(sess, var_path)
        return sess


    def optimize(self, iterations, data, path=None, batch_size=100, test_print=20, save=True):
        """
            optimize net for [iterations]. path is either absolute or 
            relative to current working directory. data is InputData object (see class for details)
            mini_batch_size as well
        """
        if path:
            sess = self.load(var_path=path)
        else:
            print "Initializing new variables..."
            sess = tf.Session()
            sess.run(tf.initialize_all_variables())
            
        if options:
            self.log_path = options.Options.tf_dir + self.dir + 'train.log'
        else:
            self.log_path = self.dir + 'train.log'
        #logging.basicConfig(filename=log_path, level=logging.DEBUG)
        
        try:
            with sess.as_default():
                for i in range(iterations):
                    batch = data.next_train_batch(batch_size)
                    ims, labels = batch

                    feed_dict = { self.x: ims, self.y_: labels }
                    if i % 3 == 0:
                        batch_loss = self.loss.eval(feed_dict=feed_dict)
                        self.log("[ Iteration " + str(i) + " ] Training loss: " + str(batch_loss))
                    if i % test_print == 0:
                        test_batch = data.next_test_batch()
                        test_ims, test_labels = test_batch
                        test_dict = { self.x: test_ims, self.y_: test_labels }
                        test_loss = self.loss.eval(feed_dict=test_dict)
                        self.log("[ Iteration " + str(i) + " ] Test loss: " + str(test_loss))
                    self.train.run(feed_dict=feed_dict)
                

        except KeyboardInterrupt:
            pass
        
        if path:
            dir, old_name = os.path.split(path)
            dir = dir + '/'
        else:
            dir = options.Options.tf_dir + self.dir
        new_name = self.name + "_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + ".ckpt"
        if save:
            save_path = self.save(sess, save_path='/media/1tb/Izzy/nets/' + new_name)
        else:
            save_path = None
        sess.close()
        self.log( "Optimization done." )
        return save_path
    

    def deploy(self, path, im):
        """
            accepts 3-channel image with pixel values from 
            0-255 and returns controls in four element list
        """
        sess = self.load(var_path=path)
        im = inputdata.im2tensor(im)
        shape = np.shape(im)
        im = np.reshape(im, (-1, shape[0], shape[1], shape[2]))
        with sess.as_default():
            return sess.run(self.y_out, feed_dict={self.x:im})
        

    def output(self, sess, im,channels):
        """
            accepts batch of 3d images, converts to tensor
            and returns four element list of controls
        """
        im = inputdata.im2tensor(im,channels)
        shape = np.shape(im)
        im = np.reshape(im, (-1, shape[0], shape[1], shape[2]))
        with sess.as_default():
            return sess.run(self.y_out, feed_dict={self.x:im}) [0]



    @staticmethod
    def reduce_shape(shape):
        """
            Given shape iterable, return total number of nodes/elements
        """
        shape = [ x.value for x in shape ]
        f = lambda x, y: 1 if y is None else x * y
        return reduce(f, shape, 1)


    def weight_variable(self, shape, stddev=.005):
        initial = tf.random_normal(shape, stddev=stddev)
        #initial = tf.random_normal(shape)
        #initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape, stddev=.01):
        initial = tf.random_normal(shape, stddev=stddev)
        #initial = tf.random_normal(shape)
        #initial = tf.constant(stddev, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool(self, x, k):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

    def log(self, message):
        print message
        f = open(self.log_path, 'a+')
        #logging.debug(message)
        f.write("DEBUG:root:" + message + "\n")
