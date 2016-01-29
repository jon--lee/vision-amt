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

    ROOT = '/Users/JonathanLee/Desktop/vision/'

    TRAIN_PATH = ROOT + 'Net/hdf/train.txt'
    TEST_PATH = ROOT + 'Net/hdf/test.txt'

    def __init__(self):
        raise NotImplementedError

    def save(self, sess, save_path=None):
        
        self.log( "Saving..." )
        saver = tf.train.Saver()
        if not save_path:
            model_name = self.name + "_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + ".ckpt"
            save_path = self.dir + model_name
        save_path = saver.save(sess, save_path)
        self.log( "Saved model to " + save_path )
        self.recent = save_path
        return save_path



    def load(self, var_path=None):
        if not var_path:
            raise Exception("No path to model variables specified")
        print "Restoring existing net from " + var_path + "..."
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            saver.restore(sess, var_path)
        return sess


    def optimize(self, iterations, path=None, data=None, batch_size=100):
        if path:
            sess = self.load(var_path=path)
        else:
            print "Initializing new variables..."
            sess = tf.Session()
            sess.run(tf.initialize_all_variables())
            
        if options:
            log_path = options.Options.tf_dir + self.dir + 'train.log'
        else:
            log_path = self.dir + 'train.log'
        logging.basicConfig(filename=log_path, level=logging.DEBUG)
        
        try:
            with sess.as_default():
                if not data:
                    print "Loading data..."                
                    data = inputdata.InputData(self.TRAIN_PATH, self.TEST_PATH)
                    print "Data loaded."
                for i in range(iterations):
                    batch = data.next_train_batch(batch_size)
                    ims, labels = batch

                    feed_dict = { self.x: ims, self.y_: labels }
                    if i % 3 == 0:
                        batch_loss = self.loss.eval(feed_dict=feed_dict)
                        self.log("[ Iteration " + str(i) + " ] Training loss: " + str(batch_loss))
                    self.train.run(feed_dict=feed_dict)
                    time.sleep(1.1)
                

        except KeyboardInterrupt:
            pass
        dir, old_name = os.path.split(path)
        new_name = self.name + "_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + ".ckpt"
        save_path = self.save(sess, save_path=dir + new_name)
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
        

    def output(self, sess, im):
        """
            accepts batch of 3d images, converts to tensor
            and returns four element list of controls
        """
        im = inputdata.im2tensor(im)
        shape = np.shape(im)
        im = np.reshape(im, (-1, shape[0], shape[1], shape[2]))
        with sess.as_default():
            return sess.run(self.y_out, feed_dict={self.x:im}) [0]



    @staticmethod
    def reduce_shape(shape):
        """
            Given shape iterable with dimension elements
            reduce shape to total nodes
        """
        shape = [ x.value for x in shape ]
        f = lambda x, y: 1 if y is None else x * y
        return reduce(f, shape, 1)



    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def log(info):
        print info
        logging.debug(info)
