"""
    Takes a list of samples in the probability matrix form of
    [   [ -------Softmax 1---------],
        [ -------Softmax 2---------],
        [ -------Softmax 3---------]   ]
"""
from scripts import similarity
import numpy as np
import random
import matplotlib.pyplot as plt
from Net.tensor import tensornet
from Net.tensor import net6_c, net6
from Net.tensor import inputdata
import tensorflow as tf
from pipeline.bincam import BinaryCamera
import cv2
import time
class OutputAnalysis():
    NUM_CLASSES = 5
    def variance(self, axis=0):
        return np.var(self.samples, axis=axis)
    def mean(self, axis=0):
        return np.mean(self.samples, axis=axis)
   
    def actions(self):
        rot_samples, ext_samples = zip(*self.samples)
        rot_samples, ext_samples = np.array(rot_samples), np.array(ext_samples)
        print "Total length:", len(rot_samples)
        

        rot_classes = np.argmax(rot_samples, axis=1) 
        ext_classes = np.argmax(ext_samples, axis=1)

        rot_class_freq = np.zeros(OutputAnalysis.NUM_CLASSES)
        for i in range(OutputAnalysis.NUM_CLASSES):
            rot_class_freq[i] = sum(np.equal(rot_classes, i))

        ext_class_freq = np.zeros(OutputAnalysis.NUM_CLASSES)
        for i in range(OutputAnalysis.NUM_CLASSES):
            ext_class_freq[i] = sum(np.equal(ext_classes, i))

        print "SAMPLES"
        print self.samples
        
        print "\nROTATION SAMPLES"  
        print rot_samples
        print "ROTATION CLASSES"
        print rot_classes
        print "ROTATION CLASS FREQUENCIES"
        print rot_class_freq

        print "\nEXTENSION SAMPLES"
        print ext_samples
        print "EXTENSION CLASSES"
        print ext_classes 
        print "EXTENSION CLASS FREQUENCIES"        
        print ext_class_freq
    

    @staticmethod
    def norm(values, type='l2'):
        if type == 'l2':
            return np.linalg.norm(values, axis=2)
        else:
            return np.max(values, axis=2)
    
    def __init__(self, images):
        self.images = images
        self.axis = 0
        self.samples = []
        self.compute_samples()
        self.samples = np.array(self.samples)
        return

    def compute_samples(self):
        with tf.Graph().as_default():
            net = net6_c.NetSix_C()
            path = '/media/1tb/Izzy/nets/net6_05-11-2016_12h09m12s.ckpt'
            sess = net.load(path)
            for frame in self.images:
                dists = net.class_dist(sess, frame, 3)
                dists = dists[:2]
                self.samples.append(dists)
        


    def print_outputs(self):
        el_var = self.variance(axis=0)
        el_mean = self.mean(axis=0)

        axis = 0

        l_inf = self.norm(self.samples, type='inf')
        l_inf_mean = np.mean(l_inf, axis=axis)
        l_inf_var = np.var(l_inf, axis=axis)

        l_2 = self.norm(self.samples, type='l2')
        l_2_mean = np.mean(l_2, axis = axis)
        l_2_var = np.var(l_2, axis = axis)
        
        print "\n\nSAMPLES"
        for i, sample in enumerate(self.samples):
            print str(i) + ". " + str(sample)   
        
        print "\nELEMENTWISE CLASSIFICATION"
        print "mean:", el_mean
        print "var: ", el_var

        print "\nL_INF CLASSIFICATION"
        print "norm:", l_inf
        print "mean:", l_inf_mean
        print "var: ", l_inf_var
        
        print "\nL_2 CLASSIFICATION"
        print "norm:", l_2
        print "mean:", l_2_mean
        print "var: ", l_2_var


class OutputAnalysisReg():

    def __init__(self, images):
        self.images = images
        self.axis = 0
        self.samples = []
        self.compute_samples()
        self.samples = np.array(self.samples)
        print self.samples
        return

    def compute_samples(self):
        with tf.Graph().as_default():
            net = net6.NetSix()
            path = '/media/1tb/Izzy/nets/net6_06-06-2016_11h27m25s.ckpt'
            sess = net.load(path)
            
            for image in self.images:
                sample = net.output(sess, image, 3, False)[:2]
                self.samples.append(sample)
                

    def print_outputs(self):            
        mean = np.mean(self.samples, axis=0)
        var = np.var(self.samples, axis=0)
        print "REGRESSION"
        print "mean: " + str(mean)
        print "var:  " + str(var)
        return        		

def softmax(values):
    """
        values is a list of floats
    """
    exps = np.exp(values)
    return exps / sum(exps)

def get_dummy_sample():
    """
        returns 4x3 probability matrix as sample
    """
    return np.array([softmax(np.random.rand(5)), softmax(np.random.rand(5))])

def dummy_data():
    samples = []
    for i in range(1):
        samples.append(get_dummy_sample())
    samples = np.array(samples)
    oa = OutputAnalysis(samples)
    oa.print_outputs()


def analyze():
    images = np.load('samples_images.npy')
    oa = OutputAnalysis(images)
    #oa.print_outputs()
    print "\n\n"
    oa.actions()


def analyze_reg():
    images = np.load('samples_images.npy')
    oa = OutputAnalysisReg(images)
    oa.print_outputs()


if __name__ == '__main__':
	#action_analysis(['exp_samples.npy'])
	#analyze(['control_samples.npy', 'exp_samples.npy'])
    #analyze()
    print analyze()
    print "\n\n"
    analyze_reg()



