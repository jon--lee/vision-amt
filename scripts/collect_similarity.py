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
from Net.tensor import net6_c
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
    
    def __init__(self, samples):
        self.samples = samples
        self.axis = 0
        return

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
        
        print "\nELEMENTWISE"
        print "mean:", el_mean
        print "var: ", el_var

        print "\nL_INF"
        print "norm:", l_inf
        print "mean:", l_inf_mean
        print "var: ", l_inf_var
        
        print "\nL_2"
        print "norm:", l_2
        print "mean:", l_2_mean
        print "var: ", l_2_var

		

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

def process_images():
    images = np.load('samples_images.npy')
    
    classGraph = tf.Graph()
    regGraph = tf.Graph()
    with classGraph.as_default():
        classNet = net6_c.NetSix_C()
        path = '/media/1tb/Izzy/nets/net6_05-11-2016_12h09m12s.ckpt'
        classSess = classNet.load(path)

    with regGraph.as_default():
        regNet = net6.NetSix()
        path = ''
        regSess = net


def run_samples(n):
    net = net6_c.NetSix_C()
    path = '/media/1tb/Izzy/nets/net6_05-11-2016_12h09m12s.ckpt'
    sess = net.load(path)

    bc = BinaryCamera('./meta.txt')
    bc.open()
    for i in range(4):
        bc.read_frame()    


    num_samples = n
    samples = []
    images = []
    for i in range(num_samples):
        print i
        try:
            while True:
                frame = bc.read_frame()
                frame = inputdata.im2tensor(frame, channels=3)
                cv2.imshow('preview', frame)
                cv2.waitKey(30)            
        except:
            pass
        frame = bc.read_frame()
        frame = similarity.color(frame)
        dists = net.class_dist(sess, frame, 3)
        dists = dists[:2]
        print dists
        images.append(frame)
        samples.append(dists)
        print "Saved sample"    
	np.save('samples.npy', samples)
	np.save('samples_images.npy', images)
    #oa = OutputAnalysis(samples)
    #oa.print_outputs()

def analyze(filenames):
	for filename in filenames:
		samples = np.load(filename)
		oa = OutputAnalysis(samples)
		oa.print_outputs()
		print '\n\n'

def action_analysis(filenames):
	for filename in filenames:
		samples = np.load(filename)
		oa = OutputAnalysis(samples)
		oa.actions()
        print '\n\n'

def test():
    images = np.load('samples_images.npy')
    net = net6_c.NetSix_C()
    path = '/media/1tb/Izzy/nets/net6_05-11-2016_12h09m12s.ckpt'
    sess = net.load(path)
    
    for image in images:
        dists = net.class_dist(sess, image, 3)
        dists = dists[:2]
        print dists

if __name__ == '__main__':
	run_samples(30)
    #run_samples(2)
    #action_analysis(['exp_samples.npy'])
	#analyze(['control_samples.npy', 'exp_samples.npy'])




