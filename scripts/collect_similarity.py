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
        images.append(frame)
        print "Saved sample"    
        np.save('samples_images.npy', images)


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




