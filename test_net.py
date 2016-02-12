from options import AMTOptions
import random
import tensorflow as tf
import cv2
from Net.tensor import inputdata, net3
import IPython
import numpy as np
import np.linalg as LA


class NetTest(object):

    def __init__(self,net_path):
        self.tf_net = net3.NetThree()
        self.tf_net_path = net_path
        self.sess = self.tf_net.load(var_path=self.tf_net_path)
        self.vals = []

    def computerER(self):
        sm = 0.0
        for val in self.vals):
            sm += val
        print "AVERAGE SQUARED EUCLIDEAN LOSS ",sm/len(self.vals)
        
    def rollCall(self):
        train_path = AMTOptions.train_file
        test_path = AMTOptions.test_file
        deltas_path = AMTOptions.deltas_file

        print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path

        deltas_file = open(deltas_path, 'r')

        for line in deltas_file:            
            path = AMTOptions.grayscales_dir
            labels = line.split()
            img_name = path+labels[0]
            deltas_t = labels[1:5]
            img = cv2.imread(img_name,0)
            img = np.reshape(img, (250, 250, 1))
            net_v = np.array(self.getNetOutput(img))
            true_v = np.array(deltas_t)
            err = 0.5*LA.norm(net_v-true_v)**2
            self.vals.append(err)

        self.computerER()


            

    def getNetOutput(self,img):
        return self.tf_net.output(self.sess, img)


if __name__ == '__main__':

    nt = NetTest(net_path ='/home/annal/Izzy/vision_amt/Net/tensor/net3/net3_02-12-2016_00h08m36s.ckpt')
    nt.rollCall()

