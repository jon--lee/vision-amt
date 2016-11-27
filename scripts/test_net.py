from options import AMTOptions
import random
import tensorflow as tf
import cv2
from Net.tensor import inputdata, net3,net4,net6,net6_c
import IPython
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


class NetTest(object):

    def __init__(self,net_path):
        self.tf_net = net6.NetSix()
        self.tf_net_path = net_path
        self.sess = self.tf_net.load(var_path=self.tf_net_path)
        self.vals = []


    def computerER(self):
        sm = 0.0
        rot_err = 0.0
        fore_err = 0.0
        for val in self.vals:
            sm += np.sum(val)
            rot_err += val[0]
            fore_err += val[1]


        print "AVERAGE SQUARED EUCLIDEAN LOSS ",sm/len(self.vals), " rotation: ", rot_err/len(self.vals), " forward: ", fore_err/len(self.vals)#*0.25

    def scale(self,deltas):
        deltas[0] = float(deltas[0])*0.02
        deltas[1] = float(deltas[1])*0.006
        deltas[2] = float(deltas[2])/0.005
        deltas[3] = float(deltas[3])/0.2
        return deltas
        
    def rollCall(self):
        train_path = AMTOptions.train_file
        test_path = AMTOptions.test_file
        deltas_path = AMTOptions.deltas_file

        print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path

        deltas_file = open(deltas_path, 'r')

        i = 0
        for line in deltas_file:            
            path = AMTOptions.colors_dir
            labels = line.split()
            img_name = path+labels[0]
            #deltas_t = labels[1:5]
            img = cv2.imread(img_name,1)
            img = np.reshape(img, (250, 250, 3))
            net_v = self.scale(np.array(self.getNetOutput(img),dtype=np.float32))
            # dists =  self.tf_net.class_dist(self.sess, img)
            # plt.subplot(2,1,1)
            # plt.plot(dists[0,:])
            
            # plt.subplot(2,1,2)
            # plt.plot(dists[1,:])
            # plt.show(block=False)

            deltas_t = [float(label) for label in labels[1:]]
            

            # true_v = np.array(self.scale(deltas_t),dtype=np.float32)
            true_v = np.array(deltas_t,dtype=np.float32)
            true_v[0] = true_v[0]/10.0
            true_v[1] = true_v[1]*.6
            true_v[2] = 0
            true_v[3] = 0
            # print net_v, true_v
            # print "NET ",net_v
            # print "TRUE ",true_v
            err1 = LA.norm(net_v[0]-true_v[0])
            err2 = LA.norm(net_v[1]-true_v[1])
            # if err1 > 0:
            #     # print "error"
            #     self.vals.append(1)
            # else:
            #     self.vals.append(0)
            # if err2 > 0:
            #     # print "error"
            #     self.vals.append(1)
            # else:
            #     self.vals.append(0)
            if i % 500 == 0:
                print i
            i += 1

            #err = 0.5*LA.norm(net_v[0]-true_v[0])**2
            self.vals.append(np.array((err1, err2)))

        #self.computerER()


            

    def getNetOutput(self,img):
        return self.tf_net.output(self.sess, img,channels=3)


if __name__ == '__main__':
    nt = NetTest(net_path =sys.args[1])
    nt.rollCall()
    nt.computerER()

