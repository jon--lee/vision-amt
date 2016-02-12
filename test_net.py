from options import AMTOptions
import random
import tensorflow as tf
import cv2
from Net.tensor import inputdata, net3
import IPython


class NetTest(object):

    def __init__(self,net_path):
        self.tf_net = net3.NetThree()
        self.tf_net_path = net_path
        self.sess = net.load(var_path=self.tf_net_path)


    def rollCall(self):
        train_path = AMTOptions.train_file
        test_path = AMTOptions.test_file
        deltas_path = AMTOptions.deltas_file

        print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path

        train_file = open(train_path, 'w+')
        test_file = open(test_path, 'w+')
        deltas_file = open(deltas_path, 'r')

        for line in range(100):            
            path = AMTOptions.grayscale_dir
            labels = line.split()
            img_name = path+labels[0]
            deltas_t = labels[1:5]
            img = cv2.imread(img_name)
            img = np.reshape(img, (250, 250, 1))
            print "TRUE DELTAS ", deltas_t
            print "NET DELTAS ", self.getNetOutput(deltas)

    def getNetOutput(self):
        deltas = net.output(sess, binary_frame)


if __name__ == '__main__':

    nt = NetTest(net_path = "")
    nt.rollCall()

