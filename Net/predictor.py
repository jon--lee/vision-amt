import caffe
import hdf
import numpy as np
import os

MODEL = './nets/net4/model4.prototxt'
WEIGHTS = './nets/net4/weights_iter_180.caffemodel'

net = caffe.Net(MODEL, WEIGHTS, caffe.TEST)

f = open('./hdf/test.txt', 'r')

for line in f:
    path = line.split(' ')[0]
    image = caffe.io.load_image(path)

    data4D = np.zeros([1,3,125,125])
    data4D[0,0,:,:] = image[:,:,0]
    data4D[0,1,:,:] = image[:,:,1]
    data4D[0,2,:,:] = image[:,:,2]

    net.forward_all(data=data4D)
    data = net.blobs['out'].data.copy()
    s = ""
    for x in data[0]:
        #x = (x - .0) * 150.0
        #if abs(x) < 20.0:
        #    s += " 0.0"
        #else:
        s += " " + str(x)
    print path + s
