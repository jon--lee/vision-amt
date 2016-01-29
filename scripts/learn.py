"""
Learn from xbox controller inputs
Comment or uncomment "options.record = True"
Specify dataset_name or leave blank to make a new one
"""

from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *

from pipeline.bincam import BinaryCamera
from options import Options
from lfd import LFD
import numpy as np
import caffe
import time
import cv2
import os
import datetime



bincam = BinaryCamera('./meta.txt')
bincam.open()

options = Options()


t = TurnTableControl() # the com number may need to be changed. Default of com7 is used
izzy = PyControl(115200, .04, [0,0,0,0,0],[0,0,0]); # same with this
c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])

"""#options.model_path = "./net/nets/net3/model3.prototxt"
#options.weights_path = "./net/nets/net3/weights_iter_115.caffemodel"
options.model_path = "./net/nets/net2/model2.prototxt"
options.weights_path = "./net/nets/net2/weights_iter_240.caffemodel"
"""
options.record = True

lfd = LFD(bincam, izzy, t, c, options=options)

dataset_name = "" # if you want to add to existing dataset, specifiy directory name (not path).
                  # else a new one is created in datetime format
lfd.learn(dataset_name)

