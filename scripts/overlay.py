from pipeline.bincam import BinaryCamera
import cv2
import time
from options import Options
import numpy as np
from Net.tensor import inputdata


def overlay(frame, o):
    o = o.copy()
    o[:,:,0] = o[:,:,2]
    o[:,:,1] = np.zeros((250, 250))
    o[:,:,2] = np.zeros((250, 250))
    #frame = np.concatenate((frame, np.ones((250, 250, 1))), axis = 2)
    frame = frame + o
    return frame


#overlay = cv2.imread("objects/synth/0.png", -1)



# overlay[:,:,2] = overlay[:,:,2] - overlay[:,:,0]
# overlay[:,:,1] = np.zeros((420,420))
# overlay[:,:,0] = np.zeros((420,420))
# overlay[:,:,0] = overlay[:,:,2]
# overlay[:,:,2] = np.zeros((420,420))
