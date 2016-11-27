from pipeline.bincam import BinaryCamera
import cv2
import time
from options import Options
import numpy as np
from Net.tensor import inputdata
import IPython

path = 'izzy_net(1).svg'
frame = cv2.imread(path)
print frame.shape
frame = inputdata.im2tensor(frame, channels=3)
cv2.imwrite('masked_' + path, frame *255)