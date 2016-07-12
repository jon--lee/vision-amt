import cv2
import time
from options import Options
import numpy as np
# from Net.tensor import inputdata

template = cv2.imread("scripts/objects/template.jpg", -1)
# overlay[:,:,2] = overlay[:,:,2] - overlay[:,:,0]
# overlay[:,:,1] = np.zeros((420,420))
# overlay[:,:,0] = np.zeros((420,420))
# overlay[:,:,0] = overlay[:,:,2]
# overlay[:,:,2] = np.zeros((420,420))
shape = np.shape(template)
h, w = shape[0], shape[1]
zeros = np.zeros((h, w, 3))
for i in range(3):
    #Binary Mask
    zeros[:,:,i] = np.round(template[:,:,i] / 255.0 - .25, 0)
# m_template = inputdata.im2tensor(template, channels = 3)
while True:
	cv2.imshow("camera", zeros)
	cv2.waitKey(30)
	# result = cv2.imwrite("scripts/objects/masked_gripper.jpg", zeros, [100])
	# zeros = cv2.imread("scripts/objects/masked_gripper.jpg", -1)
print result