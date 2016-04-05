import cv2
import numpy as np
im = cv2.imread('/home/annal/Izzy/vision_amt/data/amt/rollouts/rollout82/rollout82_frame_19.jpg')

imb = im[0:100,0:100,0]
print imb.shape
print imb
cv2.imwrite('binarytesting.jpg', imb)

