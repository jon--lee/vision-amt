from pipeline.bincam import BinaryCamera
import cv2
import time
from options import Options
import numpy as np
from Net.tensor import inputdata

bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
frames = []
fourcc = cv2.cv.CV_FOURCC(*'mp4v')

# open with transparency
i = 1
overlay = cv2.imread("yumiparts/" + str(i) + "-2.png", -1)
ref3d = cv2.imread("yumiparts/" + str(i) + "-1.png", -1)
cv2.imshow("reference",ref3d)
#convert red of overlay to yellow (b = 0, g = 255, r = 255)
overlay[:,:,1] = overlay[:,:,2]

writer = cv2.VideoWriter("hardware_reset4.mov", fourcc, 10.0, (420,420))
try:
	while True:
	    frame = bc.read_frame()
	    frame = inputdata.im2tensor(frame, channels=3)
	    #frame = np.concatenate((frame, np.ones((420, 420, 1))), axis = 2)
		#yellow - red = green
	    frame = overlay - frame
	    cv2.imshow("camera",frame)
	    cv2.waitKey(30)
	    #frames.append(frame)
	    #time.sleep(0.08)

except KeyboardInterrupt:
    pass

#for frame in frames:
#   writer.write((frame * 255.0).astype('u1'))
#writer.release()
