from pipeline.bincam import BinaryCamera
import cv2
import time
from options import Options
import numpy as np
from Net.tensor import inputdata
import IPython

bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
frames = []
fourcc = cv2.cv.CV_FOURCC(*'mp4v')

i = 4
overlay = cv2.imread("/home/annal/Izzy/vision_amt/yumiparts/" + str(i) + "-2.png", 1)
ref3d = cv2.imread("/home/annal/Izzy/vision_amt/yumiparts/" + str(i) + "-1.png", 1)
cv2.imshow("reference",ref3d)
#remove red from white 
overlay[:,:,2] = overlay[:,:,2] - overlay[:,:,0]
overlay[:,:,1] = np.zeros((420, 420))
overlay[:,:,0] = np.zeros((420, 420))
#make red yellow
overlay[:,:,1] = overlay[:,:,2]

writer = cv2.VideoWriter("newShapes.mov", fourcc, 10.0, (420,420))
try:
	while True:
	    frame = bc.read_frame()
	    frame = inputdata.im2tensor(frame, channels=3)
	    #frame = np.concatenate((frame, np.ones((420, 420, 1))), axis = 2)
		#yellow - red = green
	    # frame = np.abs(overlay - frame * 255)
	    cv2.imshow("camera",frame)
	    cv2.imwrite("obj11.png", (frame * 255.0).astype('u1'))
 
	    #IPython.embed()

	    cv2.waitKey(30)
	    frames.append(frame)
	    #time.sleep(0.08)

except KeyboardInterrupt:
    pass
#for frame in frames:
#   writer.write((frame * 255.0).astype('u1'))
#writer.release()
