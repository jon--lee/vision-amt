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
overlay = cv2.imread("scripts/0.png", -1)
overlay[:,:,2] = overlay[:,:,2] - overlay[:,:,0]
overlay[:,:,1] = np.zeros((420,420))
overlay[:,:,0] = np.zeros((420,420))
overlay[:,:,0] = overlay[:,:,2]
overlay[:,:,2] = np.zeros((420,420))

writer = cv2.VideoWriter("hardware_reset4.mov", fourcc, 10.0, (420,420))
try:
	while True:
	    frame = bc.read_frame()
	    frame = inputdata.im2tensor(frame, channels=3)
	    frame = np.concatenate((frame, np.ones((420, 420, 1))), axis = 2)
	    # for x in range(0, 420):
	    #     for y in range(0, 420):
	    #         overPix = overlay[x, y]
	    #         backPix = frame[x, y]
	    #         if (overPix[3] != 0 and backPix[2] != 0):
	    #             frame[x, y] = [255, 0, 255]
	    #         elif (overPix[3] != 0):
	    #             frame[x, y] = [255, 0, 0, 1]    
	    # overlay = overlay

	    frame = frame + overlay

	    # cv2.imshow("camera", overlay)
	    cv2.imshow("camera",frame)
	    cv2.waitKey(30)
	    #frames.append(frame)
	    #time.sleep(0.08)

except KeyboardInterrupt:
    pass

#for frame in frames:
#   writer.write((frame * 255.0).astype('u1'))
#writer.release()
