from pipeline.bincam import BinaryCamera
import cv2
import time
from options import Options
from Net.tensor import inputdata

bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
frames = []
fourcc = cv2.cv.CV_FOURCC(*'mp4v')

# open with transparency
overlay = cv2.imread("0.png", -1)

writer = cv2.VideoWriter("hardware_reset4.mov", fourcc, 10.0, (420,420))
try:
    while True:
	    frame = bc.read_frame()
	    frame = inputdata.im2tensor(frame, channels=3)
        for x in range(0, 420):
            for y in range(0, 420):
                overPix = overlay[x, y]
                backPix = frame[x, y]
                if (overPix[4] != 0):
                    frame[x, y] = [255, 0, 0]
	    cv2.imshow("camera",frame)
	    cv2.waitKey(30)
	    frames.append(frame)
	    #time.sleep(0.08)

except KeyboardInterrupt:
    pass

#for frame in frames:
#   writer.write((frame * 255.0).astype('u1'))
#writer.release()
