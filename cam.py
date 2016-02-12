from pipeline.bincam import BinaryCamera
import cv2
import time
from options import Options

bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
frames = []
fourcc = cv2.cv.CV_FOURCC(*'mp4v')

writer = cv2.VideoWriter("testing123.mov", fourcc, 10.0, (Options.WIDTH,Options.HEIGHT))
try:
    while True:
	    frame = bc.read_grayscale_frame()
	    cv2.imshow("camera",frame)

	    cv2.waitKey(30)

	    
	    frames.append(frame)
	    #time.sleep(0.08)
	    
except KeyboardInterrupt:
    pass
    
#for frame in frames:
#    writer.write(frame)
#writer.release()
