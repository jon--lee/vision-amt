from pipeline.bincam import BinaryCamera
import cv2
import time
import numpy as np
from Net.tensor import inputdata
from fastEdges import getTemplates, processFrame, findInitialMatch, traceObj, findMatch

bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
#store processed templates, find initial match
filtTemplates = getTemplates()
filtT = templates[0]
filtFrame = processFrame(frame)
pos = findInitialMatch(filtFrame, filtT)
out = traceObj(frame, pos, filtT)
cv2.imshow("camera", out)
print("initial match found- press any key to begin")
cv.waitKey(0)
try:
	while True:
	    frame = bc.read_frame()
	    filtFrame = processFrame(frame)
        pos = findMatch(pos)
        out = traceObj(frame, pos, filtT)
	    cv2.imshow("camera",out)
	    cv2.waitKey(250)
except KeyboardInterrupt:
    pass
