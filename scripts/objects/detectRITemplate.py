from pipeline.bincam import BinaryCamera
import cv2
import time
import numpy as np
from Net.tensor import inputdata
from edgeRICam import getTemplates, processFrame, findInitialMatch, traceObj, updateMatch

bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
#store processed templates; list of lists with various rotations of each template
filtTemplates = getTemplates()
sampleTemps = filtTemplates[0]
filtFrame = processFrame(frame)
pos, foundTemp = findInitialMatch(filtFrame, sampleTemps)
out = traceObj(frame, pos, foundTemp)
cv2.imshow("camera", out)
print("initial match found- press any key to begin")
cv2.waitKey(5000)
try:
	while True:
	    frame = bc.read_frame()
	    filtFrame = processFrame(frame)
        pos, foundTemp = updateMatch(pos, frame, sampleTemps)
        out = traceObj(frame, pos, foundTemp)
        cv2.imshow("camera",out)
        cv2.waitKey(250)
except KeyboardInterrupt:
    pass
