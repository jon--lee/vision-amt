from pipeline.bincam import BinaryCamera
import cv2

bc = BinaryCamera('./meta.txt')
bc.open()
frame = bc.read_frame()
#vc = cv2.VideoCapture(0)
#rval, frame = vc.read()

while True:
	frame = bc.read_frame()
	cv2.imshow('r', frame)

cv2.imwrite('testimage.jpg', frame)
