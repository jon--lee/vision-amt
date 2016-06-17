import cv2
import numpy as np
import object_finder
from pipeline.bincam import BinaryCamera
import time
from options import Options
import numpy as np

global centroid

def get_center(event, x, y, flags, param):
	global centroid

	if event == cv2.EVENT_LBUTTONDOWN:
		centroid = (x, y)

def centers(bc=None):
	global centroid
	if bc is None:
		bc = BinaryCamera("./meta.txt")
		bc.open()
		frames = []
		fourcc = cv2.cv.CV_FOURCC(*'mp4v')
		writer = cv2.VideoWriter("hardware_reset4.mov", fourcc, 10.0, (420,420))
	frame = bc.read_frame()
	cv2.imshow("image",frame)
	cv2.waitKey(30)


	centroid = (-1, -1)
	centroids = []
	cv2.setMouseCallback("image", get_center)

	while True:
		frame = bc.read_frame()
		for center in centroids:
			cv2.circle(frame,center, 10, (0,255,0), -1)
		cv2.imshow("image",frame)
		cv2.waitKey(30)


		if centroid[0] != -1:
			centroids.append(centroid)
			centroid = (-1, -1)
		a = cv2.waitKey(30)
		if a == 27:
			cv2.destroyWindow('image')
			break

	return centroids

def max_distance(centroids):
	values = []
	for center in centroids:
		mind = 99999999999
		for other in centroids:
			curd = np.linalg.norm(np.array(center) - np.array(other))
			if curd == 0:
				continue
			if mind > curd:
				mind = curd
		values.append(mind)
	return np.max(values)

# print max_distance(centers())