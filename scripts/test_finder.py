import cv2
import numpy as np
import object_finder
from pipeline.bincam import BinaryCamera
import time
from options import Options
import numpy as np

# for i in range(4,8):
# 	template_path = '/home/annal/Izzy/vision_amt/scripts/objects/obj' + str(i) + ".png"
# 	template = cv2.imread(template_path)
# 	# while True:
# 	# 	cv2.imshow("template", template)
# 	# 	cv2.waitKey(30)
# 	image = cv2.imread('/home/annal/Izzy/vision_amt/scripts/rollout1598_frame_0.jpg')
# 	print image.shape, template.shape

# 	centroid, result = object_finder.find_centroid(image, template)
# 	object_finder.display_centroid(image, [centroid], result)


bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
frames = []
fourcc = cv2.cv.CV_FOURCC(*'mp4v')

templates = []
centroids = []
for i in range(4,8):
	template_path = '/home/annal/Izzy/vision_amt/scripts/objects/obj' + str(i) + ".png"
	templates.append(cv2.imread(template_path))


writer = cv2.VideoWriter("hardware_reset4.mov", fourcc, 10.0, (420,420))
try:
	while True:
		frame = bc.read_frame()
		centroids = []
		for template in templates:
			centroid, result = object_finder.find_centroid(frame, template)
			centroids.append(centroid)
		frame = object_finder.display_centroid(frame, centroids)
		cv2.imshow("camera",frame)
		cv2.waitKey(30)
except KeyboardInterrupt:
	pass

