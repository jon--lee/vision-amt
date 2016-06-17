import cv2
import numpy as np
import sys
# from Net.tensor import inputdata
from options import Options

def im2tensor(im,channels=1):
    """
        convert 3d image (height, width, 3-channel) where values range [0,255]
        to appropriate pipeline shape and values of either 0 or 1
        cv2 --> tf
    """
    shape = np.shape(im)
    h, w = shape[0], shape[1]
    zeros = np.zeros((h, w, channels))
    for i in range(channels):
        #Binary Mask
        zeros[:,:,i] = np.round(im[:,:,i] / 255.0 - .25, 0).astype(int) * 255
        #zeros[:,:,i] = np.round(im[:,:,i] / 255.0, 0)

        #Nomarlized RGB
        #zeros[:,:,i] = im[:,:,i] / 255.0

        #zeros[:,:,i] = im[:,:,i]
    return zeros

def find_centroid(image, template):
	# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	image = im2tensor(image,3).astype('u1')
	template = im2tensor(template,3).astype('u1')
	result = cv2.matchTemplate(image, template, method=cv2.TM_CCOEFF_NORMED)
	temp_shape = template.shape
	# while True:
	# 	cv2.imshow("template", result)
	# 	cv2.waitKey(30)	
	idx = np.argmax(result)
	centroid = (idx % result.shape[1] + template.shape[1]/2, idx/result.shape[1] + template.shape[0]/2)
	return centroid, result

def find_distances(image, templates):
	pass

def display_centroid(image, centroids, result=None):
	for centroid in centroids:
		# cv2.circle(image, center = centroid, radius = 10, color='red')
		cv2.circle(image,centroid, 10, (0,255,0), -1)
		image = im2tensor(image,3).astype('u1')

	return image

		# while True:
		# 	cv2.imshow("target", image)
		# 	# cv2.imshow("result", result)
		# 	a = cv2.waitKey(30)
		# 	if a == 27:
		# 		cv2.destroyWindow('preview')
		# 		break


