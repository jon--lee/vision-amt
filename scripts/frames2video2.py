import cv2
import numpy as np
import os
from options import Options

def num_from_filename(filename):
	return int(filename.split('frame_', 1)[1][:-4])


if '__main__' == __name__:
	directory = "./data/amt/rollouts/rollout4/"
	fourcc = cv2.cv.CV_FOURCC(*'mp4v')
	writer = cv2.VideoWriter("rollout_testing.mov", fourcc, 10.0, (Options.WIDTH,Options.HEIGHT))

	filenames = [directory + fn for fn in os.listdir(directory) if fn.endswith('.jpg') ]
	filenames = sorted(filenames, key=num_from_filename)
	
	for fn in filenames:
		im = cv2.imread(fn)
		writer.write(im)
		
	writer.release()
