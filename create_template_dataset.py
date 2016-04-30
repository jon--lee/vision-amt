import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import re
import sys
sys.path.insert(0, 'Net/tensor/')
from inputdata import im2tensor
import cv2

root_dir = os.path.expanduser("~/Desktop/research/grasping_dataset/")
rollout_dir = os.path.join(root_dir, 'rollouts/')
templates_dir  = os.path.join(root_dir, 'dataset/templates/')
test_dir  = os.path.join(root_dir, 'dataset/test/')
state_file_path = os.path.join(test_dir, 'states.txt')

def extract_rollouts(dataset='train', n_folders=20, n_images=1):
	"""
	Extracts rollout images from input folder,
	and copies it to dataset folder.
	"""
	for root, dirs, files in os.walk(rollout_dir):
		np.random.shuffle(dirs)
		for i in range(min(n_folders, len(dirs))):
			print "Folder {}".format(i)
			rollout_folder = dirs[i]
			for _, _, files in  os.walk(os.path.join(rollout_dir, rollout_folder)):
				images =  [x for x in files if x.endswith('.jpg')]
				if dataset == 'templates':
					np.random.shuffle(images)
					for im in images[:n_images]:
						src = os.path.join(rollout_dir, rollout_folder, im)
						dst = os.path.join(templates_dir, im)
						copy_processed_image(src, dst)
				elif dataset == 'test':
					im = sorted(images)[-1]
					print "Image: {}".format(im)
					src = os.path.join(rollout_dir, rollout_folder, im)
					dst, new_im_name = label_image(src, im, test_dir)
					copy_processed_image(src, dst)
					copy_state_label(rollout_dir, rollout_folder, im, new_im_name)
		break

def copy_processed_image(src, dst):
	im = cv2.imread(src, 1)
	print im.shape
	processed = im2tensor(im)
	cv2.imshow("original", im)
	cv2.waitKey(10)
	cv2.imshow("filtered", processed)
	cv2.waitKey(10)
	cv2.imwrite(dst, processed)

def copy_state_label(rollout_dir, rollout_folder, old_im_name, new_im_name):
	im_file = os.path.join(rollout_dir, rollout_folder, old_im_name)
	old_state_file_path = os.path.join(rollout_dir, rollout_folder, 'states.txt')
	state_file, old_state_file = open(state_file_path, 'a'), open(old_state_file_path, 'r')
	im_state_line = [line for line in old_state_file if line.startswith(old_im_name)][0]

	state = im_state_line.split()
	state[0] = new_im_name
	state_file.write(' '.join(state) + '\n')

	state_file.close()
	old_state_file.close()


def label_image(src, im_name, test_dir):
	p = subprocess.Popen(["display", src])
	while True:
		label = raw_input("Label 'y' if circle in gripper, 'n' otherwise: ")
		if label == 'y':
			im_name = "success_" + im_name
			break
		elif label == 'n':
			im_name = "fail_" + im_name
			break
		else:
			continue
	dst = os.path.join(test_dir, im_name)
	p.kill()
	return dst, im_name

def check_dirs():
	if not os.path.isdir(templates_dir):
		os.mkdir(templates_dir)
	if not os.path.isdir(test_dir):
		os.mkdir(test_dir)
	if not os.path.isfile(state_file_path):
		os.mknod(state_file_path)

if __name__ == '__main__':
	check_dirs()
	#extract_rollouts(dataset='templates', n_folders=20, n_images=1)
	extract_rollouts(dataset='test', n_folders=50, n_images=1)

