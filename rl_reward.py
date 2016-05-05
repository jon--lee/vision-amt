from time import time
import IPython
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import re
import sys


class RL_reward():
	def __init__(self):
		# Locations of dataset
		self.dataset_addr = os.path.expanduser("~/Desktop/research/grasping_dataset/dataset/")
		self.test_scenes = os.path.join(self.dataset_addr, 'test')
		self.test_scene_states = os.path.join(self.test_scenes, 'states.txt')
		self.circle_templates_dir = os.path.join(self.dataset_addr, 'templates/circle/')
		self.gripper_templates_dir = os.path.join(self.dataset_addr, 'templates/gripper/')
		# Load dataset
		self.load_datasets()

		# Constants
		self.success_reward = 1.0

		# Constatnts (run_gc_labels.py)
		self.debug = False
		self.prev_pos = np.zeros(2)
		self.first = True

	def load_datasets(self):
		"""
		Loads the datasets.
		"""
		for root, dirs, files in os.walk(self.circle_templates_dir):
			self.circle_templates = [self.read_filter_img(os.path.join(self.circle_templates_dir, x)) for x in files if x.endswith('.jpg')]
			#self.circle_templates = [cv2.imread(os.path.join(self.circle_templates_dir, x), 1) for x in files if x.endswith('.jpg')]
			print "Circle Templates Shapes: "
			print [x.shape for x in self.circle_templates]
			break
		for root, dirs, files in os.walk(self.gripper_templates_dir):
			#x = files[0]
			#print cv2.imread(os.path.join(self.gripper_templates_dir, x), 1)
			self.gripper_templates = [self.read_filter_img(os.path.join(self.gripper_templates_dir, x)) for x in files if x.endswith('.jpg')]
			#self.gripper_templates = [cv2.imread(os.path.join(self.gripper_templates_dir, x), 1) for x in files if x.endswith('.jpg')]
			print "Gripper Templates Shapes: "
			print [x.shape for x in self.gripper_templates]
			break

	def read_filter_img(self, im_path):
		im = plt.imread(im_path)
		filtered = 255.0 * self.im2tensor_binary(im, channels=3)
		#filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
		return filtered.astype(np.uint8)

	def get_pos(self, templates, img, gripper=False):
		"""
		Uses template matching to determine the location of the object.

		Parameters:
		templates: list of images
			List of templates to try
		img: image
			Target image
		gripper: boolean
			Determines whether to track the gripper (True) or golden circle (False)

		Returns:
		pos: two-tuple of floats
			(x, y) position of the center of the object
		"""
		img2 = img.copy()
		if(gripper):
			methods = ['cv2.TM_SQDIFF']
		else: 
			methods = ['cv2.TM_CCORR_NORMED']


		for meth in methods:
			#img = self.im2tensor_binary(img2.copy(), channels=3)
			img = img2.copy()
			plt.imshow(img)
			method = eval(meth)

			# Create convolutional maps
			# print self.circle_templates
			print [x.shape for x in templates]
			conv_maps = [cv2.matchTemplate(img,temp,method) for temp in templates]
			loc_maps = [cv2.minMaxLoc(res) for res in conv_maps]

			# Find location of object, corresponding to maximum output of all convolutional maps
			max_ind = max(range(len(loc_maps)), key=lambda x: loc_maps[x][1])
			min_val, max_val, min_loc, max_loc = loc_maps[max_ind]
			# print min_val, max_val, min_loc, max_loc
			# min_val = np.mean([x[0] for x in loc_maps])
			# max_val = np.mean([x[1] for x in loc_maps])
			# min_loc = (np.mean([x[2][0] for x in loc_maps]), np.mean([x[2][1] for x in loc_maps]))
			# max_loc = (np.mean([x[3][0] for x in loc_maps]), np.mean([x[3][1] for x in loc_maps]))
			max_template = templates[max_ind]
			w = max_template.shape[0]
			h = max_template.shape[1]

			# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
			if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
				top_left = min_loc
			else:
				top_left = max_loc

			pos = np.zeros(2)
			pos[0] = top_left[0]+w/2
			pos[1] = top_left[1]+h/2

			if not gripper:
				pos = self.lowPass(pos)

			if self.debug:

				top_left =(int(pos[0] - w/2),int(pos[1] - h/2))

				bottom_right = (top_left[0] + w, top_left[1] + h)

				cv2.rectangle(img,top_left, bottom_right, 255, 2)


				self.writer.write(img)
				cv2.imshow("figue",img)
				cv2.waitKey(30)
		return pos

	def lowPass(self, cur_pos):
		dist = la.norm(cur_pos-self.prev_pos)
		
		if(self.first): 
			self.first = False
			self.prev_pos = cur_pos
			return cur_pos
		if(dist > 100):
			return self.prev_pos
		else: 
			self.prev_pos = cur_pos
			return cur_pos

	def im2tensor_binary(self, im, channels=3):
	    """
	        convert 3d image (height, width, 3-channel) where values range [0,255]
	        to appropriate pipeline shape and values of either 0 or 1
	        cv2 --> tf
	    """
	    shape = np.shape(im)
	    print "shape", shape
	    h, w = shape[0], shape[1]
	    zeros = np.zeros((h, w, channels))
	    for i in range(channels):
	        #Binary Mask
	        zeros[:,:,i] = np.round(im[:,:,i] / 255.0 - .25, 0)
	    return zeros

	def compute_bounding_box(self, angle, base_coords, w=1.0, h=1.0):
		"""
		Computes coordinates of bounding box.
		
		Parameters:
		angle: float
			Angle of the gripper arm.
		base_coords: tuple (x, y)
			Coordinates of the base of the rectangle. 
		w: float
			Width of the bounding box, extending from the base.
		h: float
			Height of the bounding box, extending from
			the base in both directions. 

		Output:
		points: list of 4 tuples (x, y)
			Returns corner of points of bounding box
			in clockwise order.
		"""
		base_coords = np.array(base_coords)
		
		# Create original box, origin at base of box
		original_box = np.array([[0, h/2], [0, -h/2], [w, -h/2], [w, h/2]])

		# Calculate rotated box
		rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
		rot_box = np.dot(rot_mat, original_box.T).T

		# Add base_coords offset to each corner
		points = base_coords + rot_box
		return points

	def compute_point_label(self, point, box_coords):
		"""
		Computes whether the given point is inside 
		the rectangle, given by its corner coordinates.

		Parameters:
		point: tuple (x,y)
		box_coords: list of 4 tuples (x, y).
			Assumes corner coordinates are in clockwise order.

		Returns:
		boolean: Indicates whether point is inside rectangle.

		Source: http://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
		"""
		x1, y1 = box_coords[0][0], box_coords[0][1]
		x2, y2 = box_coords[1][0], box_coords[1][1]
		x3, y3 = box_coords[2][0], box_coords[2][1]
		x4, y4 = box_coords[3][0], box_coords[3][1]
		p21 = (x2 - x1, y2 - y1)
		p41 = (x4 - x1, y4 - y1)

		p21magnitude_squared = p21[0] ** 2 + p21[1] ** 2
		p41magnitude_squared = p41[0] ** 2 + p41[1] ** 2

		x, y = point[0], point[1]

		p = (x - x1, y - y1)

		if 0 <= p[0] * p21[0] + p[1] * p21[1] and p[0] * p21[0] + p[1] * p21[1] <= p21magnitude_squared:
			if 0 <= p[0] * p41[0] + p[1] * p41[1] and p[0] * p41[0] + p[1] * p41[1] <= p41magnitude_squared:
				return True
			else:
				return False
		else:
			return False

	def reward_function(self, im, state, dist=True, success=False):
		"""
		Reward function for the current state, given an input 
		image and state.

		Parameters:
		im: image
			Input image of the state.
			Assumes image is already filtered.
		state: numpy array
			Internal state of the gripper.
		dist: boolean
			Whether to reward for distance.
		"""

		im = (255.0 * self.im2tensor_binary(im, channels=3)).astype(np.uint8)
		gc_pos = self.get_pos(self.circle_templates, im, gripper=False)
		gripper_pos = self.get_pos(self.gripper_templates, im, gripper=True)
		bounding_box = self.compute_bounding_box(-state[0], gripper_pos, w=125.0, h=50.0)
		point_label = self.compute_point_label(gc_pos, bounding_box)

		grip_reward = int(point_label) * self.success_reward

		# Debugging
		print "gc_pos", gc_pos
		print "gripper_pos", gripper_pos
		print "bounding_box", bounding_box
		print "point_label", point_label
		print "success", success
		print ""

		cv2.circle(im, tuple(gc_pos.astype(int)), radius=1, color=0, thickness=3)
		cv2.circle(im, tuple(gripper_pos.astype(int)), radius=1, color=0, thickness=3)
		for i in range(len(bounding_box)-1):
			cv2.line(im, tuple(bounding_box[i].astype(int)), tuple(bounding_box[i+1].astype(int)), 0)
		cv2.line(im, tuple(bounding_box[0].astype(int)), tuple(bounding_box[3].astype(int)), 0)
		cv2.imshow('figure', im)
		cv2.waitKey(100)
		if dist:
			dist_reward = -la.norm(gc_pos - gripper_pos)
			return dist_reward + grip_reward
		else:
			return grip_reward

	def test_point_label(self):
		"""
		Tests accuracy of point label, based on 
		pre-labeled template dataset
		
		Output:
		acc: float
			Accuracy of point label.
		"""
		
		# Loading data
		for root, dirs, files in os.walk(self.test_scenes):
			success_files = [x for x in files if x.startswith('success') and x.endswith('.jpg')]
			fail_files = [x for x in files if x.startswith('fail') and x.endswith('.jpg')]
			success_states = []
			for success in success_files:
				states = open(self.test_scene_states, 'r')
				state_line = [line for line in states if success in line][0]
				state = [float(x) for x in state_line.split()[1:5]]
				success_states.append(state)
				states.close()
			fail_states = []
			for fail in fail_files:
				states = open(self.test_scene_states, 'r')
				state_line = [line for line in states if fail in line][0]
				state = [float(x) for x in state_line.split()[1:5]]
				fail_states.append(state)
				states.close()
			success_images = [plt.imread(os.path.join(self.test_scenes, im_file)) for im_file in success_files]
			fail_images = [plt.imread(os.path.join(self.test_scenes, im_file)) for im_file in fail_files]
			success_labels = []

			# Process images
			for i in range(len(success_images)):
				im = success_images[i]
				state = success_states[i]
				point_label = self.reward_function(im, state, dist=False, success=True)
				success_labels.append(point_label)
			fail_labels = []
			for i in range(len(fail_images)):
				im = fail_images[i]
				state = fail_states[i]
				point_label = self.reward_function(im, state, dist=False, success=False)
				fail_labels.append(point_label)
			acc = np.mean(np.concatenate([np.array(success_labels) == 1, np.array(fail_labels) == 0]))
			return acc

if __name__ == '__main__':
	reward_obj = RL_reward()
	acc = reward_obj.test_point_label()
	print "Accuracy of Point Label: ", acc