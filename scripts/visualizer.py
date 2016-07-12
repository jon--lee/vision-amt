#visualizer

import cv2
import time
from options import Options
import numpy as np
from Net.tensor import inputdata, net3,net4,net5,net6

def command_to_line(image, delta, origin):
    pass

def state_to_pixel(image, ):
	pass

def pixelstoMeters(val):
    return 0.5461/420*val

def metersToPixels(val):
    return 420/0.5461*val


def draw_result(img):
        # ### Test angles
    #hardcoded dimensions
    L = 420 #size in pixels of the viewscreen
    base_ext = .2 # The distance in meters of the base to the edge of the screen
    ang_offset = np.pi + 0.26760734641 # the offset of horizontal from the angle
    ###

    grip_ang = -(float(state[0]) - ang_offset)
    grip_ext = self.metersToPixels(float(state[1]) + base_ext)
    base = [grip_ext*np.cos(grip_ang),grip_ext*np.sin(grip_ang)]
    # grip_ang = np.arctan2(d_vec[1]-base[0], d_vec[0]-base[1])



    grip_L = (420,int(grip_pos[1] + (L-grip_pos[0])*np.tan(grip_ang))) #the start position of the gripper
    grip_post = (int(grip_pos[0]), int(grip_pos[1]))


    # gc_est = [gc_pos[0], gc_pos[0] * np.tan(gc_ang) + base[1]]
    cv2.line(img, grip_L, grip_post, 255, 2)
    cv2.line(img, gc_L, gc_est, 255, 2)

    self.writer.write(img)
    cv2.imshow("figue",img) # draws a line along the length of the gripper
    cv2.waitKey(30)
