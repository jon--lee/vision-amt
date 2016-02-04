import cv2
import numpy as np
import datetime
import os
from options import Options

def dist(loc1, loc2):
    d_squared = dist_squared(loc1, loc2)
    return d_squared ** (.5)


def dist_squared(loc1, loc2):
    diffx = loc1[0] - loc2[0]
    diffy = loc1[1] - loc2[1]
    return diffx * diffx + diffy * diffy


def mapRange(frame, loc):
    """
    Given a point, map pixel and surrounding pixels 
    to intermediate green if it is considered dark or black
    """
    x, y = loc
    blue, green, red = frame[y,x]
    blueLow, greenLow, redLow = BinaryCamera.lower_black
    blueUp, greenUp, redUp = BinaryCamera.upper_black
    if greenLow < green \
        and greenUp > green \
        and blueLow < blue \
        and blueUp > blue \
        and redLow < red \
        and redUp > red:

        frame[y, x] =  BinaryCamera.intermediate_green
        frame[y+1, x] = BinaryCamera.intermediate_green
        frame[y, x+1] = BinaryCamera.intermediate_green
        frame[y+1, x+1] = BinaryCamera.intermediate_green


class BinaryCamera():
    
    tolerance = 2000                        # tolerance from distance around ring's border
    lower_green = np.array([30,10,0], dtype=np.uint8)       # table's lowest shade of green
    upper_green = np.array([70,140,180], dtype=np.uint8)    # highest green
    lower_black = (0,0,0)                   # ring's lowest black
    upper_black = (130,130,130)             # highest black
    intermediate_green = [120, 180, 120]    # intermediate green value (between range)


    def __init__(self, meta='./pipeline/meta.txt'):
        """
        meta -  path to file with calibration data (x,y\ndistance), e.g.
                212,207
                200.85
        """
        self.vc = None

        f = open(meta, 'r')
        self.maxRedLoc = [ int(x) for x in f.readline().split(',') ]
        self.d = int(float(f.readline()))
        self.d_squared = self.d * self.d
        f.close()
        self.recording = []
        self.states = []
        
    def open(self):
        self.vc = cv2.VideoCapture(0)
        
    def close(self):
        if self.is_open():
            self.vc.release()
    
    def is_open(self):
        return self.vc is not None and self.vc.isOpened()

    def read_frame(self, show=False, record=False, state=None):
        """ Returns cropped frame of raw video """
        rval, frame = self.vc.read()
        frame = frame[0+Options.OFFSET_Y:Options.HEIGHT+Options.OFFSET_Y, 0+Options.OFFSET_X:Options.WIDTH+Options.OFFSET_X]
        frame = cv2.resize(frame, (420, 420))
        if record:
            self.recording.append(frame)
            self.states.append(state)
        if show:
            cv2.imshow("preview", frame)
        return frame
        
    def read_binary_frame(self, show=False, record=False, state=[0.0,0.0,0.0,0.0,0.0,0.0]):
        """ Returns a cropped binary frame of the video
        Significantly slower than read_frame due to the pipeline. """
        
        rval, frame = self.vc.read()
        frame = frame[0+Options.OFFSET_Y:Options.HEIGHT+Options.OFFSET_Y, 0+Options.OFFSET_X:Options.WIDTH+Options.OFFSET_X]        
        frame_binary = self.pipe(frame)
        
        if record:
            self.recording.append(frame)
            self.states.append(state)
        if show:
            cv2.imshow("binary", frame_binary)
            
        return frame_binary
        
    def pipe(self, frame, h=125, w=125):
        """ sends frame through cv2 pipeline to render
        binary image of original frame. Assumes calibration """

        for i in range( int(self.maxRedLoc[1] - 2*self.d), int(self.maxRedLoc[1] + 2*self.d), 3 ):
            for j in range( int(self.maxRedLoc[0] - 2*self.d), int(self.maxRedLoc[0] + 2*self.d), 3 ):
                if abs(dist_squared((j, i), self.maxRedLoc) - self.d_squared) < BinaryCamera.tolerance:
                    # map dark ring to a shade of green
                    mapRange(frame, (j, i))   

        frame = cv2.resize(frame, (h, w))                 
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_binary = cv2.inRange(frame_hsv, BinaryCamera.lower_green, BinaryCamera.upper_green)
        frame_binary = 255 - frame_binary
        frame_binary = cv2.medianBlur(frame_binary,7)
    
        return frame_binary

    def destroy(self):
        cv2.destroyAllWindows()

    def save_recording(self):
        """
        save recording to a folder named "data_dir/frames/recording_{datetime}"
        such that it does not overwrite another recording
        """
        path = Options.data_dir + "record_frames/recording_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + "/"
        print "Saving to " + path
        os.makedirs(path) 
        i = 0
        f = open(path + "states.txt", "w+")
        for frame, state in zip(self.recording, self.states):
            name = "frame_" + str(i) + ".jpg"
            cv2.imwrite(path + name, frame)
            f.write(name + self.state2str(state) + "\n") 
            i+=1
        f.close()
        self.recording = []
        self.states = []

    def state2str(self, state):
        string = ""
        for s in state:
            string += " " + str(s)

        return string
        
        
        
