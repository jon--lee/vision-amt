from pipeline.bincam import BinaryCamera
import cv2
import time
from options import Options


from time import time
import IPython
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as LA
from template_finder import TemplateGUI


class VideoMaker(object):
    def __init__(self,rng =[]):
        self.addr = "data/amt/rollouts/"
        self.rollouts = self.compileList(rng)
     


    def compileList(self,rng):
        rollouts = []
        for r in rng:
            rollouts.append("rollout"+str(r))
        return rollouts


    def filmRollout(self,rollout):
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')

        writer = cv2.VideoWriter(self.addr+rollout+'.mov', fourcc, 10.0, (420,420))
        for i in range(0,100):
            img = cv2.imread(self.addr+rollout+'/'+rollout+'_frame_'+str(i)+'.jpg',1)
            writer.write(img)

        writer.release()

    def run(self):

    	for rollout in self.rollouts:
            print rollout
            self.filmRollout(rollout)
            




if __name__ == '__main__':
    print "running"
    rng = [1130,1131,1132,1133,1134,1135,1145]
    ct = VideoMaker(rng)
    ct.run()


# bc = BinaryCamera("./meta.txt")
# bc.open()
# frame = bc.read_frame()
# frames = []

# try:
#     while True:
# 	    frame = bc.read_frame()
# 	    cv2.imshow("camera",frame)

# 	    cv2.waitKey(30)

	    
# 	    frames.append(frame)
# 	    #time.sleep(0.08)
	    
# except KeyboardInterrupt:
#     pass
    
# #for frame in frames:
# #    writer.write(frame)
# #writer.release()
