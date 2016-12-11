# from pipeline.bincam import BinaryCamera
# import cv2
# import time
# from options import Options


from time import time
import IPython
import cv2, sys
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as LA

from options import AMTOptions
from template_finder import TemplateGUI


class VideoMaker(object):
    def __init__(self,rng =[],addr="data/amt/supervised_rollouts/Aimee_rollouts/",supervised=True):
        self.addr = addr
        if supervised:
            self.rollouts = self.compileList(rng)
        else:
            self.rollouts = self.compileRol(rng)
     

    def compileRol(self,rng):
        rollouts = []
        for r in rng:
            rollouts.append("rollout"+str(r))
        return rollouts


    def compileList(self,rng):
        rollouts = []
        for r in rng:
            rollouts.append("supervised"+str(r))
        return rollouts


    def filmRollout(self,rollout):
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')

        writer = cv2.VideoWriter(self.addr+rollout+'.mov', fourcc, 10.0, (420,420))

        for i in range(0,100):
            img = cv2.imread(self.addr+rollout+'/'+rollout+'_frame_'+str(i)+'.jpg',1)
            writer.write(img)

        writer.release()

    def filmRollouts(self,rollouts):
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')

        writer = cv2.VideoWriter(self.addr+'rollouts'+'.mov', fourcc, 10.0, (420,420))
        print self.rollouts
        for rollout in self.rollouts:
            for i in range(0,100):
                # img = cv2.imread(self.addr+rollout+'/'+'Aimee_'+rollout+'_frame_'+str(i)+'.jpg',1)
                addr = self.addr+"Johan_rollouts/Alter_Sup60_test/"+rollout + "/Johan_Alter_Sup60_test_" + rollout + "_frame_" + str(i) + ".jpg"
                print addr
                img = cv2.imread(addr)
                writer.write(img)

        writer.release()

    def filmSupervised(self,rollouts):
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')

        writer = cv2.VideoWriter(self.addr+'supervised'+'.mov', fourcc, 10.0, (420,420))
        
        for rollout in self.rollouts:
            for i in range(0,100):
                img = cv2.imread(self.addr+rollout+'/'+'Aimee_' + rollout+'_frame_'+str(i)+'.jpg',1)
                writer.write(img)

        writer.release()

    def run(self):
    	for rollout in self.rollouts:
            print rollout
            self.filmRollout(rollout)

    def run_many(self):
        self.filmSupervised(self.rollouts)

    def run_rols(self):
        # self.filmAnis()
        self.filmRollouts(self.rollouts)

    def filmAnis(self):
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')

        writer = cv2.VideoWriter(self.addr+'anim'+'.mov', fourcc, 20.0, (1200,1200))
        
        for rollout in self.rollouts:
            for i in range(0,150):
                img = cv2.imread(self.addr+str(i)+'.jpg',1)
                writer.write(img)

        writer.release()
            




if __name__ == '__main__':
    print "running"
    rng = [i for i in range(0,30)]
    ct = VideoMaker(rng,addr=AMTOptions.rollouts_dir,supervised=False)
    ct.run_rols()


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
