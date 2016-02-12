from time import time
import IPython
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as LA
from template_finder import TemplateGUI


class CircleTracker(object):
    def __init__(self,rng =[],debug = False):
        self.Workers = dict()
        self.addr = ""
        self.rollouts = self.compileList(rng)
        self.gripper = cv2.imread('tmplates/gripper.jpg',1)
        self.gc = cv2.imread('tmplates/gc.jpg',1)
        self.debug = debug
        self.first = True
        self.prev_pos = np.zeros(2)
        self.file_lbl = open(self.addr+'labels.txt','w')


    def compileList(self,rng):
        rollouts = []
        for i in range(rng[0],rng[1]):
            rollouts.append("rollout"+str(i))
        return rollouts


    def getRollout(self,rollout):
        file_str = open(self.addr+rollout+'/states.txt','r')
        self.frames = []
        self.img_name = []
        self.states = []
        for i in range(0,100):
            self.frames.append(cv2.imread(self.addr+rollout+'/'+rollout+'_frame_'+str(i)+'.jpg',1))
            self.img_name.append(self.addr+rollout+'/'+rollout+'_frame_'+str(i)+'.jpg')

            line = file_str.readline()
            line = line.split()
            state = line[1:5]
            print state
            self.states.append(state)

    def lowPass(self, cur_pos):
        dist = LA.norm(cur_pos-self.prev_pos)
        print self.prev_pos
        print cur_pos
        print dist
        if(self.first): 
            self.first = False
            self.prev_pos = cur_pos
            return cur_pos
        if(dist > 50):
            return self.prev_pos
        else: 
            self.prev_pos = cur_pos
            return cur_pos


    def getTemplate(self,img):
        tg = TemplateGUI(img=img)
        self.gc = tg.getTemplate()
        

    def getPose(self,template,img,gripper = False): 
       


        w = template.shape[0]
        h = template.shape[1]
        img2 = img.copy()
        if(gripper):
            methods = ['cv2.TM_SQDIFF']
        else: 
            methods = ['cv2.TM_CCORR_NORMED']

        for meth in methods:
            img = img2.copy()
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            if(self.debug):
                bottom_right = (top_left[0] + w, top_left[1] + h)

                cv2.rectangle(img,top_left, bottom_right, 255, 2)

                plt.subplot(121),plt.imshow(res,cmap = 'gray')
                plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(img)
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                plt.suptitle(meth)
              
                plt.show()

        pos = np.zeros(2)
        pos[0] = top_left[0]+w/2
        pos[1] = top_left[1]+h/2
        if(not gripper):
            pos = self.lowPass(pos)
        return pos

    def pixelstoMeters(self,val):
        return 0.5461/420*val
    def computeLabel(self,img):

        grip_pos = self.getPose(self.gripper,img,gripper=True)
        gc_pos = self.getPose(self.gc,img)
        label = np.zeros(4)

        #Get extension and gripper 

        d_vec = gc_pos - grip_pos

        d_vec = self.pixelstoMeters(d_vec)

        label[0] = d_vec[1]
        label[1] = d_vec[0]
        
        return label


    def writeLabel(self,label,img_n):
        line = img_n+" "+str(label[0])+" "+str(label[1])+" "+str(label[2])+" "+str(label[3])+"\n"
        self.file_lbl.write(line)

    def run(self):

        for rollout in self.rollouts:
            self.getRollout(rollout)
            idx = 0
            self.first = True

            self.gc = self.getTemplate(self.frames[0])
            for img in self.frames:
                # if(idx == 55):
                #     IPython.embed()
                #     self.debug = True
                label = self.computeLabel(img)
                self.writeLabel(label,self.img_name[idx])
                idx += 1


if __name__ == '__main__':
    print "running"
    rng = [12,14]
    ct = CircleTracker(rng,debug= False)
    ct.run()
    ct.file_lbl.close()





















