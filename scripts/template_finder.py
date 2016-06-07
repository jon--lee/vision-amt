import cv2
import numpy as np
import IPython


# mouse callback function

class TemplateGUI(object):

    def __init__(self,img=None):
        self.img = img.copy()
        self.true_img = img
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_circle)
        self.done = False
        self.template = None
        self.drawing = False # true if mouse is pressed
        self.mode = True # if True, draw rectangle. Press 'm' to toggle to curve
        self.ix,self.iy = -1,-1


        

    def getTemplate(self):
        while(1):
            cv2.imshow('image',self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif self.done: 
                break 
        cv2.destroyAllWindows()
        return self.template


      


    def draw_circle(self,event,x,y,flags,param):
        ix = self.ix
        iy = self.iy

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix,self.iy = x,y
            print "HERE"

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.rectangle(self.img,(ix,iy),(x,y),color=(0,255,0),thickness=-1)
      

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(self.img,(ix,iy),(x,y),color=(0,255,0),thickness=-1)

            self.template = self.true_img[iy:y,ix:x:].copy()
            self.done = True


if __name__ == '__main__':
    tg = TemplateGUI()
    template = tg.getTemplate()
    IPython.embed()

