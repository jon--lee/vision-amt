from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *

#from pipeline.bincam import BinaryCamera

PREVIEW = False

def teleop(c, izzy, t, bc=None):
    try:
        while True:
            #state = izzy.getState()
            controls = c.getUpdates()
            #update_gripper(controls)
            print "Teleop: " + str(controls)
            time.sleep(.02)
    except KeyboardInterrupt:
        pass


def update_gripper(controls):
    controls[1] = 0
    controls[3] = 0
    self.izzy.control(controls)
    self.turntable.control([controls[5]])

if __name__ == '__main__':


    #bincam = BinaryCamera('./meta.txt')
    #bincam.open()

    t = TurnTableControl() # the com number may need to be changed. Default of com7 is used
    izzy = PyControl(115200, .04, [0,0,0,0,0],[0,0,0]); # same with this    
    c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])

    teleop(c, izzy, t)
