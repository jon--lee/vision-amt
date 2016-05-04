import time
from serial import Serial
import random
import numpy as np
import IPython
pi = np.pi
# import imp
# import fileperm
# all_perms=fileperm.get_perms(r'\\Simon\share\db\a.txt')
import sys
if '/../../GPIS/src/grasp_selection/control/DexControls' not in sys.path[0]:
    sys.path[0] = sys.path[0] + '/../../GPIS/src/grasp_selection/control/DexControls'
import DexRobotZeke
import ZekeState
import DexRobotTurntable
from TurntableState import TurntableState
DexRobotZeke = DexRobotZeke.DexRobotZeke
DexRobotTurntable = DexRobotTurntable.DexRobotTurntable
ZekeState = ZekeState.ZekeState



class reset():
    def __init__(self, izzy, turn):
	
        self.izzy = izzy
        self.turn = turn

#    def __init__(self, izzy, turn):

#        if '/../GPIS/src/grasp_selection/control/DexControls' not in sys.path[0]:
#            sys.path[0] = sys.path[0] + '/../GPIS/src/grasp_selection/control/DexControls'
#        import DexRobotZeke
#        import ZekeState
#        import DexRobotTurntable
#        from TurntableState import TurntableState
#        DexRobotTurntable = DexRobotTurntable.DexRobotTurntable
#        self.izzy = izzy
#        self.turn = turn

    # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
    def move_reset(self):
        val = random.random()
        time.sleep(1*val)
        DexRobotZeke.PHI += 0.3
        ZekeState([])
        
        originalPHI = DexRobotZeke.PHI
        serial = self.turn._turntable._dex_serial
        serial.ser = Serial(serial._comm, serial._baudrate)
        serial.ser.setTimeout(serial._timeout)


        # keep between -255 and 255
       
        #for i in range(100):
       # while not self.turn.is_action_complete():
       #     pass
       # if(i % 2 == 0):
       #     self.turn.gotoState(TurntableState([5.2259]),  pi/180 * 150, .3)
       # else:
       #     self.turn.gotoState(TurntableState([5.2259 + 0.48132]), pi/180 * 150, .3)
        for i in range(2):
            self.izzy.gotoState(ZekeState([None, .1, 0.01, None, None, None]), tra_speed = .04)
            self.izzy.gotoState(ZekeState([3.46, None, None, None, None, None]), tra_speed = .04)
            
            self.izzy.gotoState(ZekeState([None, None, .3, None, None, None]), tra_speed = .04)
            self.izzy.gotoState(ZekeState([4.08, None, .3, None, None, None]), tra_speed = .04)
            self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .04)
            self.izzy.gotoState(ZekeState([3.65, None, None, None, None, None]),rot_speed = pi/8, tra_speed = .01)
            self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .04)
            self.izzy.gotoState(ZekeState([2.9, None, None, None, None, None]),tra_speed = .04)
            self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .04)
            self.izzy.gotoState(ZekeState([3.25, None, None, None, None, None]), rot_speed = pi/8, tra_speed = .01)
            self.izzy.gotoState(ZekeState([None, .1, 0.01, None, None, None]), tra_speed = .04)
            self.izzy.gotoState(ZekeState([3.46, 0.02, None, None, .0681, None]), tra_speed = .04)
            
           

            while not self.izzy.is_action_complete():
                pass # busy waiting

            self.turn.gotoState(TurntableState([2.0*(.5-val) * pi]),  .1, .1)
            DexRobotZeke.PHI = originalPHI

            while not self.turn.is_action_complete():
                pass
            
            mag = 200
            sleep = .15

            for i in range(40):
                if i % 2 == 0:
                    serial._control([mag * -1])
                else:
                    serial._control([mag])
                time.sleep(sleep)
            serial._control([0])

            print "sleeping"
            time.sleep(5)
            print "done sleeping"
      
        


        return
        
        

if __name__ == '__main__':
    print "hello world"
    r = reset(DexRobotZeke(), DexRobotTurntable())
    r.move_reset()
