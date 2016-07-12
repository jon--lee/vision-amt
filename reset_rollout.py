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
# import DexRobotTurntable
# from TurntableState import TurntableState
DexRobotZeke = DexRobotZeke.DexRobotZeke
# DexRobotTurntable = DexRobotTurntable.DexRobotTurntable
ZekeState = ZekeState.ZekeState



class reset():
    def __init__(self, izzy):
	
        self.izzy = izzy
        # self.turn = turn

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
        originalPHI = DexRobotZeke.PHI
        DexRobotZeke.PHI += 0.3
        
        # serial = self.turn._turntable._dex_serial
        # serial.ser = Serial(serial._comm, serial._baudrate)
        # serial.ser.setTimeout(serial._timeout)
        start = [3.5857, 0.0017, 0.170, 1.1239, 0.0002, 0.0]
        self.izzy.gotoState(ZekeState(start), tra_speed = .04)
        while not self.izzy.is_action_complete():
            print "moving ", self.izzy.getState()
            time.sleep(.3)
        # diffs = []
        # for i in range(10):
        #     start = [3.5857, 0.0017, 0.170, 1.1239, 0.0292, 0.0]
        #     end = [3.5257, 0.0017, 0.170, 1.1239, 0.0292, 0.0]
        #     self.izzy.gotoState(ZekeState(start), tra_speed = .04)
        #     while not self.izzy.is_action_complete():
        #         print "moving ", self.izzy.getState()
        #         time.sleep(.3)
        #     time.sleep(.3)
        #     reached = self.izzy.getState().state
        #     print "done start ", 
        #     diffs.append((start[0] - reached[0], start[2] - reached[2]))
        #     self.izzy.gotoState(ZekeState(end), tra_speed = .04)
        #     while not self.izzy.is_action_complete():
        #         print "moving ", self.izzy.getState()
        #         time.sleep(.3)
        #     time.sleep(.3)
        #     reached = self.izzy.getState().state
        #     diffs.append((end[0] - reached[0], end[2] - reached[2]))
        #     print "done end ", reached
        # self.izzy.gotoState(ZekeState([3.5857, 0.0017, 0.010, 1.1239, 0.0292, 0.0]), tra_speed = .04)
        # self.izzy.gotoState(ZekeState([3.5857, 0.0017, 0.170, 1.1239, 0.0292, 0.0]), tra_speed = .04)
        # self.izzy.gotoState(ZekeState([3.080, 0.0017, 0.170, 1.1239, 0.0292, 0.0]), tra_speed = .04)
        # print "moved"
        # self.izzy.gotoState(ZekeState([2.9827, 0.0166, 0.0717, 1.0763, 0.0002, 0.0]), tra_speed = .04)
        # self.izzy.gotoState(ZekeState([3.587, 0.0156, 0.1974, 1.1239, 0.0292, 0.0]), tra_speed =.04)
        # while not self.izzy.is_action_complete():
        #     print "moving again ", self.izzy.getState()
        #     time.sleep(.3)
        # [3.5857, 0.0013, 0.0992, 1.1239, 0.0292, 0.0]
        # keep between -255 and 255
       
        #for i in range(100):
       # while not self.turn.is_action_complete():
       #     pass
       # if(i % 2 == 0):
       #     self.turn.gotoState(TurntableState([5.2259]),  pi/180 * 150, .3)
       # else:
       #     self.turn.gotoState(TurntableState([5.2259 + 0.48132]), pi/180 * 150, .3)\
        # for i in range(2):
        #     self.izzy.gotoState(ZekeState([None, .1, 0.01, None, None, None]), tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([3.46, None, None, None, None, None]), tra_speed = .04)
            
        #     self.izzy.gotoState(ZekeState([None, None, .3, None, None, None]), tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([4.08, None, .3, None, None, None]), tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([3.65, None, None, None, None, None]),rot_speed = pi/8, tra_speed = .01)
        #     self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([2.9, None, None, None, None, None]),tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([3.25, None, None, None, None, None]), rot_speed = pi/8, tra_speed = .01)
        #     self.izzy.gotoState(ZekeState([None, .1, 0.01, None, None, None]), tra_speed = .04)
        #     self.izzy.gotoState(ZekeState([3.46, 0.02, None, None, .0681, None]), tra_speed = .04)
            
           

        #     while not self.izzy.is_action_complete():
        #         pass # busy waiting

        #     self.turn.gotoState(TurntableState([2.0*(.5-val) * pi]),  .1, .1)
        #     DexRobotZeke.PHI = originalPHI

        #     while not self.turn.is_action_complete():
        #         pass
            
        #     mag = 200
        #     sleep = .15

        #     for i in range(40):
        #         if i % 2 == 0:
        #             serial._control([mag * -1])
        #         else:
        #             serial._control([mag])
        #         time.sleep(sleep)
        #     serial._control([0])

        #     print "sleeping"
        #     time.sleep(5)
        #     print "done sleeping"
      
        #     pass # busy waiting
        print "completed"
        # print diffs
        # while True:
        #     print "moving again ", self.izzy.getState()
        #     time.sleep(.3)



        return
        
        

if __name__ == '__main__':
    print "hello world"
    izzy = DexRobotZeke()
    print "started izzy"
    r = reset(izzy)
    r.move_reset()
