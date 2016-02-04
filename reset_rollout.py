import time
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
        self.izzy.reset(tra_speed = .25)
        originalPHI = DexRobotZeke.PHI
        self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .25)
        self.izzy.gotoState(ZekeState([None, None, .3, None, None, None]), tra_speed = .25)
        self.izzy.gotoState(ZekeState([1.25 * pi + DexRobotZeke.PHI, None, .3, None, None, None]))
        self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .25)
        self.izzy.gotoState(ZekeState([1.05 * pi + DexRobotZeke.PHI, None, None, None, None, None]))
        self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .25)
        self.izzy.gotoState(ZekeState([.75 * pi + DexRobotZeke.PHI, None, None, None, None, None]))
        self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .25)
        self.izzy.gotoState(ZekeState([.95 * pi + DexRobotZeke.PHI, None, None, None, None, None]))
        self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .25)
        self.izzy.reset(tra_speed = .25)
        while not self.izzy.is_action_complete():
            pass # busy waiting
        self.turn.gotoState(TurntableState([(.5-val) * pi]),  .1, .1)
        DexRobotZeke.PHI = originalPHI
        return
        
        

if __name__ == '__main__':
    print "hello world"
    r = reset(DexRobotZeke(), DexRobotTurntable())
    r.move_reset()
