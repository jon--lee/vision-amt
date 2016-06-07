from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *
from options import AMTOptions
#from pipeline.bincam import BinaryCamera
import sys
sys.path[0] = sys.path[0] + '/../../../GPIS/src/grasp_selection/control/DexControls'
        
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from DexRobotTurntable import DexRobotTurntable
from TurntableState import TurntableState

ROTATE_UPPER_BOUND = 3.82
ROTATE_LOWER_BOUND = 3.06954

GRIP_UPPER_BOUND = .06
GRIP_LOWER_BOUND = .0023

TABLE_LOWER_BOUND = .002
TABLE_UPPER_BOUND = 7.0 


def teleop(c, izzy, t):
    target_state_i = get_state(izzy.getState())
    target_state_t = get_state(t.getState())
    try:
        while True:
            controls = c.getUpdates()
            deltas = controls2deltas(controls)
            if not all(d == 0.0 for d in deltas):
                
                print "Current: ", target_state_i, target_state_t
                new_izzy, new_t = apply_deltas(deltas, target_state_i, target_state_t)
                target_state_i, target_state_t = new_izzy, new_t 
                print "Teleop: ", new_izzy, new_t
                izzy._zeke._queueState(ZekeState(new_izzy))
                t.gotoState(TurntableState(new_t), .25, .25)

                time.sleep(.05)                
                
    except KeyboardInterrupt:
        pass


def get_state(state):
    if isinstance(state, ZekeState) or isinstance(state, TurntableState):
        return state.state
    return state

def controls2deltas(controls):
    deltas = [0.0] * 4
    deltas[0] = controls[0] / 5300#300.0
    deltas[1] = controls[2] / 30000#9000#1000.0
    deltas[2] = controls[4] / 8000.0
    deltas[3] = controls[5] / 800.0
    if abs(deltas[0]) < 8e-8:
        deltas[0] = 0.0
    if abs(deltas[1]) < 8e-4:#8e-4: #2e-2:
        deltas[1] = 0.0
    if abs(deltas[2]) < 5e-3:
        deltas[2] = 0.0
    if abs(deltas[3]) < 2e-2:
        deltas[3] = 0.0 
    return deltas
    

def apply_deltas(delta_state,t_i,t_t):
    """
        Get current states and apply given deltas
        Handle max and min states as well
    """
    
    t_i[0] += delta_state[0]
    t_i[1] = 0.00952
    t_i[2] += delta_state[1]
    t_i[3] = 4.211
    t_i[4] = 0.0004#0.054# 0.0544 #delta_state[2]
    t_t[0] += delta_state[3]
    t_i[0] = min(ROTATE_UPPER_BOUND, t_i[0])
    t_i[0] = max(ROTATE_LOWER_BOUND, t_i[0])
    t_i[4] = min(GRIP_UPPER_BOUND, t_i[4])
    t_i[4] = max(GRIP_LOWER_BOUND, t_i[4])
    t_t[0] = min(TABLE_UPPER_BOUND, t_t[0])
    t_t[0] = max(TABLE_LOWER_BOUND, t_t[0])

    return t_i, t_t


if __name__ == '__main__':
    options = AMTOptions()
    izzy = DexRobotZeke()
    izzy._zeke.steady(False)
    t = DexRobotTurntable()
    c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])

    teleop(c, izzy, t)
   
