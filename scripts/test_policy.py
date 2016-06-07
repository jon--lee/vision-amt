import sys
import tty, termios
from options import AMTOptions
from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *
from pipeline.bincam import BinaryCamera
from Net.tensor import inputdata, net3,net4,net5,net6

import time
import datetime
import os
import random
import cv2
import imp
import IPython
import reset_rollout
import numpy as np
import compile_sets
import tensorflow as tf

sys.path[0] = sys.path[0] + '/../../GPIS/src/grasp_selection/control/DexControls'
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from DexRobotTurntable import DexRobotTurntable
from TurntableState import TurntableState
from serial import Serial


def getch():
    """
        Pause the program until key press
        Return key press character
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch



class PolicyTest():

    def __init__(self, bincam, izzy, turntable, options=AMTOptions()):
        self.bc = bincam
        self.izzy = izzy
        self.turntable = turntable
        self.options = options
        self.flip = True


    def clear_camera_buffer(self):
        """
            Clear excess frames out of the camera buffer
            to avoid off by k in frames and controls
        """
        for i in range(4):
            self.bc.vc.grab()

    def process_frame(self, frame):
        color_frame = cv2.resize(frame.copy(), (250, 250))
        return self.read_write_image(color_frame)


    def read_write_image(self, frame):
        cv2.imwrite('get_jp.jpg', frame)
        return cv2.imread('get_jp.jpg')

    def state(self, state):
        """
            Necessary wrapper to converting between PyControl and ZekeStates
        """
        if isinstance(state, ZekeState) or isinstance(state, TurntableState):
            return state.state
        return state

    def long2short_state(self, izzy_state, turntable_state):
        """
            Convert to shortened four element state
        """
        return [izzy_state[0], izzy_state[2], izzy_state[4], turntable_state[0]]


    def rescale(self, deltas):
        deltas[0] = deltas[0]*0.2
        deltas[1] = deltas[1]*0.01
        deltas[2] = deltas[2]*0.005
        deltas[3] = deltas[3]*0.2
        return deltas

    def deltaSafetyLimits(self, deltas):
        #Rotation 15 degrees
        #Extension 1 cm 
        #Gripper 0.5 cm
        #Table 15 degrees

        deltas[0] = np.sign(deltas[0])*np.min([0.2,np.abs(deltas[0])])
        deltas[1] = np.sign(deltas[1])*np.min([0.01,np.abs(deltas[1])])
        deltas[2] = 0.0#np.sign(deltas[2])*np.min([0.005,np.abs(deltas[2])])
        deltas[3] = np.sign(deltas[3])*np.min([0.2,np.abs(deltas[3])])
        return deltas


    @staticmethod
    def lst2str(lst):
        """
            returns a space separated string of all elements. A space
            also precedes the first element.
        """
        s = ""
        for el in lst:
            s += " " + str(el)
        return s



    def apply_deltas(self, delta_state):
        """
            Get current states and apply given deltas
            Handle max and min states as well
        """
        izzy_state = self.state(self.izzy.getState())
        t_state = self.state(self.turntable.getState())
        izzy_state[0] += delta_state[0]
        izzy_state[1] = 0.00952
        izzy_state[2] += delta_state[1]
        izzy_state[3] = 4.211
        izzy_state[4] =0.054# 0.0544 #delta_state[2]
        t_state[0] += delta_state[3]
        izzy_state[0] = min(self.options.ROTATE_UPPER_BOUND, izzy_state[0])
        izzy_state[0] = max(self.options.ROTATE_LOWER_BOUND, izzy_state[0])
        izzy_state[4] = min(self.options.GRIP_UPPER_BOUND, izzy_state[4])
        izzy_state[4] = max(self.options.GRIP_LOWER_BOUND, izzy_state[4])
        t_state[0] = min(self.options.TABLE_UPPER_BOUND, t_state[0])
        t_state[0] = max(self.options.TABLE_LOWER_BOUND, t_state[0])

        return izzy_state, t_state


    def rollout(self, num_frames = 80):
        net = self.options.tf_net
        path = self.options.tf_net_path
        sess = net.load(var_path=path)
        recording = []

        self.clear_camera_buffer()
        
        try:
            for i in range(num_frames):
                self.clear_camera_buffer()
                frame = self.bc.read_frame()
                
                pro_frame = self.process_frame(frame)
                pro_frame = np.reshape(pro_frame, (250, 250, 3))

                cv2.imshow('preview', pro_frame)
                cv2.waitKey(20)

                izzy_state = self.state(self.izzy.getState())
                turntable_state = self.state(self.turntable.getState())
                current_state = self.long2short_state(izzy_state, turntable_state)

                raw_deltas = net.output(sess, pro_frame, channels=3)
                scaled_deltas = self.rescale(raw_deltas)
                delta_state = self.deltaSafetyLimits(scaled_deltas)
                delta_state[2] = 0 # zeroing gripper deltas for now
                
                recording.append((frame, current_state, delta_state))
                new_izzy, new_t = self.apply_deltas(delta_state)

                print "DELTA STATE: ", delta_state
                self.izzy._zeke._queueState(ZekeState(new_izzy))
                self.turntable.gotoState(TurntableState(new_t), .25, .25)

                time.sleep(.005)

        except KeyboardInterrupt:
            pass

        sess.close()

        # prompt save? or save automatically?
        # TODO: handle having multiple graphs possible instead of just always using the default graph        
        
        return recording

    def hardware_reset(self):
        """
            This code comes straight from reset_rollout.py
            Copied here to avoid thread conflicts
        """


        while not (self.izzy.is_action_complete() and self.turntable.is_action_complete()):
            pass

        print "sleep"
        time.sleep(1)
        print "done sleeping"
        pi = np.pi
        val = random.random()
        time.sleep(1*val)
        DexRobotZeke.PHI += 0.3
        ZekeState([])
        
        originalPHI = DexRobotZeke.PHI


        
        self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .04)
        time.sleep(.5)
        self.izzy.gotoState(ZekeState([None, None, 0.03, None, None, None]), tra_speed = .04)
        print "izzy state" + str(self.izzy.getState())
        time.sleep(.5)

        self.izzy.gotoState(ZekeState([3.46, None, None, None, None, None]), rot_speed = pi/20, tra_speed = .04)
        print "izzy state" + str(self.izzy.getState())
        time.sleep(.5)

       
        DexRobotZeke.PHI = originalPHI
        

        self.izzy.gotoState(ZekeState([None, None, .3, None, None, None]), tra_speed = .04)
        print "izzy state" + str(self.izzy.getState())
        time.sleep(.5)

        self.izzy.gotoState(ZekeState([4.08, None, .3, None, None, None]), rot_speed = pi/20, tra_speed = .04)
        print "izzy state" + str(self.izzy.getState())
        time.sleep(.5)
        self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .04)
        print "izzy state" + str(self.izzy.getState())
        self.izzy.gotoState(ZekeState([3.65, None, None, None, None, None]),rot_speed = pi/20, tra_speed = .04)
        time.sleep(.5)
        self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .04)
        time.sleep(.5)
        self.izzy.gotoState(ZekeState([2.9, None, None, None, None, None]),rot_speed = pi/20,tra_speed = .04)
        time.sleep(.5)

        self.izzy.gotoState(ZekeState([None, .026, None, None, None, None]), tra_speed = .04)
        time.sleep(.5)

        self.izzy.gotoState(ZekeState([3.25, None, None, None, None, None]), rot_speed = pi/20, tra_speed = .04)
        time.sleep(.5)

        print "izzy state" + str(self.izzy.getState())

        self.izzy.gotoState(ZekeState([None, .1, None, None, None, None]), tra_speed = .04)
        time.sleep(.5)
        self.izzy.gotoState(ZekeState([None, None, 0.01, None, None, None]), tra_speed = .04)

        time.sleep(.5)

        self.izzy.gotoState(ZekeState([3.46, 0.02, None, None, .0681, None]), rot_speed = pi/20, tra_speed = .04)


        print "izzy state" + str(self.izzy.getState())
        while not self.izzy.is_action_complete():
            pass
        print "izzy state" + str(self.izzy.getState())

        print "turntable state" + str(self.turntable.getState())
        if(self.flip):
            new_t_state = 1.35 + np.random.rand() * np.pi / 5
            self.turntable.gotoState(TurntableState([new_t_state]),  .1, .1)
            self.flip = False
        else:
            new_t_state = 2.9 + np.random.rand() * np.pi / 5
            self.turntable.gotoState(TurntableState([new_t_state]),  .1, .1)
            self.flip = True

        
        while not self.turntable.is_action_complete():
            #print "turntable state" + str(self.turntable.getState())
            pass
        
        
        print "sleeping"
        time.sleep(1)
        print "done sleeping"
        self.izzy._zeke.clear_state()
        
        turn = self.turntable._turntable
        shake_mag = 100
        sleep = .40
        for i in range(10):
            if i % 2 == 0:
                turn._queueControl([shake_mag * -1])
            else:
                turn._queueControl([shake_mag])
            time.sleep(sleep)
        turn._queueControl([0.0])

        self.turntable._turntable.clear_state()

        print "sleeping"
        time.sleep(1)
        print "done sleeping"

        # serial = self.turntable._turntable._dex_serial
        
        # serial.ser = Serial(serial._comm, serial._baudrate)
        # serial.ser.setTimeout(serial._timeout)

        # shake_magnitude = 100
        # sleep = .15
        # for i in range(20):
        #     if i % 2 == 0:
        #         serial._control([shake_magnitude * -1])
        #     else:
        #         serial._control([shake_magnitude])
        #     time.sleep(sleep)
        # serial._control([0])
        # print "sleeping"
        # time.sleep(2)
        # print 'done sleeping'


    def policy(self, num_samples=10):
        test_name = Saver.next_test_name()
        #self.hardware_reset()
        num_successes = 0
        for i in range(num_samples):
            print "Policy rollout test #" + str(i)
            recording = self.rollout()
            Saver.save(recording, test_name,  i)
            print "rollout should be finished"
            # if successs, increment num
            # hardware reset
            self.hardware_reset()
        print "Total successful samples: "  + str(num_successes)



class Saver():

    @staticmethod
    def next_test_name():
        i = 0
        prefix = AMTOptions.policies_dir + 'test'
        path = prefix + str(i) + '/'
        while os.path.exists(path):
            i += 1
            path = prefix + str(i) + '/'
        return 'test' + str(i)

    @staticmethod
    def make_test_dir(test_name):
        path = AMTOptions.policies_dir + test_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def make_rollout_dir(test_path, rollout_name):
        rollout_path = test_path + rollout_name + '/'
        if not os.path.exists(rollout_path):
            os.makedirs(rollout_path)
        return rollout_path

    @staticmethod
    def save(recording, test_name, rollout_index):
        test_path = Saver.make_test_dir(test_name)
        rollout_name = test_name + '_rollout' + str(rollout_index)
        rollout_path = Saver.make_rollout_dir(test_path, rollout_name) 
        print "Saving rollout to " + str(rollout_path) + '...'
        rollout_states_file = open(rollout_path + 'states.txt', 'a+')
        #TODO: should we still save these rollouts as if we will use them for training?
        # this may require some tweaking of the overall file system
        i = 0
        for frame, state, deltas in recording:
            filename = rollout_name + '_frame_' + str(i) + '.jpg'
            rollout_states_file.write(filename + PolicyTest.lst2str(state) + "\n")
            cv2.imwrite(rollout_path + filename, frame)
            i += 1
        rollout_states_file.close()
        recording = []
        print "Done saving"

if __name__ == '__main__':
    bincam = BinaryCamera('./meta.txt')
    bincam.open()
    
    options = AMTOptions()
    
    izzy = DexRobotZeke()
    izzy._zeke.steady(False)
    t = DexRobotTurntable()
    
    g = tf.Graph()

    with g.as_default():
        options.tf_net = net6.NetSix()
        options.tf_net_path = '/media/1tb/Izzy/nets/net6_03-27-2016_12h04m13s.ckpt'

        pt = PolicyTest(bincam, izzy, t, options=options)
        while True:
            print "Waiting for an action ('q' -> quit, 't' -> execute test): "
            char = getch()
            if char == 'q':
                print "Quitting..."
                break
            elif char == 't':
                print "Running policy test... \n"
                pt.policy()
                print "Done running policy."


    print "Done."

