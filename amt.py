# TODO: rollout directories read and write
# TODO: caffe train and rollout

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

from query_cam import query_cam

sys.path[0] = sys.path[0] + '/../../GPIS/src/grasp_selection/control/DexControls'
        
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from DexRobotTurntable import DexRobotTurntable
from TurntableState import TurntableState


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


class AMT():

    def __init__(self, bincam, izzy, turntable, controller, options=AMTOptions()):
        self.bc = bincam
        self.izzy = izzy
        self.turntable = turntable
        self.c = controller
        self.options = options
        self.r = reset_rollout.reset(izzy, turntable)

        # self.qc = query_cam(self.bc)




    def initial_demonstration(self, controller):
        print "Starting supervisor demonstration..."
        recording = []
        try:
            while True:
                controls = controller.getUpdates()
                deltas = self.controls2deltas(controls)
                print deltas
                if not all(d == 0.0 for d in deltas):
                    frame = self.bc.read_frame()
                    new_izzy, new_t = self.apply_deltas(deltas)
                    recording.append((frame, deltas))
                                               
                    self.izzy._zeke._queueState(ZekeState(new_izzy))
                    self.turntable.gotoState(TurntableState(new_t), .25, .25)
                        
                    time.sleep(0.08)

        except KeyboardInterrupt:
            pass

        self.save_initial(recording)
        print "Supervisor demonstration done."

    # [rot, elev, ext, wrist, grip, turntable]
    @staticmethod
    def controls2deltas(controls):
        deltas = [0.0] * 4
        deltas[0] = controls[0] / 300.0
        deltas[1] = controls[2] / 1000.0
        deltas[2] = controls[4] / 8000.0
        deltas[3] = controls[5] / 800.0
        if abs(deltas[0]) < 2e-3:
            deltas[0] = 0.0
        if abs(deltas[1]) < 2e-2:
       	    deltas[1] = 0.0
       	if abs(deltas[2]) < 5e-3:
       	    deltas[2] = 0.0
        if abs(deltas[3]) < 2e-2:
            deltas[3] = 0.0 
        return deltas

    def rescale(self,deltas):

        deltas[0] = deltas[0]*0.2
        deltas[1] = deltas[1]*0.01
        deltas[2] = deltas[2]*0.005
        deltas[3] = deltas[3]*0.2
        return deltas



    def deltaSafetyLimites(self,deltas):
        #Rotation 15 degrees
        #Extension 1 cm 
        #Gripper 0.5 cm
        #Table 15 degrees

        deltas[0] = np.sign(deltas[0])*np.min([0.2,np.abs(deltas[0])])
        deltas[1] = np.sign(deltas[1])*np.min([0.01,np.abs(deltas[1])])
        deltas[2] = 0.0#np.sign(deltas[2])*np.min([0.005,np.abs(deltas[2])])
        deltas[3] = np.sign(deltas[3])*np.min([0.2,np.abs(deltas[3])])
        return deltas

    def rollout_tf(self, num_frames=100):
        net = self.options.tf_net
        path = self.options.tf_net_path
        sess = net.load(var_path=self.options.tf_net_path)
        recording = []
        # self.qc = query_cam(self.bc)
        # #Clear Buffer ... NEED TO TEST
        # # self.qc.start()
        # while(self.qc.read_frame() is None):
        #     print self.qc.frame
        #     pass # wait until images start coming through
        #
        for i in range(4):
            self.bc.vc.grab()
        try:

            for i in range(num_frames):
                # Read from the most updated frame
                for i in range(4):
                    self.bc.vc.grab()
                frame = self.bc.read_frame()
                #frame = self.qc.read_frame()
                # done reading
                if(False):
                    gray_frame = self.gray(frame) 
                elif(False):
                    gray_frame = self.segment(frame)
                elif(True):
                    gray_frame = self.color(frame)

                gray_frame = np.reshape(gray_frame, (250, 250, 3))
            

                cv2.imshow("camera",gray_frame)
                cv2.waitKey(30)

 
                current_state = self.long2short_state(self.state(self.izzy.getState()), self.state(self.turntable.getState()))
                


                delta_state = self.rescale(net.output(sess, gray_frame,channels=3))
                #delta_state = net.output(sess, gray_frame,channels=3)
                delta_state = self.deltaSafetyLimites(delta_state)
                delta_state[2] = 0.0
                recording.append((frame, current_state,delta_state))
                new_izzy, new_t = self.apply_deltas(delta_state)

                # TODO: uncomment these to update izzy and t
                print "DELTA STATE ",delta_state
                self.izzy._zeke._queueState(ZekeState(new_izzy))
                self.turntable.gotoState(TurntableState(new_t), .25, .25)

                
                time.sleep(.005)
       
        except KeyboardInterrupt:
            pass

        # stop querying the camera
        # self.qc.terminate()
        # terminated

       
        sess.close()
        # self.izzy._zeke.steady(True)
        self.prompt_save(recording)
        # self.r.move_reset()
        # self.izzy._zeke.steady(False)
    
    def test(self):
        try:
            while True:
                izzy_state = self.state(self.izzy.getState())
                turntable_state = self.state(self.turntable.getState())
                print self.long2short_state(izzy_state, turntable_state)
                time.sleep(.03)    
        except KeyboardInterrupt:
            pass
            
    @staticmethod
    def state(state):
        """
            Necessary wrapper for quickly converting between PyControl and ZekeCode
        """
        if isinstance(state, ZekeState) or isinstance(state, TurntableState):
            return state.state
        return state

    def prompt_save(self, recording):
        num_rollouts = len(AMT.rollout_dirs())
        print "There are " + str(num_rollouts) + " rollouts. Save this one? (y/n): "
        char = getch()
        if char == 'y':
            return self.save_recording(recording)
        elif char == 'n':
            recording = []   # erase recordings and states
            return None
        self.prompt_save()

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


    @staticmethod
    def short2long_state(short_state):
        """
            Convert 4-element state to izzy and turntable states
            Returns a tuple (first element is izzy state, second is turntable)
        """
        izzy_state = [short_state[0], 0, short_state[1], 0, short_state[2], 0]
        t_state = [short_state[-1]]
        return izzy_state, t_state

    @staticmethod
    def long2short_state(izzy_state, t_state):
        """
            Convert given izzy state and t state to four element state
        """
        return [izzy_state[0], izzy_state[2], izzy_state[4], t_state[0]]

    
    def update_weights(self, iterations=10):
        net = self.options.tf_net
        path = self.options.tf_net_path
        data = inputdata.AMTData(self.options.train_file, self.options.test_file)
        self.options.tf_net_path = net.optimize(iterations, data, batch_size=50, path=path)

    def segment(self, frame):
        binary_frame = self.bc.pipe(np.copy(frame))
        return binary_frame

    def gray(self, frame):
        grayscale = self.bc.gray(np.copy(frame))
        return grayscale
    def color(self,frame): 
        color_frame = cv2.resize(frame.copy(), (250, 250))
        cv2.imwrite('get_jp.jpg',color_frame)
        color_frame= cv2.imread('get_jp.jpg')
        return color_frame


    def write_train_test_sets(self):
        deltas_file = open(self.options.deltas_file, 'r')
        train_writer = open(self.options.train_file, 'w+')
        test_writer = open(self.options.test_file, 'w+')
        for line in deltas_file:
            new_line = self.options.binaries_dir + line
            if random.random() > .2:
                train_writer.write(new_line)
            else:
                test_writer.write(new_line)
    
    def save_initial(self, tups):
        """
            Different from save recording in that this is intended
            for saving initial supervisor demonstrations
        """
        tups = self.roll(tups, 4)
        print "Saving initial demonstration"
        prefix = datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss")
        print "Saving raw frames to " + self.options.originals_dir + "..."
        print "Saving binary frames to " + self.options.binaries_dir + "..."
        deltas_file = open(self.options.deltas_file, 'a+')
        i = 0
        for frame, delta in tups:
            filename = prefix + "_frame_" + str(i) + ".jpg"
            deltas_file.write(filename + self.lst2str(delta) + "\n")
            cv2.imwrite(self.options.originals_dir + filename, frame)
            cv2.imwrite(self.options.binaries_dir + filename, self.segment(frame))
            i += 1
        deltas_file.close()
        print "Done saving."

    def save_recording(self, recording):
        """
            Save instance recordings and states by writing filename and corresponding state
            to states files and writing images to master frames dir and appropriate rollout dir.
            Clear recordings and states from memory when done writing
            :return:
        """
        rollout_name = self.next_rollout()
        rollout_path = self.options.rollouts_dir + rollout_name + '/'
        print "Saving rollout to " + rollout_path + "..."
        os.makedirs(rollout_path)
        rollout_states_file = open(rollout_path + "states.txt", 'a+')
        rollout_deltas_file = open(rollout_path + "net_deltas.txt", 'a+')
        print "Saving raw frames to " + self.options.originals_dir + "..."
        print "Saving binaries to " + self.options.binaries_dir + "..."
        print "Saving colors to " + self.options.colors_dir + "..."

        raw_states_file = open(self.options.originals_dir + "states.txt", 'a+')

        i = 0
        for frame, state,deltas in recording:
            filename = rollout_name + "_frame_" + str(i) + ".jpg"
            raw_states_file.write(filename + self.lst2str(state) + "\n") 
            rollout_states_file.write(filename + self.lst2str(state) + "\n")
            rollout_deltas_file.write(filename + self.lst2str(deltas) + "\n")
            cv2.imwrite(self.options.originals_dir + filename, frame)
            cv2.imwrite(self.options.grayscales_dir + filename, self.gray(frame))
            cv2.imwrite(self.options.binaries_dir + filename, self.segment(frame))
            cv2.imwrite(self.options.colors_dir + filename, self.color(frame))
            cv2.imwrite(rollout_path + filename, frame)
            i += 1
        raw_states_file.close()
        rollout_states_file.close()
        rollout_deltas_file.close()
        recording = []
        print "Done saving."

    @staticmethod
    def rollout_dirs():
        """
        :return: list of strings that are the names of rollout dirs
        """
        return list(os.walk(AMTOptions.rollouts_dir))[0][1]

    @staticmethod
    def next_rollout():
        """
        :return: the String name of the next new potential rollout
                (i.e. do not overwrite another rollout)
        """
        i = 0
        prefix = AMTOptions.rollouts_dir + 'rollout'
        path = prefix + str(i) + "/"
        while os.path.exists(path):
            i += 1
            path = prefix + str(i) + "/"
        return 'rollout' + str(i)

    @staticmethod
    def lst2str(lst):
        """
        returns a space separated string of all elements. A space
        also precedes the first element.
        :param lst:
        :return:
        """
        s = ""
        for el in lst:
            s += " " + str(el)
        return s
        
    @staticmethod
    def roll(tuples, change):
        frames, states =  zip(*tuples)
        frames = frames[change:]
        states = states[:-change]
        return zip(frames, states)

if __name__ == "__main__":
    

    bincam = BinaryCamera('./meta.txt')
    bincam.open()

    options = AMTOptions()

    #t = TurnTableControl() # the com number may need to be changed. Default of com7 is used
    #izzy = PyControl(115200, .04, [0,0,0,0,0],[0,0,0]) # same with this
    c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])
    izzy = DexRobotZeke()
    izzy._zeke.steady(False)
    t = DexRobotTurntable()

    #options.tf_net = net5.NetFive()
    #options.tf_net_path = '/home/annal/Izzy/vision_amt/Net/tensor/net5/net5_02-15-2016_11h58m56s.ckpt'
    options.tf_net = net6.NetSix()

    #options.tf_net_path = '/media/1tb/Izzy/nets/net6_02-26-2016_17h58m15s.ckpt'
    #options.tf_net_path = '/media/1tb/Izzy/nets/net6_02-27-2016_15h30m01s.ckpt'
    #options.tf_net_path = '/media/1tb/Izzy/nets/net6_03-12-2016_15h03m44s.ckpt'
    options.tf_net_path = '/media/1tb/Izzy/nets/net6_03-28-2016_14h51m12s.ckpt'
    amt = AMT(bincam, izzy, t, c, options=options)

    while True:
        print "Waiting for keypress ('q' -> quit, 'r' -> rollout, 'u' -> update weights, 't' -> test, 'd' -> demonstrate, 'c' -> compile train/test sets): "
        char = getch()
        if char == 'q':
            print "Quitting..."
            break
        
        elif char == 'r':
            print "Rolling out..."
            ro = amt.rollout_tf()
            print "Done rolling out."

        elif char == 'u':
            print "Updating weights..."
            amt.update_weights()
            print "Done updating."

        elif char == 'd':
            print "Initial demonstration..."
            amt.initial_demonstration(c)
            print "Done demonstrating."

        elif char == 'c':
            print 'Compiling train and test sets...'
            compile_sets.compile()
            print 'Done compiling sets'

        elif char == 't':
            amt.test()

    print "Done."
