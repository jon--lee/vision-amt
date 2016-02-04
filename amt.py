# TODO: rollout directories read and write
# TODO: caffe train and rollout

import sys
import tty, termios
from options import AMTOptions
from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *
from pipeline.bincam import BinaryCamera
#from Net.tensor import inputdata, net3
import time
import os
import random
import cv2
import imp
import IPython
import reset_rollout

sys.path[0] = sys.path[0] + '/../../GPIS/src/grasp_selection/control/DexControls'
        
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from DexRobotTurntable import DexRobotTurntable
from TurntableState import TurntableState


def getch():
    """
        getch will cause python program to wait until
        command is given rather than constantly checking
        from updates while looping.
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

        self.recording = []
        self.states = []
        
        self.r = reset_rollout.reset(izzy, turntable)

    def initial_demonstration(self, controller):
        recording = []
        try:
            while True:

                initial_state = self.long2short_state(self.izzy.getState().state, self.turntable.getState().state)
                
                controls = controller.getUpdates()
                deltas = self.controls2deltas(controls)
                new_izzy, new_t = self.apply_deltas(deltas)
                self.izzy.gotoState(ZekeState(new_izzy))
                self.turntable.gotoState(TurntableState(new_t))
                print new_izzy, new_t
                self.izzy.gotoState(ZekeState(new_izzy))

                final_state = self.long2short_state(self.izzy.getState().state, self.turntable.getState().state)
                delta_state = [ final - initial for final, initial in zip(final_state, initial_state) ]

                frame = self.bc.read_frame()

                recording.append((frame, delta_state))

                self.save_initial(recording)                

                time.sleep(.03)
        except KeyboardInterrupt:
            pass

    @staticmethod
    def controls2deltas(controls):
        controls = [ c if abs(c) > self.options.drift else 0.0 for c in controls ]
        deltas = [0.0] * 4
        deltas[0] = controls[0] / 500.0
        deltas[1] = controls[2] / 10000.0
        deltas[2] = controls[4] / 500.0
        deltas[3] = 0.0
        if abs(deltas[0]) < 2e-3:
            deltas[0] = 0.0
        if abs(deltas[2]) < 2e-3:
       	    deltas[2] = 0.0
       	if abs(deltas[4]) < 2e-2:
       	    deltas[4] = 0.0 
        return deltas

    def rollout_caffe(self, num_frames=50):
        raise NotImplementedError

    # TODO: uncomment stuff for real rollout
    def rollout_tf(self, num_frames=150):
        #net = self.options.tf_net
        #path = self.options.tf_net_path
        #sess = net.load(var_path=path)
        try:
            for i in range(num_frames):
                frame = self.bc.read_frame()
                #binary_frame = self.bc.pipe(frame)
                
                # for pycontrol
                self.recording.append((frame, self.long2short_state(self.izzy.getState(), self.turntable.getState())))

                # for zekestate
                #self.recording.append((frame, self.long2short_state(self.izzy.getState().state, self.turntable.getState().state)))
                #binary_frame = np.reshape(binary_frame, (125, 125, 1))

                #delta_state = net.output(sess, binary_frame)
                #new_izzy, new_t = self.apply_deltas(delta_state)

                # TODO: uncomment these to update izzy and t
                #print new_izzy, new_t
                #self.izzy.sendStateRequest(new_izzy)
                #self.turntable.sendStateRequest(new_t)

                time.sleep(.03)
                
        except KeyboardInterrupt:
            pass
        #sess.close()
        self.prompt_save()
        #self.r.move_reset()
    
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
        if isinstance(state, ZekeState) or isinstance(state, TurntableState):
            return state.state
        return state

    def prompt_save(self):
        num_rollouts = len(AMT.rollout_dirs())
        print "There are " + str(num_rollouts) + " rollouts. Save this one? (y/n): "
        char = getch()
        if char == 'y':
            return self.save_recording()
        elif char == 'n':
            self.recording = []   # erase recordings and states
            return None
        self.prompt_save()

    def apply_deltas(self, delta_state):
        izzy_state = self.state(self.izzy.getState())
        t_state = self.state(self.turntable.getState())
        izzy_state[0] += delta_state[0]
        izzy_state[1] = None
        izzy_state[2] += delta_state[1]
        izzy_state[3] = None
        izzy_state[4] += delta_state[2]
        t_state[0] += delta_state[3]
        return izzy_state, t_state


    @staticmethod
    def short2long_state(short_state):
        izzy_state = [short_state[0], 0, short_state[1], 0, short_state[2], 0]
        t_state = [short_state[-1]]
        return izzy_state, t_state

    @staticmethod
    def long2short_state(izzy_state, t_state):
        return [izzy_state[0], izzy_state[2], izzy_state[4], t_state[0]]

    def update_weights(self, iterations=10):

        net = self.options.tf_net
        path = self.options.tf_net_path
        data = inputdata.AMTData(self.options.train_file, self.options.test_file)
        self.options.tf_net_path = net.optimize(iterations, batch_size=300, path=path,  data=data)

    def segment(self, frame):
        binary_frame = self.bc.pipe(frame)
        return binary_frame


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
    
    def save_initial(self, tups)
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

    def save_recording(self):
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


        print "Saving raw frames to " + self.options.originals_dir + "..."
        print "Saving binaries to " + self.options.binaries_dir + "..."

        raw_states_file = open(self.options.originals_dir + "states.txt", 'a+')

        i = 0
        for frame, state in self.recording:
            filename = rollout_name + "_frame_" + str(i) + ".jpg"
            raw_states_file.write(filename + self.lst2str(state) + "\n") 
            rollout_states_file.write(filename + self.lst2str(state) + "\n")
            cv2.imwrite(self.options.originals_dir + filename, frame)
            cv2.imwrite(self.options.binaries_dir + filename, self.segment(frame))
            cv2.imwrite(rollout_path + filename, frame)
            i += 1
        raw_states_file.close()
        rollout_states_file.close()
        self.recording = []
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


if __name__ == "__main__":
    

    bincam = BinaryCamera('./meta.txt')
    bincam.open()

    options = AMTOptions()

    #t = TurnTableControl() # the com number may need to be changed. Default of com7 is used
    #izzy = PyControl(115200, .04, [0,0,0,0,0],[0,0,0]) # same with this
    #c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])
    c = None
    izzy = DexRobotZeke()
    t = DexRobotTurntable()


    #options.tf_net = net2.NetTwo()
    #options.tf_net_path = options.tf_dir + 'net2/net2_01-21-2016_02h14m08s.ckpt'
    #options.tf_net = net3.NetThree()
    #options.tf_net_path = options.tf_dir + 'net3/net3_01-19-2016_00h47m49s.ckpt'
    options.tf_net_path = None    

    # TODO: set actual train/test paths
    options.train_path = 'path/to/train.txt'
    options.test_path = 'path/to/test.txt'
    # TODO: set path to raw image directory and segmented image directory
    options.raw_path = 'path/to/raw_images/'
    options.seg_path = 'path/to/seg_images/'

    amt = AMT(bincam, izzy, t, c, options=options)

    while True:
        print "Waiting for keypress ('q' -> quit, 'r' -> rollout, 'u' -> update weights, 't' -> test): "
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

        elif char == 't':
            amt.test()

    print "Done."
