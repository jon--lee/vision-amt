import sys
sys.path.append('/home/annal/Izzy/vision_amt/scripts')
sys.path.append('/home/annal/Izzy/vision_amt/scripts/objects/')
import tty, termios
from options import AMTOptions
from pipeline.bincam import BinaryCamera
from Net.tensor import inputdata, net3,net4,net5,net6, net6_c
from scripts.objects import singulationImg
from scripts import overlay, click_centers, transfer_weights
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



sys.path[0] = sys.path[0] + '/../../GPIS/src/grasp_selection/control/DexControls'
        
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState


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

    def __init__(self, bincam, izzy, options=AMTOptions()):
        self.bc = bincam
        self.izzy = izzy
        self.options = options
        self.succeed = 0.0
        self.total = 0
        self.graphs = []

    def rescale_sup(self, deltas):
        deltas[0] = deltas[0]*0.026666666666667
        deltas[1] = deltas[1]*0.006
        return deltas



    def deltaSafetyLimites(self,deltas):
        deltas[0] = np.sign(deltas[0])*np.min([0.2,np.abs(deltas[0])])
        deltas[1] = np.sign(deltas[1])*np.min([0.01,np.abs(deltas[1])])
        return deltas

    def rollout_tf(self, num_frames=100):
        net = self.options.tf_net
        path = self.options.tf_net_path[0]
        sess = net.load(var_path=self.options.tf_net_path)
        recording = []
        current_state = self.state(self.izzy.getState()) 
        
        for i in range(4):
            self.bc.vc.grab()
        try:
            for i in range(num_frames):
                # Read from the most updated frame
                start = time.time()
                for i in range(4):
                    self.bc.vc.grab()
                frame = self.bc.read_frame()
                # done reading

                gray_frame = self.color(frame)

                gray_frame = np.reshape(gray_frame, (250, 250, 3))
                h, w = frame.shape[0], frame.shape[1]
                disp_frame = np.zeros((h, w, frame.shape[2]))
                for i in range(frame.shape[2]):
                    #Binary Mask
                    disp_frame[:,:,i] = np.round(frame[:,:,i] / 255.0 - .25, 0)
                cv2.imshow("camera", disp_frame)
                cv2.waitKey(30)

                
                # delta_states = []
                # magnitudes = []
                true_state = np.array(self.state(self.izzy.getState()))
                # current_state = np.array(true_state)
                # outval = net.output(sess, gray_frame,channels=3, clasfc=True)
                outval = net.output(sess, gray_frame,channels=3)
                print "outval: ", outval
                delta_state = self.rescale_sup(outval)
                delta_state = self.deltaSafetyLimites(delta_state)
                # delta_states.append(delta_state)
                # magnitudes.append(np.linalg.norm(delta_state))
                # bst = np.argmax(magnitudes)
                # delta_state = delta_states[bst]
                

                recording.append((frame, current_state, delta_state, true_state))
                new_izzy = self.apply_deltas(current_state, delta_state)

                # TODO: uncomment these to update izzy and t
                print "DELTA STATE ",delta_state
                print "current_state: ", new_izzy
                self.izzy._zeke._queueState(ZekeState(new_izzy))
                # self.turntable.gotoState(TurntableState(new_t), .25, .25)
                current_state = new_izzy[:]

                offset = max(0, .3 - (time.time() - start))
                print "offset", offset
                time.sleep(offset)
                print "total time: ", time.time() - start

       
        except KeyboardInterrupt:
            pass
        self.return_to_start(current_state)
        cv2.destroyAllWindows()
        sess.close()
        self.prompt_save(recording)

    def safety(self, delta):
        delta[0] = np.sign(delta[0]) * min(abs(delta[0]), .02)
        delta[2] = np.sign(delta[2]) * min(abs(delta[2]), .007)
        return delta

    def return_to_start(self, current_state):
        print current_state, type(current_state)
        # destination = np.array([3.4701, 0.021, 0.024, 4.2359, 0.0004, 7.138])
        destination = np.array([3.5857, 0.0017, 0.0117, 1.1239, 0.0002, 0.0])
        while np.linalg.norm(current_state - destination) > .001:
            print np.linalg.norm(current_state - destination)
            print self.safety(destination - np.array(current_state))
            current_state = current_state + self.safety(destination - np.array(current_state))
            self.izzy._zeke._queueState(ZekeState(current_state))
            time.sleep(.1)
            print current_state
        time.sleep(.25)
                
    @staticmethod
    def state(state):
        """
            Necessary wrapper for quickly converting between PyControl and ZekeCode
        """
        if isinstance(state, ZekeState):
            return state.state
        return state

    def prompt_save(self, recording):
        num_rollouts = len(AMT.rollout_dirs())
        print "There are " + str(num_rollouts) + " rollouts. Save this one? (y/n): "
        char = getch()
        if char == 'y':
            self.total += 1
            print "Click the centers to determine success"
            distance = 100#click_centers.max_distance(click_centers.centers(self.bc))
            cv2.destroyAllWindows()
            print "Did the task succeed? (y/n), max distance was: " + str(distance) + " assumed success: " + str(distance>=100.0) 
            char = getch()
            if char == 'y':
                self.succeed += 1
            return self.save_recording(recording)
        elif char == 'n':
            recording = []   # erase recordings and states
            return None
        self.prompt_save(recording)

    def apply_deltas(self, izzy_state, delta_state):
        """
            Get current states and apply given deltas
            Handle max and min states as well
        """
        izzy_state[0] += delta_state[0]
        izzy_state[1] = 0.0017
        izzy_state[2] += delta_state[1]
        izzy_state[3] = 1.1239

        izzy_state[0] = min(self.options.ROTATE_UPPER_BOUND, izzy_state[0])
        izzy_state[0] = max(self.options.ROTATE_LOWER_BOUND, izzy_state[0])
        izzy_state[2] = min(self.options.EXTENSION_UPPER_BOUND, izzy_state[2])
        izzy_state[2] = max(self.options.EXTENSION_LOWER_BOUND, izzy_state[2])
        return izzy_state

    @staticmethod
    def long2short_state(izzy_state, t_state):
        """
            Convert given izzy state and t state to four element state
        """
        return [izzy_state[0], izzy_state[2]]

    
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

    def save_recording(self, recording):
        """
            Save instance recordings and states by writing filename and corresponding state
            to states files and writing images to master frames dir and appropriate rollout dir.
            Clear recordings and states from memory when done writing
            :return:
        """
        print "saving statistics to: " + self.options.rollouts_dir + "../statistics_singulation.txt"
        statistics_file = open(self.options.rollouts_dir + "../statistics_singulation.txt", 'w')
        statistics_file.write("total: " + str(self.total) + "\n")
        statistics_file.write("success rate: " + str(self.succeed/self.total) + "\n")
        statistics_file.write("success number: " + str(self.succeed))
        statistics_file.close()
        rollout_name = self.next_rollout()
        rollout_path = self.options.rollouts_dir + rollout_name + '/'

        print "Saving rollout to " + rollout_path + "..."
        os.makedirs(rollout_path)
        rollout_states_file = open(rollout_path + "states.txt", 'a+')
        rollout_deltas_file = open(rollout_path + "net_deltas.txt", 'a+')
        print "Saving template to " + rollout_path + "..."
        template = cv2.imread("/home/annal/Izzy/vision_amt/scripts/objects/template.png")
        np.save(rollout_path + "template.npy", template)
        print "Saving raw frames to " + self.options.originals_dir + "..."
        print "Saving binaries to " + self.options.binaries_dir + "..."
        print "Saving colors to " + self.options.colors_dir + "..."

        raw_states_file = open(self.options.originals_dir + "states.txt", 'a+')
        true_states_file = open(rollout_path + "true_states.txt", 'a+')

        i = 0
        for frame, state,deltas, true_state in recording:
            filename = rollout_name + "_frame_" + str(i) + ".jpg"
            raw_states_file.write(filename + self.lst2str(state) + "\n") 
            rollout_states_file.write(filename + self.lst2str(state) + "\n")
            rollout_deltas_file.write(filename + self.lst2str(deltas) + "\n")
            true_states_file.write(filename + self.lst2str(true_state) + "\n")
            cv2.imwrite(self.options.originals_dir + filename, frame)
            cv2.imwrite(self.options.grayscales_dir + filename, self.gray(frame))
            cv2.imwrite(self.options.binaries_dir + filename, self.segment(frame))
            cv2.imwrite(self.options.colors_dir + filename, self.color(frame))
            cv2.imwrite(rollout_path + filename, frame)
            i += 1
        raw_states_file.close()
        rollout_states_file.close()
        rollout_deltas_file.close()
        true_states_file.close()
        recording = []
        print "Done saving."

    def display_template(self, template=None):
        if template is None:
            template = cv2.imread("/home/annal/Izzy/vision_amt/scripts/objects/template.png")
        template[:,:,1] = template[:,:,2]
        template[:,:,0] = np.zeros((420, 420))
        # template[:,:,2] = np.zeros((420, 420))
        # template = cv2.resize(template, (250, 250))

        while 1:
            frame = self.bc.read_frame()
            frame = inputdata.im2tensor(frame, channels = 3)
            final = np.abs(-frame + template/255.0)
            cv2.imshow('camera', final)
            a = cv2.waitKey(30)
            if a == 27:
                cv2.destroyWindow('camera')
                break
            elif a == ord(' '):
                return 'next'
            time.sleep(.005)

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

    izzy = DexRobotZeke()
    izzy._zeke.steady(False)

    options.tf_net = net6.NetSix()

    # template_file = open(options.templates_dir + '/saved_template_paths.txt', 'r')
    # options.tf_net_path = '/media/1tb/Izzy/nets/net6_07-01-2016_16h24m28s.ckpt'

    #Net used for Singulation Demo 
    template_file = open(options.templates_dir + '/demo_templates_paths.txt', 'r')
    options.tf_net_path = '/media/1tb/Izzy/nets/net6_07-01-2016_16h24m28s.ckpt'


    amt = AMT(bincam, izzy, options=options)

    current_state = amt.state(amt.izzy.getState())
    amt.return_to_start(current_state)


    while True:
        print "Waiting for keypress ('q' -> quit, 'r' -> rollout, 'u' -> update weights, 'd' -> demonstrate, 'c' -> compile train/test sets, 'p' -> run on previous template, 'l' -> run templates saved in 'last_templates'): "
        char = getch()
        if char == 'q':
            print "Quitting..."
            break
        
        elif char == 'r':
            #izzy.gotoState(ZekeState([None, None, .05, None, None, None]), tra_speed = .04)
            print "Displaying template"
            singulationImg.generate_template()
            amt.display_template()
            print "Rolling out..."
            ro = amt.rollout_tf()
            print "Done rolling out."

        elif char == 'u':
            print "deprecated, and does not work"
            print "Updating weights..."
            amt.update_weights()
            print "Done updating."

        elif char == 'd':
            print "demo loop"
            names = []
            i = 0
            while True:
                try:
                    name = template_file.next()
                    names.append(name)
                except StopIteration:
                    name = names[i]
                    i += 1
                    i = i % len(names)
                print 'Using template: ' + name
                template = np.load(name[:name.find('\n')])
                amt.display_template(template)
                print "Rolling out..."
                ro = amt.rollout_tf()
                print "Done rolling out."

        elif char == 'c':
            print 'Compiling train and test sets...'
            compile_sets.compile()
            print 'Done compiling sets'

        elif char == 'p':
            print 'Displaying last template'
            amt.display_template()
            print "Rolling out..."
            ro = amt.rollout_tf()
            print "Done rolling out."

        elif char == 'l':
            #note that using this saves the wrong template. Find the template where it was referenced
            try:
                name = template_file.next()
            except StopIteration:
                print 'Completed all saved templates'
                continue
            
            print 'Using template: ' + name
            template = np.load(name[:name.find('\n')])
            amt.display_template(template)
            print "Rolling out..."
            ro = amt.rollout_tf()
            print "Done rolling out."

        elif char == 't':
            #note that using this saves the wrong template. Find the template where it was referenced
            while True:
                try:
                    name = template_file.next()
                except StopIteration:
                    print 'Completed all saved templates'
                    break
                
                print 'Using template: ' + name
                template = np.load(name[:name.find('\n')])
                amt.display_template(template)
                print "Rolling out..."
                ro = amt.rollout_tf()
                print "Done rolling out."


        # elif char == 't':
        #     amt.test()

    template_file.close()
    print "Done."
