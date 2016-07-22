import sys
sys.path.append('/home/annal/Izzy/vision_amt/scripts')
from Net.tensor import net6, net6_c
from Net.tensor import inputdata
from pipeline.bincam import BinaryCamera
import cv2
import tensorflow as tf
import overlay
import numpy as np
import time
from options import AMTOptions


def rescale(deltas):
    deltas[0] = deltas[0]*0.15
    deltas[1] = deltas[1]*0.01
    deltas[2] = deltas[2]*0.0005
    deltas[3] = deltas[3]*0.02
    return deltas

T = 2

def show(image):
    while True:
        cv2.imshow('preview', image)
        a = cv2.waitKey(30)
        if a == 27:
            break

def color(frame): 
    color_frame = cv2.resize(frame.copy(), (250, 250))
    # cv2.imwrite('get_jp.jpg',color_frame)
    # color_frame= cv2.imread('get_jp.jpg')
    return color_frame

class AnalysisReg():
    def __init__(self):
        self.net = net6.NetSix()
        self.path = '/media/1tb/Izzy/nets/net6_06-09-2016_12h06m48s.ckpt'
        #self.path = '/media/1tb/Izzy/nets/net6_06-06-2016_11h27m25s.ckpt'
        self.sess = self.net.load(self.path) 

    def compare_batches(self, reals, synths):
        differences = np.zeros((len(reals), 2))
        print "Reals         |       Synths"
        for i in range(len(differences)):
            real = reals[i]
            synth = synths[i]
            deltas_real = np.array(rescale(self.net.output(self.sess, real, 3, clasfc=False, mask=False))[:2])
            deltas_synth = np.array(rescale(self.net.output(self.sess, synth, 3, clasfc=False, mask=False))[:2])
            
            print str(deltas_real) + "  |  " + str(deltas_synth)

            differences[i] = deltas_real - deltas_synth
            
        self.diff_mean = np.mean(differences, axis=0)
        self.diff_var = np.var(differences, axis=0)


    def print_results(self):
        print "REGRESSION"
        print "Differences"
        print "rot   |   ext"
        print "mean: " + str(self.diff_mean)
        print "var:  " + str(self.diff_var)


class AnalysisClass():
    def __init__(self):
        self.net = net6_c.NetSix_C()
        self.path = '/media/1tb/Izzy/nets/net6_05-11-2016_12h09m12s.ckpt'
        self.sess = self.net.load(self.path)

    def compare_batches(self, reals, synths):
        N = len(reals)
        rot_agree = 0
        ext_agree = 0

        rot_diffs = []
        ext_diffs = []

        for i in range(N):
            print "\n"
            real = reals[i]
            synth = synths[i]
            real_dist = self.net.class_dist(self.sess, real, 3, mask=False)[:2]
            real_rot_dist, real_ext_dist = real_dist[0], real_dist[1]
            synth_dist = self.net.class_dist(self.sess, synth, 3, mask=False)[:2]
            synth_rot_dist, synth_ext_dist = synth_dist[0], synth_dist[1]
            
            print real_dist
            print synth_dist

            max_real_rot, max_synth_rot = np.argmax(real_rot_dist), np.argmax(synth_rot_dist)
            max_real_ext, max_synth_ext = np.argmax(real_ext_dist), np.argmax(synth_ext_dist)

            print "rot: " + str(np.array([max_real_rot, max_synth_rot]))
            print "ext: " + str(np.array([max_real_ext, max_synth_ext]))


            if np.argmax(real_rot_dist) == np.argmax(synth_rot_dist):
                rot_agree += 1
                print "up2"
            if np.argmax(real_ext_dist) == np.argmax(synth_ext_dist):
                ext_agree += 1
                print "up"
            
            # is this informative?
            rot_arg_diff = abs(np.argmax(real_rot_dist) - np.argmax(synth_rot_dist))
            ext_arg_diff = abs(np.argmax(real_ext_dist) - np.argmax(synth_ext_dist))
            
            rot_diffs.append(rot_arg_diff)
            ext_diffs.append(ext_arg_diff)

        #print "done looping"
        #print rot_diffs
        self.diff_rot_mean = np.mean(rot_diffs)
        self.diff_rot_var = np.var(rot_diffs)
        self.diff_ext_mean = np.mean(ext_diffs)
        self.diff_ext_var = np.var(ext_diffs)
        #print "adsf"
        self.perc_rot_agree = float(rot_agree) / float(N)
        self.perc_ext_agree = float(ext_agree) / float(N)
        


    def print_results(self):
        print "CLASSIFICATION"
        print "Agreement Percentage"
        print "rot   |   ext"
        print np.array([self.perc_rot_agree, self.perc_ext_agree])
        print "Differences"
        print "rot   |   ext"
        print "mean: " + str(np.array([self.diff_rot_mean, self.diff_ext_mean]))
        print "var:  " + str(np.array([self.diff_rot_var, self.diff_ext_var]))
    

def compare_batches(reals, synths):
    for real, synth in zip(reals, synths):
        break
    return


def process_image(image):
    image = color(image)    
    image = inputdata.im2tensor(image, channels=3)
    return image

def load_batches():
    reals = np.load('real_samples.npy')
    synths = np.load('synth_samples.npy')
    return reals, synths


def sample_real(bc, load = False):


    direct = 'scripts/objects/synth/'
    overlays = []
    for i in range(T):
        image = cv2.imread(direct + str(i) + '.png')
        overlays.append(image)
    reals = []
    if load:
        reals = np.load('real_samples.npy')
    for i in range(T):
        try:
            while True:
                frame = inputdata.im2tensor(bc.read_frame(), channels=3)
                #frame = process_image(frame)
                final = overlay.overlay(frame, overlays[i])

                cv2.imshow('preview', final)
                cv2.waitKey(30)
                time.sleep(.005)
                        
        except KeyboardInterrupt:
            pass
        print i
        frame = bc.read_frame()
        frame = color(frame)
        reals.append(frame)
        #np.save('real_samples.npy', reals)
    #np.save('real_samples.npy', reals)


def sample_synth():
    direct = 'scripts/objects/synth/'
    synths = []
    for i in range(T):
        im_path = direct + str(i) + '.png'
        im = cv2.imread(im_path)
        im = color(im)
        synths.append(im)
    np.save('synth_samples.npy', synths)
        


def combine_mask(reals, synths):
    for i in range(len(reals)):
        real = inputdata.im2tensor(reals[i], channels=3)
        synth = inputdata.im2tensor(synths[i], channels=3)
        gripper_channel = real[:,:,1]
        for j in range(3):
            synth[:,:,j] += gripper_channel
        reals[i] = real
        synths[i] = synth
    return reals, synths

def sample_rollouts(start, end):
    """
        Get real and synth samples from trajectories
        (int) start to (int) end
        return lists of resized but unmasked images as np.array
    """
    reals = []
    synths = []
    for i in range(start, end + 1):
        dir_path = AMTOptions.rollouts_dir + 'rollout' + str(i) + '/'
        synth_path = dir_path + 'template.npy'
        real_path = dir_path + 'rollout' + str(i) + '_frame_0.jpg'
        real = cv2.imread(real_path)
        synth = np.load(synth_path)
        real = color(real)
        synth = color(synth)
        reals.append(real)
        synths.append(synth)
    return reals, synths

if __name__ == '__main__':
    #sample_synth()
    #bc = BinaryCamera("./meta.txt")
    #bc.open()
    #time.sleep(1)
    #sample_real(bc)                                # obtain real samples (uncomment everything above)
    #reals, synths = load_batches()                 # load saved samples from previous
    reals, synths = sample_rollouts(1608,1698)    # search rollouts for samples
    reals, synths = combine_mask(reals, synths)
    #show(reals[24] * 255.0)
    #show(synths[24] * 255.0)
    
    #show(reals[0] * 255.0)
    with tf.Graph().as_default():
        ar = AnalysisReg()
        ar.compare_batches(reals, synths)
        ar.print_results()
    with tf.Graph().as_default():
        ac = AnalysisClass()
        ac.compare_batches(reals, synths)
        ac.print_results()




