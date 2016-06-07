from Net.tensor import net6, net6_c
from Net.tensor import inputdata
from pipeline.bincam import BinaryCamera
import cv2
import tensorflow as tf
import numpy as np

def rescale(deltas):
    deltas[0] = deltas[0]*0.15
    deltas[1] = deltas[1]*0.01
    deltas[2] = deltas[2]*0.0005
    deltas[3] = deltas[3]*0.02
    return deltas



def color(frame): 
    color_frame = cv2.resize(frame.copy(), (250, 250))
    # cv2.imwrite('get_jp.jpg',color_frame)
    # color_frame= cv2.imread('get_jp.jpg')
    return color_frame

class AnalysisReg():
    def __init__(self):
        self.net = net6.NetSix()
        self.path = '/media/1tb/Izzy/nets/net6_06-06-2016_11h27m25s.ckpt'
        self.sess = net.load(path) 

    def compare_batches(self, reals, synths):
        differences = np.zeros((len(reals), 2))
        print "Reals         |       Synths"
        for i in range(len(differences)):
            real = reals[i]
            synth = synths[i]
            deltas_real = np.array(rescale(self.net.output(self.sess, real, 3, False))[:2])
            deltas_synth = np.array(rescale(self.net.output(self.sess, synth, 3, False))[:2])
            print str(deltas_real) + "  |  " + str(deltas_synth)
            differences[i] = deltas_real - deltas_synth
        self.diff_mean = np.mean(differences, axis=0)
        self.diff_var = np.var(differences, axis=0)


    def print_results():
        print "REGRESSION"
        print "Differences"
        print "rot   |   ext"
        print "mean: " + self.diff_mean
        print "var:  " + self.diff_var


class AnalysisClass():
    def __init__(self):
        self.net = net6_c.NetSix_C()
        self.path = '/media/1tb/Izzy/nets/net6_05-11-2016_12h09m12s.ckpt'
        self.sess = net.load(path)

    def compare_batches(self, reals, synths):
        N = len(reals)
        rot_agree = 0
        ext_agree = 0

        rot_diffs = []
        ext_diffs = []

        for i in range(N):
            real = reals[i]
            synth = synths[i]
            real_dist = self.net.class_dist(self.sess, real, 3)[:2]
            real_rot_dist, real_ext_dist = real_dist[0], real_dist[1]
            synth_dist = self.net.class_dist(self.sess, real, 3)[:2]
            synth_rot_dist, synth_ext_dist = synth_dist[0], synth_dist[1]

            if np.argmax(real_rot_dist) == np.argmax(synth_rot_dist):
                rot_agree += 1
            if np.argmax(real_rot_dist) == np.argmax(synth_rot_dist):
                ext_agree += 1
            
            # is this informative?
            rot_arg_diff = abs(np.argmax(real_rot_dist) - np.argmax(synth_rot_dist))
            ext_arg_diff = abs(np.argmax(real_ext_dist) - np.argmax(synth_ext_dist))
            
            rot_diffs.append(rot_diffs)
            ext_diffs.append(ext_diffs)

        
        self.diff_rot_mean = np.mean(rot_diffs)
        self.diff_rot_var = np.var(rot_diffs)
        self.diff_ext_mean = np.mean(ext_diffs)
        self.diff_ext_var = np.var(ext_diffs)
        
            
        self.perc_rot_agree = float(rot_agree) / N
        self.perc_ext_agree = float(ext_agree) / N
        

    def print_results():
        print "CLASSIFICATION"
        print "Agreement Percentage"
        print "rot   |   ext"
        print np.array([self.perc_rot_agree, self.perc_ext_agree])
        print "Differences"
        print "rot   |   ext"
        print "mean: " + np.array([self.diff_rot_mean, self.diff_ext_mean])
        print "var:  " + np.array([self.diff_rot_var, self.diff_ext_var])
    

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


def sample_real(load = False):
    bc = BinaryCamera('./meta.txt')
    bc.open()
    reals = []
    if load:
        reals = np.load('real_samples.npy')
    for i in range(30):
        try:
            while True:
                frame = bc.read_frame()
                process_image(frame)
                cv2.imshow('preview', frame)
                cv2.waitKey(30)

            frame = bc.read_frame()
            frame = process_image(frame)
            reals.append(frame)
            np.save('real_samples.npy', reals)            
        except KeyboardInterrupt:
            pass
    np.save('real_samples.npy', reals)


def sample_synth():
    direct = 'objects/'
    synths = []
    for i in range(30):
        im_path = direct + str(i) + '.png'
        im = cv2.imread(im_path)
        im = process_image(im)
        synths.append(im)
    np.save('synth_samples.npy', synths)
        


if __name__ == '__main__':
    sample_synth()
    sample_real()
    reals, synths = load_batches()
    ar = AnalysisReg()
    ar.compare_batches(reals, synths)
    ar.print_results()
    #ac = AnalysisClass()

