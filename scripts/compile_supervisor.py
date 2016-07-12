import sys
sys.path.append('/home/annal/Izzy/vision_amt/')
from options import AMTOptions
import numpy as np
import random


def norm_forward(f_val):
    return float(f_val)/0.006 

def norm_ang(ang):
    # return float(ang)/0.02
    return float(ang)/0.02666666666667 

def scale(deltas):
    deltas[0] = norm_ang(deltas[0])
    deltas[1] = norm_forward(deltas[1])
    # deltas[2] = 0.0#.0float(deltas[2])/0.005
    # deltas[3] = float(deltas[3])/0.2
    return deltas

def reg_to_class(deltas):
    cls = np.zeros([10])
    angles = np.array(AMTOptions.CLASS_ANGLES)/.026666666666667
    idx_0 = np.argmin (np.abs(angles - deltas[0]))
    # print deltas[0], angles, idx_0
    forward = np.array(AMTOptions.CLASS_FORWARD)/.006
    idx_1 = np.argmin(np.abs(forward - deltas[1]))

    cls[idx_0] = 1.0
    cls[idx_1 + 5] = 1.0
    return cls

def compile_reg():
    train_path = AMTOptions.train_file
    test_path = AMTOptions.test_file
    deltas_path = AMTOptions.deltas_file
    
    print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path
    train_file = open(train_path, 'w+')
    test_file = open(test_path, 'w+')
    deltas_file = open(deltas_path, 'r')
    
    for line in deltas_file:
        path = AMTOptions.colors_dir
        labels = line.split()
        # print line
        deltas = scale(labels[1:3])
        # line = labels[0] + " " + str(deltas[0]) + " " + str(deltas[1]) + " " + str(deltas[2]) + " " + str(deltas[3]) + "\n
        line = labels[0] + " " 
        for bit in deltas:
            line += str(bit) + " "
        line = line[:-1] + '\n'
        # train_file.write(line)
        if random.random() > .2:
            train_file.write(path + line)
        else:
            test_file.write(path + line)


def compile():
    train_path = AMTOptions.train_file
    test_path = AMTOptions.test_file
    deltas_path = AMTOptions.deltas_file
    #deltas_path = AMTOptions.data_dir + 'amt/labels_amt_exp_mrg.txt'    
    #deltas_path = AMTOptions.data_dir + 'amt/deltas_0_60.txt'
    #deltas_path = AMTOptions.data_dir + 'amt/amt__exp_me_full.txt'


    print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path

    train_file = open(train_path, 'w+')
    test_file = open(test_path, 'w+')
    deltas_file = open(deltas_path, 'r')

    for line in deltas_file:            
        #path = AMTOptions.originals_dir
        path = AMTOptions.colors_dir 
        labels = line.split()
        deltas = scale(labels[1:3])
        deltas_c = reg_to_class(deltas)

        line = labels[0]#+" "+str(deltas[0])+" "+str(deltas[1])+" "+str(deltas[2])+" "+str(deltas[3])+"\n"
        for i in range(20):
            line += " "+str(deltas_c[i])

        line += "\n"
        if random.random() > .2:
            train_file.write(path + line)
        else:
            test_file.write(path + line)
    
    print "Done moving images and labels."


if __name__ == '__main__':
    compile_reg()
