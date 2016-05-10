from options import AMTOptions
import numpy as np
import random


def scale(deltas):
    deltas[0] = float(deltas[0])/0.2
    deltas[1] = float(deltas[1])/0.01
    deltas[2] = 0.0#.0float(deltas[2])/0.005
    deltas[3] = float(deltas[3])/0.2
    return deltas

def reg_to_class(deltas):
    cls = np.zeros([20])
    idx_0 = np.round(2*deltas[0]+2)
    idx_1 = np.round(2*deltas[1]+2)+5
    idx_2 = np.round(2*deltas[2]+2)+10
    idx_3 = np.round(2*deltas[3]+2)+15

    cls[idx_0] = 1.0
    cls[idx_1] = 1.0
    cls[idx_2] = 1.0
    cls[idx_3] = 1.0
    return cls

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
        deltas = scale(labels[1:5])
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
    compile()
