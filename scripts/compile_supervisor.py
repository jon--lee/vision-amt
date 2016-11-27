import sys
sys.path.append('/home/annal/Izzy/vision_amt/')
from options import AMTOptions
import numpy as np
from scipy import signal
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

def traj_num(line):
    if line.find('_supervised') != -1:
        return int(line[line.find('_supervised') + len('_supervised'):line.find('_frame_')])
    else:
        return int(line[line.find('_rollout') + len('_rollout'):line.find('_frame_')])

def get_values(x):
    values = x.split(' ')
    return [float(values[1]), float(values[2])]

def replace(x, vals):
    d_line = x.split(' ')
    return d_line[0] + ' ' + ' '.join([str(val) for val in vals]) + '\n'

def resevoir(rols_used, k):
    R = [0 for i in range(k)]
    for i in range(k):
        R[i] = rols_used[i]

    for i in range(k,len(rols_used)):
        j = random.randint(0, i)
        if j < k:
            R[j] = rols_used[i]
    return R

def compile_reg(skipped=[], smooth = None, num=10):
    train_path = AMTOptions.train_file
    test_path = AMTOptions.test_file
    deltas_path = AMTOptions.deltas_file
    
    print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path
    train_file = open(train_path, 'w+')
    test_file = open(test_path, 'w+')

    if len(skipped) == 0:
        get_new_skip = True
    else:
        get_new_skip = False
    skipped = set(skipped)
    seen = set()
    
    if get_new_skip:
        deltas_file = open(deltas_path, 'r')
        rols_used = set()
        for line in deltas_file:
            rols_used.add(traj_num(line))
        skipped = resevoir(list(rols_used), num)
        deltas_file.close()

    deltas_file = open(deltas_path, 'r')

    last_trajectory = []
    last_trajectory_str = []
    last_rol = -1
    for line in deltas_file:
        r_num = traj_num(line)
                
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
        
        if r_num in skipped:
            test_file.write(path + line)
        # elif r_num not in seen:
        #     if get_new_skip:
        #         if random.random() > .2:
        #             train_file.write(path + line)
        #             seen.add(r_num)
        #         else:
        #             test_file.write(path+line)
        #             skipped.add(r_num)
        #     else:
        #         train_file.write(path + line)
        #         seen.add(r_num)
        elif smooth is not None:
            if last_rol != r_num:
                if last_rol != -1:
                    last_trajectory = np.array(last_trajectory)
                    a,b = signal.butter(smooth, 0.05)
                    for r in range(last_trajectory.shape[1]):
                        last_trajectory[:,r] = signal.filtfilt(a, b, last_trajectory[:,r])
                    for i in range(len(last_trajectory_str)):
                        last_trajectory_str[i] = replace(last_trajectory_str[i], last_trajectory[i])
                        train_file.write(path + last_trajectory_str[i])
                last_rol = r_num
                last_trajectory = []
                last_trajectory_str = []
            last_trajectory.append(deltas)
            last_trajectory_str.append(line)
        else:
            train_file.write(path + line)

    print "skipped: "
    print skipped
    return skipped


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
        if random.random() > .12:
            train_file.write(path + line)
        else:
            test_file.write(path + line)
    
    print "Done moving images and labels."


if __name__ == '__main__':
    compile_reg()
