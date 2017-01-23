from options import AMTOptions
import numpy as np, argparse
from scripts import compile_supervisor, merge_supervised
from scipy import signal


def traj_num(line):
    if line.find('_supervised') != -1:
        return int(line[line.find('_supervised') + len('_supervised'):line.find('_frame_')])
    else:
        return int(line[line.find('_rollout') + len('_rollout'):line.find('_frame_')])

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

def replace(x, vals):
    d_line = x.split(' ')
    return d_line[0] + ' ' + ' '.join([str(val) for val in vals]) + '\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-tg", "--target", type=str,
                        help="name of the file to place the target changed values, with format PATH, ORIGINAL VALUE, NEW VALUE, DIFFERENCE")
    parser.add_argument("-sm", "--smooth", type=int,
                        help="determines if a butterworth filter should be applied over the data, of order N")
    parser.add_argument("-ho", "--holdout", type=str,
                        help="a path to a file, which contains the line 'Holdout Trajectories:' followed by space separated holdout numbers")
    parser.add_argument("-n", "--name", type=str,
                    help="run experiment, will prompt for name")
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollouts to be used for training")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollouts to be used for training")
    args = parser.parse_args()

    holdout = []
    skip_file = open(AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/" + args.holdout, 'r')
    for line in skip_file:
        if line.find("Holdout Trajectories:") != -1:
            values = line.split(' ')
            print values
            for value in values[2:-1]:
                holdout.append(int(value))

    outfile = open(AMTOptions.amt_dir + 'deltas.txt', 'w+')
    failure = merge_supervised.load_rollouts(False, False, (args.first,args.last), (0,100), outfile, name = args.name)
    if failure:
        print "did not have the sufficient rollouts specified"
    outfile.close()


    deltas_path = AMTOptions.deltas_file
    deltas_file = open(deltas_path, 'r')
    difference_file = open(AMTOptions.amt_dir + args.target, 'w')
    print AMTOptions.amt_dir + args.target
    full_trajectory = []

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
        
        if last_rol != r_num:
            if last_rol != -1:
                next_trajectory = np.array(last_trajectory)
                a,b = signal.butter(args.smooth, 0.05)
                for r in range(next_trajectory.shape[1] - 1):
                    next_trajectory[:,r] = signal.filtfilt(a, b, next_trajectory[:,r])
                # print next_trajectory, last_trajectory
                for i in range(len(last_trajectory_str)):
                    diff = np.linalg.norm(next_trajectory[i] - np.array(last_trajectory[i]))
                    # print diff, last_trajectory[i], next_trajectory[i]
                    last_trajectory_str[i] = replace(last_trajectory_str[i], next_trajectory[i])
                    difference_file.write(path + last_trajectory_str[i][:-1] + " " + " ".join(map(str, last_trajectory[i])) + " " + str(diff) + "\n")
            last_rol = r_num
            last_trajectory = []
            last_trajectory_str = []
        last_trajectory.append(deltas)
        last_trajectory_str.append(line)
    deltas_file.close()
    difference_file.close()


