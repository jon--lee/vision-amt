import cv2, sys
sys.path.append('/home/annal/Izzy/vision_amt/')
import time
from options import AMTOptions
import numpy as np
from Net.tensor import inputdata, net3,net4,net5,net6, net6_c
import matplotlib.pyplot as plt
import scripts.izzy_feedback as feedback
import scripts.visualizer_supervised.py as vis

def get_names(name):
    if len(name) != 0:
        folname = '/' + name + '_rollouts/'
        imname = '/' + name + '_'
    return folname, imname

def traj_num(line):
    if line.find('_supervised') != -1:
        return int(line[line.find('_supervised') + len('_supervised'):line.find('_frame_')])
    else:
        return int(line[line.find('_rollout') + len('_rollout'):line.find('_frame_')])

def frame_num(line):
    if line.find('_supervised') != -1:
        return int(line[line.find('_frame_') + len('_frame_'):line.find('.jpg')])
    else:
        return int(line[line.find('_frame_') + len('_frame_'):line.find('.jpg')])

def get_test(folname, imname, test):
    if len(test) != 0:
        folname = folname + test + '/'
        imname = imname + test + '_'
    print test, folname
    return folname, imname

def states_file_path(name, test, traj):
    return AMTOptions.supervised_dir + name + "_rollouts/" + test + "/supervised" + str(traj)+ "/states.txt"


def assign_path(image_path, rollout_num, options, name='', test='', frame_num = 0):
    folname, imname = get_names(name)
    folname, imname = get_test(folname, imname, test)
    if image_path == options.supervised_dir:
        return options.supervised_dir + folname + 'supervised'+str(rollout_num) +'/' +name + '_supervised'+str(rollout_num) + '_frame_' + str(frame_num) + '.jpg'
    elif image_path == options.rollouts_dir:
        return options.rollouts_dir + folname + 'rollout'+str(rollout_num) +'/rollout'+str(rollout_num) + '_frame_' + str(frame_num) + '.jpg'
    return None

def convert_delta(f_delta):
    new_delta = [0.0] * 6
    new_delta[0] = delta[0]
    new_delta[2] = delta[1]
    return new_delta

def draw_trajectory(image_path, traj_num, evaluations, name='', test = '', hard=0):
    initial_image = cv2.imread(image_path)
    states = open(states_file_path(name, test, traj_num), 'r')
    first = True
    i = 0
    for stv, delta in zip(states, evaluations):
        if first:
            last = vis.bound_state(vis.string_state(stv))
            first = False
            continue
        state = vis.bound_state(vis.string_state(stv))
        sup_change = vis.states_to_line(last, state)
        state_pts = vis.state_to_pixel(state)

        f_delta = convert_delta(f_delta)
        f_change = vis.command_to_line(f_delta, state)
        if hard == i:
            color = (0,191,191)
        else:
            color = (50, 100, 100)
        vis.draw_result(initial_image, f_change, color = (0,191,191))
        vis.draw_result(initial_image, sup_change, color = (0,255,0), thick = 1)
        state = last
        i += 1
    return initial_image

def highest_errors(all_names_errs, num):
    highest = [-1 for _ in range(num)]
    i = 0
    for name, e, label, ev, stv in all_names_errs:
        err = np.linalg.norm(e)
        if err > highest[-1][1]:
            highest[-1] = (i, err)
        i += 1
    return highest

def threshold_errs(all_names_errs, threshold):
    over = []
    i = 0
    for name, e, label, ev, stv in all_names_errs:
        err = np.linalg.norm(e)
        if err > threshold:
            over.append(i)
        i += 1
    return over

def read_err_line(line):
    vals = line.split('\t')
    name = vals[0]
    sp = vals[1].split(' ')
    e = np.array([float(sp[0]), float(sp[1])])
    sp = vals[2].split(' ')
    label = np.array([float(sp[0]), float(sp[1])])
    sp = vals[3].split(' ')
    evaluation = np.array([float(sp[0]), float(sp[1])])
    sp = vals[4].split(' ')
    state = np.array([float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])])
    return (name, e, label, ev, state)

def get_evaluations(all_names_errs, i):
    evals = []
    states = []
    name, e, label, ev, stv = all_names_errs[i]
    rnum = traj_num(name)
    curnum = rnum 
    at = i
    while curnum == rnum:
        evals = [ev] + evals
        states = [stv] + states
        at -= 1
        name, e, label, ev, stv = all_names_errs[at]
        curnum = traj_num(name)

    curnum = rnum 
    at = i+1
    while True:
        name, e, label, ev, stv = all_names_errs[at]
        curnum = traj_num(name)
        if curnum != rnum:
            break
        evals = [ev] + evals
        states = [stv] + states
        at += 1
    return evals, states




def gather_feedback(error_path, new_deltas_path, pname='', test=''):
    errs_vals = open(error_path, 'r')
    all_names_errs = []

    new_deltas = []
    for name, e, label, ev, stv in all_names_errs:
        new_deltas.append(label)

    for line in errs_vals:
        all_names_errs.append(read_err_line(line))
    above = threshold_errs(all_names_errs, .001)
    worst = highest_errors(all_names_errs, 20)

    for a in above:
        name, e, label, ev, stv = all_names_errs[a]:
        i = frame_num(name)
        rnum = traj_num(name)
        pth = assign_path(AMTOptions.supervised_dir, rnum, AMTOptions, pname, i)
        evaluations, states = get_evaluations_states(all_names_errs, i)
        base_im = draw_trajectory(pth, rnum, evaluations, pname, test, hard)
        deltas = feedback.correct_rollout(deltas, image_path, i, states)
        for delta in deltas:
            new_deltas[a + i - delta(0)] = delta[1]
    new_delta_file = open(new_deltas_path, 'w')
    for delta in new_deltas:
        new_delta_file.write(new_deltas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="run experiment, will prompt for name")
    parser.add_argument("-t", "--test", type=str, help="test folder name")
    parser.add_argument("-o", "--output", type=str, help="file to output values to")
    args = parser.parse_args()
    if args.test == None:
        args.test = ''
    k_test(6, AMTOptions.deltas_file, .02, AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/" + args.output, args.name, args.test)