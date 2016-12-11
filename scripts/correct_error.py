import cv2, sys, argparse
sys.path.append('/home/huarsc/research/vision-amt/')
# sys.path.append('/home/annal/Izzy/vision_amt/')
import time
from options import AMTOptions
import numpy as np
import matplotlib.pyplot as plt
import scripts.izzy_feedback as feedback
import scripts.visualizer_supervised as vis

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

def convert_delta(delta):
    new_delta = [0.0] * 6
    new_delta[0] = delta[0]
    new_delta[2] = delta[1]
    return new_delta



def highest_errors(all_names_errs, num):
    highest = [(-1, -1) for _ in range(num)]
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
    errs = []
    for name, e, label, ev, stv in all_names_errs:
        err = np.linalg.norm(e)
        errs.append(err)
        if err > threshold:
            print(e)
            over.append(i)
        i += 1
    # errs.sort()
    # plt.plot(errs)
    # plt.show()
    return over

def read_err_line(line):
    vals = line.split('\t')
    name = vals[0]
    sp = vals[1].split(' ')
    e = np.array([float(sp[0]), float(sp[1])])
    sp = vals[2].split(' ')
    label = np.array([float(sp[0]), float(sp[1])])
    sp = vals[3].split(' ')
    ev = np.array([float(sp[0]), float(sp[1])])
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
        try:
            name, e, label, ev, stv = all_names_errs[at]
            curnum = traj_num(name)
            if curnum != rnum:
                break
            evals = evals + [ev]
            states = states + [stv]
            at += 1
        except IndexError as e:
            break
    return evals, states




def gather_feedback(error_path, new_deltas_path, output_file_path, pname='', test=''):
    errs_vals = open(error_path, 'r')
    all_names_errs = []

    new_deltas = dict()
    for name, e, label, ev, stv in all_names_errs:
        new_deltas.append(label)

    for line in errs_vals:
        all_names_errs.append(read_err_line(line))
    above = threshold_errs(all_names_errs, 1.3)
    worst = highest_errors(all_names_errs, 20)

    # for a in above:
    #     name, e, label, ev, stv = all_names_errs[a]
    #     i = frame_num(name)
    #     rnum = traj_num(name)
    #     pth = assign_path(AMTOptions.supervised_dir, rnum, AMTOptions, pname, frame_num=i)
    #     evaluations, states = get_evaluations(all_names_errs, a)
    #     # base_im = draw_trajectory(pth, rnum,  i)
    #     # impath = pth[:pth.find('.jpg')-1]
    #     if rnum in new_deltas:
    #         deltas = feedback.correct_rollout(pth, i, states, evaluations, rnum, pname, test, new_deltas[rnum])
    #     else:            
    #         deltas = feedback.correct_rollout(pth, i, states, evaluations, rnum, pname, test)
    #     if deltas:
    #         if rnum not in new_deltas:
    #             new_deltas[rnum] = dict()
    #         for index, delta in deltas.items():
    #             new_deltas[rnum][index] = delta
    # new_delta_file = open(new_deltas_path + output_file_path, 'w')
    # for rnum in new_deltas.keys():
    #     for fnum in new_deltas[rnum].keys():
    #         new_delta_file.write(pname + "\t" + str(rnum) + "\t" + str(fnum) + "\t" + str(new_deltas[rnum][fnum][0]) + "\t" + str(new_deltas[rnum][fnum][1]) + "\n")
    # new_delta_file.close()



    # new_remove_file = open(new_deltas_path + "new_remove.txt", 'w')
    # for a in above:
    #     name, e, label, ev, stv = all_names_errs[a]
    #     fnum = frame_num(name)
    #     rnum = traj_num(name)
    #     new_remove_file.write(pname + "\t" + str(rnum) + "\t" + str(fnum) + "\t" + "d" + "\t" + "e" + "\n")
    # new_remove_file.close()

    new_robot_replace_file = open(new_deltas_path + "new_robot_alter.txt", 'w')
    for a in above:
        name, e, label, ev, stv = all_names_errs[a]
        delta = vis.rescale_sup(ev)
        fnum = frame_num(name)
        rnum = traj_num(name)
        new_robot_replace_file.write(pname + "\t" + str(rnum) + "\t" + str(fnum) + "\t" + str(delta[0]) + "\t" + str(delta[1]) + "\n")
    new_robot_replace_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="run experiment, will prompt for name")
    parser.add_argument("-t", "--test", type=str, help="test folder name")
    parser.add_argument("-o", "--output", type=str, help="file to output values to")
    args = parser.parse_args()
    if args.test == None:
        args.test = ''
    gather_feedback(AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/comparisons.txt", AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/", args.output, args.name, args.test)
