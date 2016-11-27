import sys, os, time
sys.path.append('/home/annal/Izzy/vision_amt/')
from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net6_c,net8
from Net.tensor import inputdata
from options import AMTOptions
import numpy as np, argparse
from scripts import compile_supervisor, merge_supervised
from tensorflow.python.framework import ops

def traj_num(line):
    if line.find('_supervised') != -1:
        return int(line[line.find('_supervised') + len('_supervised'):line.find('_frame_')])
    else:
        return int(line[line.find('_rollout') + len('_rollout'):line.find('_frame_')])

def get_traj(deltas_path):
    deltas_file = open(deltas_path, 'r')
    rols_used = set()
    for line in deltas_file:
        rols_used.add(traj_num(line))
    deltas_file.close()
    return list(rols_used)


def kfolds(k, trajs):
    '''
    performs k folds on trajectory numbers, where the last fold is the holdout
    '''
    num_fold = len(trajs)/k
    np.random.shuffle(trajs)
    return num_fold, [trajs[:num_fold*i] + trajs[num_fold*(i+1):num_fold*(k-1)] for i in range(k-1)], trajs[num_fold*(k-1):]

def write_errors(net, data, sess):
    evaluations = []
    full_batch = data.all_test_batch(50)
    j = 0
    with sess.as_default():
        while full_batch is not None:
            full_ims, full_labels = full_batch
            full_dict = { net.x: full_ims, net.y_: full_labels }
            print "Evaluating... finished ", j
            num_used = len(full_ims)
            print num_used
            j += len(full_ims)
            full_batch = data.all_test_batch(50)
            evaluations += net.y_out.eval(feed_dict=full_dict).tolist()
            # rot_loss = net.rot.eval(feed_dict=full_dict) # regression
            # fore_loss = net.fore.eval(feed_dict=full_dict) # regression
            # norm_loss = rot_loss/.02666667 + fore_loss/.006
            # losses = [losses[0] + rot_loss*num_used , losses[1] + fore_loss*num_used, losses[2]+norm_loss*num_used]
    print "done evaluating"
    return np.array(evaluations)

def states_file_path(name, test, traj):
    return AMTOptions.supervised_dir + name + "_rollouts/" + test + "/supervised" + str(traj)+ "/states.txt"

def convert_state(state):
    # converts a state into a length six list, 
    # differentiates between four number and six number states
    split = state.split(' ')
    if len(split) == 5:
        new_state = [0.0] * 6
        new_state[0] = float(split[1])
        new_state[2] = float(split[2])
        return new_state
    elif len(split) == 7:
        # print "state", state, "state", string_state(state), "fin"
        return string_state(state)

def string_state(state):
    split = state.split(' ')
    # print state
    labels = np.array( [ float(x) for x in split[1:] ] )
    # print labels
    return labels

def stringify_state(state):
    total = ''
    for val in state:
        total += str(val) + ' '
    return total[:-1]

def k_test(k, deltas_path, threshold, errors_path, name, test):
    trajs = get_traj(deltas_path)
    n_in, folds, holdout = kfolds(k,trajs)
    # print trajs

    print holdout
    sholdout = set(holdout)
    # print folds

    states = []
    for traj in trajs:
        if traj not in sholdout:
            states_file = open(states_file_path(name, test, traj), 'r')
            for state in states_file:
                states.append(convert_state(state))
            states_file.close()

    k_at = 0
    path = AMTOptions.colors_dir
    all_names_errs = []
    train_nets = []


    for group in folds:
        print "iteration " +str(k_at)
        sgroup = set(group)
        new_train = AMTOptions.amt_dir + 'train_folds.txt'
        new_test = AMTOptions.amt_dir + 'test_folds.txt'

        old_train = deltas_path

        old_trainf = open(old_train, 'r')
        testf = open(new_test, 'w')
        trainf = open(new_train, 'w')
        for line in old_trainf:
            labels = line.split()
            deltas = compile_supervisor.scale(labels[1:3])
            line = labels[0] + " " 
            for bit in deltas:
                line += str(bit) + " "
            line = line[:-1] + '\n'
            if traj_num(line) in sgroup:
                trainf.write(path+ line)
            elif traj_num(line) not in sholdout:
                testf.write(path+ line)
        old_trainf.close()
        trainf.close()
        testf.close()
        data = inputdata.AMTData(new_train, new_test,channels=3, shuffle = False)
        net = net6.NetSix()
        net_name = net.optimize(200,data, batch_size=100)
        outf = open(AMTOptions.amt_dir + 'last_net.txt', 'w')
        outf.write(net_name)
        train_nets.append(net_name)
        outf.close()
        ops.reset_default_graph()
        net = net6.NetSix()
        sess = net.load(var_path=net_name)
        evaluations = write_errors(net, data, sess)
        sess.close()
        data.all_reset()
        labels = []
        named_errs = []
        full_batch = data.all_test_batch(500)
        full_paths = data.all_test_paths_batch(500)
        while full_batch is not None:
            full_ims, full_labels = full_batch
            full_impaths, full_labels = full_paths
            labels += full_labels
            named_errs += full_impaths
            full_batch = data.all_test_batch(500)
            full_paths = data.all_test_paths_batch(500)
        labels = np.array(labels)
        errs = np.abs(evaluations-labels).tolist()
        named_errs = zip(named_errs, errs, labels.tolist(), evaluations.tolist())
        # worst = [-1 for i in range(10)]
        # below = []
        # i = 0
        # # print [ev for ev in evaluations]
        # for name, e, label, ev in named_errs:
        #     if np.linalg.norm(e) > threshold:
        #         below.append(i + k_at*n_in)
        #     if np.linalg.norm(e) > np.linalg.norm(worst[0]):
        #         worst[0] = e
        #         worst.sort(key=lambda e: np.linalg.norm(e))
        #     i += 1
        print k_at
        k_at += 1
        all_names_errs += named_errs
        # print all_names_errs 
        ops.reset_default_graph()
        time.sleep(3)
    error_file = open(errors_path + "comparisons.txt", 'w')
    statistics_eval = []
    i=0
    print len(states)
    print len(all_names_errs)
    for name, e, label, ev in all_names_errs:
        #statistics_eval.append((name, e, label, ev, states[i]))
        error_file.write(name + "\t" + str(e[0]) + " " + str(e[1]) + "\t" + str(label[0]) + " " + str(label[1]) + "\t" + str(ev[0]) + " " + str(ev[1]) + "\t" + stringify_state(states[i]) + "\n")
        i += 1
    error_file.close()
    # write some values
    stats_file = open(errors_path + "stats.txt", 'w')
    stats_file.write(str(train_nets) + '\n')
    holdouts = "Holdout Trajectories: "
    for hout in holdout:
        holdouts += str(hout) + " "
    stats_file.write(holdouts)
    stats_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="run experiment, will prompt for name")
    parser.add_argument("-t", "--test", type=str, help="test folder name")
    parser.add_argument("-o", "--output", type=str, help="file to output values to")
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollouts to be used for training")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollouts to be used for training")
    parser.add_argument("-i", "--initial", type=int,
                        help="enter the initial value of frames to be used for training")
    parser.add_argument("-e", "--end", type=int,
                        help="enter the ending value of the frames to be used for training")
    args = parser.parse_args()
    if args.test == None:
        args.test = ''
    if args.first is not None:
        first = args.first
    else:
        print "please enter a first value with -f"
        sys.exit()
    if args.last is not None:
        last = args.last
    else:
        print "please enter a last value with -l (not inclusive)"
        sys.exit()
    initial = args.initial if args.initial is not None else 0
    end = args.end if args.end is not None else 100

    outfile = open(AMTOptions.amt_dir + 'deltas.txt', 'w+')
    failure = merge_supervised.load_rollouts(False, False, (first,last), (initial,end), outfile, name = args.name)
    if failure:
        print "did not have the sufficient rollouts specified"
    outfile.close()
    k_test(12, AMTOptions.deltas_file, .02, AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/", args.name, args.test)
