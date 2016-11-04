import sys, os
sys.path.append('/home/annal/Izzy/vision_amt/')
from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net6_c,net8
from Net.tensor import inputdata
from options import AMTOptions
import numpy as np, argparse
from scripts import compile_supervisor, merge_supervised


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
    num_fold = len(trajs)/k
    np.random.shuffle(trajs)
    return num_fold, [trajs[:num_fold*i] + trajs[num_fold*(i+1):] for i in range(k)]

def write_errors(new_name, data, sess):
	evaluations = []
	full_batch = data.all_test_batch(500)
	with sess.as_default():
	    while full_batch is not None:
	        full_ims, full_labels = full_batch
	        full_dict = { net.x: full_ims, net.y_: full_labels }
	        print "Evaluating... finished ", j
	        num_used = len(full_ims)
	        j += len(full_ims)
	        full_batch = data.all_test_batch(500)
	        evaluations += net.y_out.eval(feed_dict=full_dict).tolist()
	        # rot_loss = net.rot.eval(feed_dict=full_dict) # regression
	        # fore_loss = net.fore.eval(feed_dict=full_dict) # regression
	        # norm_loss = rot_loss/.02666667 + fore_loss/.006
	        # losses = [losses[0] + rot_loss*num_used , losses[1] + fore_loss*num_used, losses[2]+norm_loss*num_used]
	return np.array(evaluations)

def k_test(k, deltas_path, threshold, name ='Caleb'):
    trajs = get_traj(deltas_path)
    print trajs
    n_in, folds = kfolds(k,trajs)

    k_at = 0
    for group in folds:
        sgroup = set(group)
        new_train = AMTOptions.amt_dir + 'train_folds.txt'
        new_test = AMTOptions.amt_dir + 'test_folds.txt'

        old_train = deltas_path

        old_trainf = open(old_train, 'r')
        testf = open(new_test, 'w')
        trainf = open(new_train, 'w')
        for line in old_trainf:
            if traj_num(line) in sgroup:
                trainf.write(AMTOptions.supervised_dir + name + '_rollouts/supervised' + str(traj_num(line)) + '/' + line)
            else:
                testf.write(AMTOptions.supervised_dir + name + '_rollouts/supervised' + str(traj_num(line)) + '/' + line)
        old_trainf.close()
        trainf.close()
        testf.close()
        data = inputdata.AMTData(new_train, new_test,channels=3)
        net = net6.NetSix()
        net_name = net.optimize(400,data, batch_size=100)
        outf = open(AMTOptions.amt_dir + 'last_net.txt', 'w')
        outf.write(net_name)
        outf.close()
        options.tf_net_path = new_name
    	sess = net.load(var_path=options.tf_net_path)
    	evaluations = write_errors(new_name, data, sess)
        data.all_train_reset()
        labels = []
        full_batch = data.all_test_batch(500)
        while full_batch is not None:
            full_ims, full_labels = full_batch
            labels += full_labels
            full_batch = data.all_test_batch(500)
        labels = np.array(labels)
        errs = np.abs(evaluations-labels)
        worst = [-1 for i in range(10)]
        below = []
        i = 0
        for e in errs:
            if e > threshold:
                below.append(i + k_at*n_in)
            if e > worst[-1]:
                worst[-1] = e
                worst.sort()
            i += 1
        k_at += 1
        print worst

if __name__ == '__main__':
    k_test(6, AMTOptions.deltas_file, .02)