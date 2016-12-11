import cv2, sys, argparse
sys.path.append('/home/huarsc/research/vision-amt/')
from scripts import merge_supervised
from options import AMTOptions

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

def edit_deltas(new_deltas_path, full_deltas_path, name ='', test = '', remove_all=False):
	old_deltas = open(AMTOptions.amt_dir + 'deltas.txt', 'r')
	full_deltas = open(full_deltas_path, 'w')
	new_deltas = open(new_deltas_path, 'r')

	corrections = dict()
	for line in new_deltas:
		vals = line.split('\t')
		rnum = int(vals[1])
		fnum = int(vals[2])
		delta = [0.0] * 4
		if vals[3] != 'd':
			delta[0] = float(vals[3])
			delta[1] = float(vals[4])
		else:
			delta = "delete"	
		corrections[(rnum, fnum)] = delta

	for line in old_deltas:
		fnum = frame_num(line)
		rnum = traj_num(line)
		line_vals = line.split(" ")
		if (rnum, fnum) in corrections: 
			if corrections[(rnum, fnum)] == "delete" or remove_all:
				print corrections[(rnum, fnum)]
				continue
			new_line = line_vals[0] + " " + " ".join(map(str, corrections[(rnum, fnum)])) + "\n"
			full_deltas.write(new_line)
		else:
			full_deltas.write(line)
	full_deltas.close()
	new_deltas.close()
	old_deltas.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="run experiment, will prompt for name")
    parser.add_argument("-t", "--test", type=str, help="test folder name")
    parser.add_argument("-o", "--output", type=str, help="file to output values to")
    parser.add_argument("-d", "--deltas", type=str, help="file tfrom which to read deltas")
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollouts to be used for training")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollouts to be used for training")
    parser.add_argument("-r", "--removeall", help="removes all deltas found",
                    action="store_true")
    args = parser.parse_args()
    if args.test == None:
        args.test = ''
    outfile = open(AMTOptions.amt_dir + 'deltas.txt', 'w+')
    merge_supervised.load_rollouts(False, False, (args.first,args.last), (0,100), outfile, name = args.name)
    outfile.close()
    edit_deltas(AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/" + args.deltas, AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/" + args.output, args.name, args.test, args.removeall)