import cv2, sys, argparse

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

def edit_deltas(new_deltas_path, full_deltas_path, name ='', test = ''):
	old_deltas = open(AMTOptions.amt_dir + 'deltas.txt', 'r')
	full_deltas = open(full_deltas_path, 'w')
	new_deltas = open(new_deltas_path, 'r')

	corrections = dict()
	for line in new_deltas:
		vals = line.split('\t')
		rnum = int(vals[1])
		fnum = int(vals[2])
		delta = [0.0] * 4
		delta[0] = float(vals[3])
		delta[1] = float(vals[4])
		corrections[(rnum, fnum)] = delta

	for line in full_deltas:
		fnum = frame_num(line)
		rnum = traj_num(line)
		line_vals = line.split(" ")
		if (rnum, fnum) in corrections: 
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
    args = parser.parse_args()
    if args.test == None:
        args.test = ''
    edit_deltas(AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/new_deltas.txt", AMTOptions.amt_dir + "cross_computations/" + args.name + "_crossvals/full_deltas.txt", args.name, args.test)