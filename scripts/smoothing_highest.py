import cv2, sys
sys.path.append('/home/annal/Izzy/vision_amt/')

from options import AMTOptions
import numpy as np, argparse
from scripts import compile_supervisor, merge_supervised
from scipy import signal
import matplotlib.pyplot as plt




names = ['Sona', 'Aimee', 'Dave', 'Sherdil', 'Jonathan', 'Johan', 'Jacky', 'Richard']
# names = ['Sona']

highest_differences = open(AMTOptions.amt_dir + "highest_diffs.txt", 'w')
leng = 25
paths = []
differences = []
for name in names:
	difference_file = open(AMTOptions.amt_dir + "cross_computations/" + name + "_crossvals/smoothings.txt", 'r')
	for line in difference_file:
		values = line.split('\t')
		difference = np.linalg.norm(np.array(map(float, values[1].split(' '))))
		if len(paths) < leng:
			paths.append((values[0], difference))
			paths.sort(key=lambda x: x[1])
		else:
			if difference > paths[0][1] :
				paths[0] = (values[0], difference) 
				paths.sort(key=lambda x: x[1])
		# print line
		if difference > 2.0:
			difference = 2.0
		difference = difference /2.0
		differences.append(difference)
	differences.sort()
	difference_file.close()
	highest_differences.write(name + "\n")
	for dif in paths:
		highest_differences.write(dif[0] + '\t' + str(dif[1]) + '\n')
print len(differences)

highest_differences.close()
plt.ylabel('Amount of change, in interval [0,2]')
plt.title('Sorted Magnitude of Change due to smoothing')
plt.plot(differences)
plt.show()


names = ['Sona', 'Aimee', 'Dave', 'Sherdil', 'Jonathan', 'Johan', 'Jacky', 'Richard']
# names = ['Sona']

highest_differences = open(AMTOptions.amt_dir + "highest_diffs.txt", 'w')
leng = 25
paths = []
differences = []
for name in names:
	difference_file = open(AMTOptions.amt_dir + "cross_computations/" + name + "_crossvals/comparisons.txt", 'r')
	for line in difference_file:
		values = line.split('\t')
		difference = np.linalg.norm(np.array(map(float, values[1].split(' '))))
		if len(paths) < leng:
			paths.append((values[0], difference))
			paths.sort(key=lambda x: x[1])
		else:
			if difference > paths[0][1] :
				paths[0] = (values[0], difference) 
				paths.sort(key=lambda x: x[1])
		# print line
		if difference > 2.0:
			difference = 2.0
		difference = difference / 2.0
		differences.append(difference)
	differences.sort()
	difference_file.close()
	highest_differences.write(name + "\n")
	for dif in paths:
		highest_differences.write(dif[0] + '\t' + str(dif[1]) + '\n')
print len(differences)
highest_differences.close()
plt.axhline(y=.65, xmin=0, xmax=30000, linewidth=1, color = 'r')
# RIGHT VERTICAL
plt.axvline(x=28150, ymin=0, ymax =1, linewidth=1, color='r')
# LEFT VERTICAL
plt.ylabel('Amount of error, in interval [0,2]')
plt.title('Sorted Magnitude of Error on Cross-validation')
plt.plot(differences)
plt.show()
# # print '\n'.join(map(str, paths))

def traj_num(line):
	if line.find('_supervised') != -1:
		return int(line[line.find('_supervised') + len('_supervised'):line.find('_frame_')])
	else:
		return int(line[line.find('_rollout') + len('_rollout'):line.find('_frame_')])

names = ['Sona', 'Aimee', 'Dave', 'Sherdil', 'Jonathan', 'Johan', 'Jacky', 'Richard']
# names = ['Sona']
fit = []
all_trajs = np.array([0.0 for i in range(54)])
for name in names:
	traj_error = []
	difference_file = open(AMTOptions.amt_dir + "cross_computations/" + name + "_crossvals/smoothings.txt", 'r')
	traj_num_last = -1
 	this_traj = []
	for line in difference_file:
		values = line.split('\t')
		difference = np.linalg.norm(np.array(map(float, values[1].split(' '))))
		if traj_num_last == -1:
			traj_num_last = traj_num(values[0])
		elif traj_num_last != traj_num(values[0]):
			traj_num_last = traj_num(values[0])
			traj_error.append(np.mean(np.array(this_traj)))
			this_traj = [difference]
		else:
			this_traj.append(difference)
	# fit.append(c[0])
	# print c[0]

	print len(traj_error)
	all_trajs += np.array(traj_error)

c = np.polyfit(np.array(range(len(all_trajs/8))), all_trajs/8, 1)
plt.plot([c[0]*i + c[1] for i in range(len(traj_error))])
plt.ylabel('Average Smoothing Error')
plt.xlabel('Rollout number')
plt.title('Change in Smoothing Error with Experience')
plt.plot(all_trajs/8)
plt.show()
print np.mean(fit)

names = ['Sona', 'Aimee', 'Dave', 'Sherdil', 'Jonathan', 'Johan', 'Jacky', 'Richard']
# names = ['Sona']
fit = []
all_trajs = np.array([0.0 for i in range(54)])
for name in names:
	traj_error = []
	difference_file = open(AMTOptions.amt_dir + "cross_computations/" + name + "_crossvals/comparisons.txt", 'r')
	traj_num_last = -1
 	this_traj = []
	for line in difference_file:
		values = line.split('\t')
		difference = np.linalg.norm(np.array(map(float, values[1].split(' '))))
		if traj_num_last == -1:
			traj_num_last = traj_num(values[0])
		elif traj_num_last != traj_num(values[0]):
			traj_num_last = traj_num(values[0])
			traj_error.append(np.mean(np.array(this_traj)))
			this_traj = [difference]
		else:
			this_traj.append(difference)
	# fit.append(c[0])
	# print c[0]

	print len(traj_error)
	all_trajs += np.array(traj_error)

c = np.polyfit(np.array(range(len(all_trajs/8))), all_trajs/8, 1)
plt.plot([c[0]*i + c[1] for i in range(len(traj_error))])
plt.ylabel('Average Smoothing Error')
plt.xlabel('Rollout number')
plt.title('Change in Cross Validation Error with Experience')
plt.plot(all_trajs/8)
plt.show()
print np.mean(fit)
			


