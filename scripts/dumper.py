# dumper
import scripts
from options import AMTOptions
import scipy
import scipy.stats
# import scipy.stats.pearsonr
import os

def traj_num(line):
    if line.find('_supervised') != -1:
        return int(line[line.find('_supervised') + len('_supervised'):line.find('_frame_')])
    else:
        return int(line[line.find('_rollout') + len('_rollout'):line.find('_frame_')])

def frame_num(line):
    return int(line[line.find('_frame_') + len('_frame_'):line.find('.jpg')])



supervised_dir = AMTOptions.supervised_dir
supervised_dir += "Dave" + "_rollouts/"

retro_deltas_file = open(supervised_dir + '/retroactive_feedback0_4.txt', 'r')

frames_values = set()

for l in retro_deltas_file:
	frames_values.add((traj_num(l), frame_num(l)))

target = open("combined_for_retro.txt", 'w')
rollouts = [x[0] for x in os.walk(supervised_dir)]
supervisor_dirs = [rollout_dir for rollout_dir in rollouts if rollout_dir != supervised_dir]
rol_num = lambda x: int(x[x.find('s/supervised') + len('s/supervised'):])
supervisor_dirs.sort(key = rol_num)
supervisor_dirs = supervisor_dirs[0:5]

for dirname in supervisor_dirs:
	deltas_file = open(dirname + '/deltas.txt', 'r')
	for l in deltas_file:
		if (traj_num(l), frame_num(l)) in frames_values:
			target.write(l)
	deltas_file.close()

target.close()
retro_deltas_file.close()

teleop = open('/home/annal/Izzy/vision_amt/data/amt/comparisons/pictures_ICRA2016/deltas_teleop_comp.csv', 'r')
retro = open('/home/annal/Izzy/vision_amt/data/amt/comparisons/pictures_ICRA2016/deltas_retro_comp.csv', 'r')
t_forward = []
r_forward = []
t_angle = []
r_angle = []
for l in zip(teleop, retro):
	tel = l[0].split('\t')
	ret = l[1].split('\t')
	if tel[0] == ret[0]:
		t_forward.append(float(tel[2]))
		r_forward.append(float(ret[2]))
		t_angle.append(float(tel[1]))
		r_angle.append(float(ret[1]))

print len(r_forward)

print scipy.stats.pearsonr(t_forward, r_forward)
print scipy.stats.pearsonr(t_angle, r_angle)
