import os, sys
from options import AMTOptions

def generate_templates(rollout_lst, name, target):
	target_file = open(AMTOptions.amt_dir + 'templates/' + target + '_paths.txt', 'w')
	for i in range(len(rollout_lst)):
		rol = rollout_lst[i]
		writestring = AMTOptions.supervised_dir + name + '_rollouts/' + 'supervised' + str(rol) + '/template.npy\n'
		if i == len(rollout_lst) - 1:
			target_file.write(writestring[:-1])
		else:
			target_file.write(writestring)
	target_file.close()
	

if __name__ == '__main__':
	rol_lst = [69, 76, 73, 106, 74, 102, 101, 105, 77, 71, 99, 85, 107, 93, 110, 72, 70, 84, 94, 91, 89, 86, 97, 68, 88, 82, 95, 75, 65, 79, 66, 80, 81, 78, 100, 92, 87, 83, 90, 98, 104, 67, 109, 96, 108, 103]
	rol_lst.reverse()
	generate_templates(rol_lst, sys.argv[1], sys.argv[2])