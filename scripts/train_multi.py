import sys
sys.path.append('/home/annal/Izzy/vision_amt/')
from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net6_c,net8
from Net.tensor import inputdata
from options import AMTOptions
import numpy as np
from scripts import compile_supervisor, merge_supervised


if __name__ == "__main__":
	num_nets = 2
	net_paths = []
	for _ in range(num_nets):
		clean = False
		rand = False
		outfile = open(AMTOptions.amt_dir + 'deltas.txt', 'w+')
		merge_supervised.load_rollouts(clean, rand, (0,20), (0,-1), outfile)
		outfile.close()
		compile_supervisor.compile_reg()
		data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
		#net = net6_c.NetSix_C()
		net = net6.NetSix()
		net_paths.append(net.optimize(50,data, batch_size=200))

	train_writer = open(AMTOptions.amt_dir + 'net_cluster.txt', 'w+')
	for path in net_paths:
		train_writer.write(path + ' \n')

	train_writer.close()