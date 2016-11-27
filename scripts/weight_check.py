#boostnet test
import tensorflow as tf
import sys, cv2, time
from options import AMTOptions
# sys.path.append('/home/annal/Izzy/')
# from Tensor_Net.tensor import boost_net, inputdata, izzynet_boost
import numpy as np
from Net.tensor import net3,net4,net5,net6, net6_c
from tensorflow.python.framework import ops

def get_layer_variables(sess, layer):
    weights = sess.run(layer)
    return weights

if __name__ == '__main__':
	name = '/media/1tb/Izzy/nets/boostTests/test/'
	conv_path = '/media/1tb/Izzy/nets/boostTests/net6_08-04-2016_16h33m02s.ckpt'
	# boosted_path = '/media/1tb/Izzy/nets/net6_07-20-2016_14h57m06s.ckpt'
	# boosted_path = '/media/1tb/Izzy/nets/net6_07-18-2016_13h43m29s.ckpt'
	persons = {}
	persons['jonathan'] = ['/media/1tb/Izzy/nets/net6_07-14-2016_16h10m54s.ckpt', '/media/1tb/Izzy/nets/net6_07-15-2016_11h48m14s.ckpt', '/media/1tb/Izzy/nets/net6_07-15-2016_11h57m33s.ckpt', '/media/1tb/Izzy/nets/net6_07-14-2016_17h46m20s.ckpt', '/media/1tb/Izzy/nets/net6_07-15-2016_10h44m06s.ckpt']
	persons['jacky'] = ['/media/1tb/Izzy/nets/net6_07-18-2016_13h43m29s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_15h47m45s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_10h45m25s.ckpt', '/media/1tb/Izzy/nets/net6_07-18-2016_14h32m27s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_14h14m23s.ckpt']
	persons['aimee'] = ['/media/1tb/Izzy/nets/net6_07-19-2016_16h47m52s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_14h57m06s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h06m14s.ckpt', '/media/1tb/Izzy/nets/net6_07-19-2016_17h27m25s.ckpt', '/media/1tb/Izzy/nets/net6_07-19-2016_17h59m22s.ckpt']
	persons['chris'] = ['/media/1tb/Izzy/nets/net6_07-20-2016_10h28m56s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_10h38m04s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_13h04m31s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_11h21m08s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_11h51m23s.ckpt']
	persons['dave'] = ['/media/1tb/Izzy/nets/net6_07-20-2016_11h08m10s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h37m33s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h50m37s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_11h53m16s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h26m59s.ckpt']
	persons['lauren'] = ['/media/1tb/Izzy/nets/net6_07-21-2016_12h36m29s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_13h56m55s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_14h06m39s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_13h17m00s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_14h16m19s.ckpt']
	persons['johan'] = ['/media/1tb/Izzy/nets/net6_07-21-2016_15h30m43s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_13h42m56s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_13h52m02s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_16h17m35s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_11h51m53s.ckpt']
	persons['sherdil'] = ['/media/1tb/Izzy/nets/net6_07-25-2016_12h38m34s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_13h40m49s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_12h40m57s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_13h12m20s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_13h29m25s.ckpt']
	persons['richard'] = ['/media/1tb/Izzy/nets/net6_07-26-2016_16h12m46s.ckpt', '/media/1tb/Izzy/nets/net6_07-27-2016_15h02m33s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_17h45m12s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_16h42m18s.ckpt', '/media/1tb/Izzy/nets/net6_07-27-2016_12h26m57s.ckpt']
	persons['sona'] = ['/media/1tb/Izzy/nets/net6_07-27-2016_18h22m44s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_16h16m38s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_16h31m52s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_11h14m28s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_16h07m15s.ckpt']

	a_weight = []
	for person in persons.keys():
		p_weight = [person]
		for boosted_path in persons[person]:
			net = net6.NetSix()
			sess = net.load(boosted_path)
			weights = {}
			weights['w_fc1'] = get_layer_variables(sess, net.w_fc1)
			weights['b_fc1'] = get_layer_variables(sess, net.b_fc1)
			weights['w_conv1'] = get_layer_variables(sess, net.w_conv1)
			weights['b_conv1'] = get_layer_variables(sess, net.b_conv1)
			weights['w_fc2'] = get_layer_variables(sess, net.w_fc2)
			weights['b_fc2'] = get_layer_variables(sess, net.b_fc2)

			average_weight = np.sum([np.sum(np.abs(weights['w_fc1'])), np.sum(np.abs(weights['b_fc1'])),
				np.sum(np.abs(weights['w_conv1'])), np.sum(np.abs(weights['b_conv1'])), np.sum(np.abs(weights['w_fc2'])),
				np.sum(np.abs(weights['b_fc2']))])

			print "average_weight: ", average_weight
			p_weight.append(average_weight)
			sess.close()
			ops.reset_default_graph()
		a_weight.append(p_weight)
	print a_weight