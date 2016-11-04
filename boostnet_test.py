#boostnet test
import tensorflow as tf
import sys, cv2, time
from options import AMTOptions
sys.path.append('/home/annal/Izzy/')
from Tensor_Net.tensor import boost_net, inputdata, izzynet_boost
import numpy as np

if __name__ == '__main__':
	name = '/media/1tb/Izzy/nets/boostTests/test/'
	conv_path = '/media/1tb/Izzy/nets/boostTests/net6_08-04-2016_16h33m02s.ckpt'
	boosted_path = '/media/1tb/Izzy/nets/boostTests/test/ensemble_net_08-05-2016_13h52m53s.ckpt'
	# net = boost_net.boostNet(name, iters=100)
	# net_path = net.optimize(AMTOptions.train_file, AMTOptions.test_file,iters=3
	# 	, conv_path=conv_path, transfer=False)
	# sess = net.begin()
	data = inputdata.IMData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
	dat, lab = data.iterate_test(1000)
	diffs = []
	# net = izzynet_boost.IzzyNet_W()
	# sess = net.load(conv_path)
	net = izzynet_boost.IzzyNet_BEX(100)
	sess = net.load(boosted_path)
	for im, lab in zip(dat, lab):
		# data = inputdata.binary_mask(inputdata.im2tensor(im, channels=3), channels=3)
		data = im.reshape((250,250,3))
		# val, im = net.output(sess, data)
		val, im = net.output(sess, data, channels=3)

		# while 1:
		# 	frame = im.reshape((250,250,3))
		# 	# final = np.abs(-frame + template/255.0)
		# 	cv2.imshow('camera', frame)
		# 	a = cv2.waitKey(30)
		# 	if a == 27:
		# 		cv2.destroyAllWindows()
		# 		break
		# 	time.sleep(.005)
		diffs.append(np.abs(lab-val))
		print val, lab, np.sum(np.abs(lab-val))
	print np.max(diffs), np.linalg.norm(diffs, ord=1)/len(diffs)
	# net.end(sess)