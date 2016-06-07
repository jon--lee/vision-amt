from options import AMTOptions
from Net.tensor import inputdata, net6_c, net6
import cv2
import numpy as np
from pipeline.bincam import BinaryCamera
import tensorflow as tf
import time
import matplotlib.pyplot as plt
def color(frame):
	color_frame = cv2.resize(frame.copy(), (250, 250))
	return color_frame


if __name__ == '__main__':
	bc = BinaryCamera('./meta.txt')
	bc.open()
	options = AMTOptions()
	

	net = net6_c.NetSix_C()
	path = '/media/1tb/Izzy/nets/net6_05-11-2016_12h09m12s.ckpt'
	
	sess = net.load(path)
	#sess = tf.Session()
	#sess.run(tf.initialize_all_variables())
		
	last_frame = None
	for i in range(4):
		bc.read_frame()
	
	for i in range(1):
		try:
			for i in range(10000):
				frame = bc.read_frame()
				frame = color(frame)
				frame = np.reshape(frame, (250, 250, 3))
				cv2.imshow('camera', frame)
				cv2.waitKey(30)
				
				dists = net.class_dist(sess, frame, 3)
				plt.clf()
				x = np.array([-1,-0.5,0,0.5,1])
				plt.subplot(2,1,1)        
				plt.plot(x,dists[0,:])
				plt.xlabel('Rotation')
				plt.subplot(2,1,2)
				plt.plot(x,dists[1,:])
				plt.xlabel('Extension')
				plt.draw()
				plt.show(block=False)
				time.sleep(.005)	
		except KeyboardInterrupt:
			pass

		#print frame
		
		
	
