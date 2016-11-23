import sys
sys.path.append('/home/annal/Izzy/vision_amt/')
from pipeline.bincam import BinaryCamera
import cv2
import numpy as np
from Net.tensor import inputdata
bc = BinaryCamera('./meta.txt')
bc.open()
frame = bc.read_frame()
# net_path = '/media/1tb/Izzy/nets/net4_02-17-2016_19h07m23s.ckpt'
#vc = cv2.VideoCapture(0)
#rval, frame = vc.read()

# tf_net = net4.NetFour()
# tf_net_path = net_path
# sess = tf_net.load(var_path=tf_net_path)


while True:
	frame = bc.read_frame()

	# img = cv2.resize(frame.copy(), (250, 250))
	img = frame
	img = inputdata.im2tensor(img,channels=3)
	# img = np.reshape(img, (250, 250, 3))
	# cv2.imwrite("test.jpg",img.copy())
	img_jpg = cv2.imread("/home/annal/Izzy/vision_amt/data/amt/supervised_rollouts/Caleb_rollouts/supervised1/Caleb_supervised1_frame_0.jpg")

	# not_saved = tf_net.output(sess, img,channels=3)
	# saved = tf_net.output(sess, img,channels=3)

	# print "NOT SAVED ",not_saved
	# print "SAVED ",saved
	print img.shape
	cv2.imshow("camera", img)
	cv2.waitKey(30)
	cv2.imshow("test", img_jpg)
	cv2.waitKey(30)

