from Net.tensor import tensornet, net2
from pipeline.bincam import BinaryCamera
from options import Options
import cv2
import numpy as np
bc = BinaryCamera('./meta.txt')
bc.open()

options = Options()
options.show = True
options.record = False

tfnet = net2.NetTwo()
sess = tfnet.load(var_path=options.tf_dir + 'net2/net2_01-21-2016_02h14m08s.ckpt')
im = bc.read_binary_frame(show=options.show, record=False)
im = np.reshape(im, (125, 125, 1))
print(tfnet.output(sess, im))
sess.close()
sess = tfnet.load(var_path=options.tf_dir + 'net2/net2_01-21-2016_02h14m08s.ckpt')
print(tfnet.output(sess, im))
sess.close()
