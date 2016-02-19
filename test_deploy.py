from Net.tensor import net3, inputdata
from options import AMTOptions as opt
from tensorflow.python.framework import ops
import cv2
test_file = open(opt.train_file, 'r')
line_split = test_file.next().split(' ')
path = line_split[0]
labels = line_split[1:]

net = net3.NetThree()

print "Expected label: " + str(labels)

test_im = cv2.imread(path)

model_path = opt.tf_dir + 'net3/model.ckpt'

sess = net.load(var_path=model_path)
print "Actual label: " + str(net.output(sess, test_im))

