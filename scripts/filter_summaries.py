from Net.tensor import net5, inputdata
from options import AMTOptions as opt
import tensorflow as tf
import numpy as np
import cv2
path = opt.tf_dir + 'net5/net5_02-15-2016_13h22m34s.ckpt'
net = net5.NetFive()

sess = net.load(var_path=path)

filter_summaries = []
slices = tf.split(3, 5, net.w_conv1)
for i, slice in enumerate(slices):
    print slice.get_shape().as_list()
    scope = 'jonathan' + str(i)
    filter_summaries.append(tf.image_summary(scope, slice, max_images=1))



summary_writer = tf.train.SummaryWriter('/tmp/logs', sess.graph_def)
test_image_path = opt.colors_dir + 'rollout54_frame_16.jpg'
test_image = cv2.imread(test_image_path)
im = inputdata.im2tensor(test_image, 3)
shape = np.shape(im)
im = np.reshape(im, (-1, shape[0], shape[1], shape[2]))
feed_dict = {net.x: im}
with sess.as_default():
    for filtsum in filter_summaries:
        filtsum = filtsum.eval(feed_dict=feed_dict)
        summary_writer.add_summary(filtsum, 0)
