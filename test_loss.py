from options import AMTOptions
from Net.tensor import inputdata, net5, net6, net7, net8
import numpy as np
import cv2


def scale2net(deltas):
    deltas[0] = float(deltas[0])/0.2
    deltas[1] = float(deltas[1])/0.01
    deltas[2] = float(deltas[2])/0.005
    deltas[3] = float(deltas[3])/0.2
    return deltas

def mean(lst):
    sm = 0.0
    for val in lst:
     	sm += val
    loss = sm/len(lst)*0.25
    print "AVERAGE SQUARED EUCLIDEAN LOSS ",loss
    return loss

def euclidean_loss(net, model_path, dir, channels):
    sess = net.load(var_path=model_path)

    deltas_file = open(AMTOptions.deltas_file, 'r')

    losses = []

    for line in deltas_file:
        path = dir
        split = line.split(' ')
        image_path = path + split[0]
        deltas = [ float(x) for x in split[1:]]
        img = cv2.imread(image_path)
        net_output = np.array(net.output(sess, img, channels=channels))
        expected_output = np.array(scale2net(deltas))
        loss = .5 * np.linalg.norm(net_output - expected_output) ** 2.
        losses.append(loss)

    mean(losses)

    sess.close()

def euclidean_loss_color(net, model_path):
    return euclidean_loss(net, model_path, AMTOptions.colors_dir, 3)


if __name__ == '__main__':
    net = net6.NetSix()
    model_path = AMTOptions.nets_dir + 'net6_02-19-2016_17h14m04s.ckpt'
    #net = net8.NetEight()
    #model_path = AMTOptions.nets_dir + 'net8_02-19-2016_23h49m00s.ckpt'
    euclidean_loss_color(net, model_path)



