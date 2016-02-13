from Net.tensor import net3, inputdata
from options import AMTOptions as opt
from tensorflow.python.framework import ops
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train', required=False, action='store_true')
args = vars(ap.parse_args())



if __name__ == '__main__':
    
    data = inputdata.AMTData(opt.train_file, opt.test_file)
    images, labels = data.next_test_batch()
    test_image = images[0]
    test_label = labels[0]
    
    test_file = open(opt.test_file, 'r')
    line_split = test_file.next().split(' ')
    path = line_split[0]
    labels = line_split[1:]
   
    test_im = cv2.imread(path) 
    
    path = opt.tf_dir + 'net3/model.ckpt'

    """sess = net.load(var_path=path)
    print "loaded session"
    print "Expected label: " + str(labels)
    print "Label before training: " + str(net.output(sess, test_im))
   
    sess.close()
    ops.reset_default_graph()
    """



    print "acutal label is: " + str(labels)

    if args['train']:


        net = net3.NetThree()
        net.optimize(100, data=data, path=path, batch_size=150)
    
        ops.reset_default_graph()
        net = net3.NetThree()

        #sess = net.load(var_path=path)
        #print "Label after reloading grpah: " + str(net.output(sess, test_im))
        #sess.close()

    else:
        net = net3.NetThree()
        sess = net.load(var_path=path)
        print "Lable after loading graph: " + str(net.output(sess, test_im))
        sess.close()

