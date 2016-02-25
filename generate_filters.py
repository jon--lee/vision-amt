from Net.tensor import net5, net6, inputdata, net4, net7
import numpy as np
from options import AMTOptions as opt
import cv2

def show(im):
    while True:
        cv2.imshow('preview', im)
        if cv2.waitKey(20) == 27:
            break

def generate_filters(net, model_path, filter_layer, preview=False):
    sess = net.load(var_path = model_path)
    filter = sess.run(filter_layer)
    for j in range(filter.shape[-1]):
        img = filter[:,:,:,j] - np.amin(filter[:,:,:,j])
        img = img * 255.0/np.amax(img)
        print img
        path = net.name + '_filter_' + str(j) + '.jpg'
        print path
        cv2.imwrite('filters/' + path, np.array(img))

    sess.close()

def generate_inputs(net, model_path, conv_layer, test_image, preview=False):
    sess = net.load(var_path=model_path)
    im = inputdata.im2tensor(test_image, channels=3)
    shape = im.shape
    im = np.reshape(im, (-1, shape[0], shape[1], shape[2]))

    with sess.as_default():
        filter = sess.run(conv_layer, feed_dict={net.x:im})
        for i in range(filter.shape[-1]):
            filter_image = filter[0,:,:,i] * 255.0
            filter_path = net.name + '_input' + str(i) + '.jpg'
            print filter_path
            cv2.imwrite('filters/' + filter_path, np.array(filter_image))
    cv2.imwrite('filters/' + net.name +'_sample.jpg', test_image)
    sess.close()

if __name__ == '__main__':
    #model_path = opt.tf_dir + 'net6/net6_02-15-2016_13h57m43s.ckpt'
    #model_path = opt.tf_dir + 'net5/net5_02-15-2016_13h22m34s.ckpt'
    #net = net5.NetFive()
    #net = net6.NetSix()

    #model_path = 'Netnet4_02-13-2016_18h44m24s.ckpt'
    #net = net4.NetFour()

    #model_path = '/media/1tb/Izzy/nets/net4_02-18-2016_16h18m23s.ckpt'
    #net = net4.NetFour()

    net = net4.NetFour()
    model_path = '/media/1tb/Izzy/nets/net4_02-23-2016_13h56m26s.ckpt'

    conv_layer = net.h_conv1
    filter_layer = net.w_conv1
    
    test_image_path = '/home/annal/Izzy/vision_amt/data/amt/colors/rollout12_frame_4.jpg'
    test_image = cv2.imread(test_image_path)
    print "input image shape: " + str(test_image.shape)
    generate_inputs(net, model_path, conv_layer, test_image, preview=False)   
    generate_filters(net, model_path, filter_layer)
