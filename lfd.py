from options import Options
from dataset import Dataset
import datetime
import caffe
import time
import numpy as np
import tensorflow as tf
from Net.tensor import net2, inputdata

class LFD():

    def __init__(self, bincam, izzy, turntable, controller, options=Options()):
        self.bc = bincam
        self.izzy = izzy
        self.turntable = turntable
        self.c = controller
        self.options = options




    def test(self):
        try:
            while True:
                if self.c.override():
                    print "supervisor override"
                state = self.izzy.getState()
                frame = self.bc.read_frame(show=self.options.show, record=self.options.record, state=state)
                controls = self.c.getUpdates()
                self.update_gripper(controls)
                
                print "test: " + str(controls)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        
        if self.options.record:
            self.bc.save_recording()




    def deploy(self, dataset_name=""):
        net = caffe.Net(self.options.model_path, self.options.weights_path, caffe.TEST)        
        
        if not dataset_name:
            dataset = Dataset.create_ds(self.options, prefix="deploy")
        else:           
            dataset = Dataset.get_ds(self.options, dataset_name)

        try:
            while True:
                controls = self.c.getUpdates()
                state = self.izzy.getState()
                
                if self.c.override():
                    # supervised
                    controls = self.c.getUpdates()
                    self.update_gripper(controls)
                    
                    controls = self.controls2simple(controls)
                    if not all(int(c) == 0 for c in controls):
                        frame = self.bc.read_frame(show=self.options.show, record=self.options.record, state=state)
                        dataset.stage(frame, controls, state)
                    
                    print "supervisor: " + str(controls)
                else:
                    # autonomous
                    frame = self.bc.read_binary_frame(record=self.options.record, state=state)
                    data4D = np.zeros([1, 3, 125, 125])
                    frame = frame / 255.0
                    data4D[0,0,:,:] = frame
        	    data4D[0,1,:,:] = frame
        	    data4D[0,2,:,:] = frame

                    net.forward_all(data=data4D)
                    net_controls = net.blobs['out'].data.copy()[0]
                    print controls
                    controls = self.net2controls(net_controls)
                    
                    self.update_gripper(controls)
                    
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        dataset.commit()
        if self.options.record:
            self.bc.save_recording()


    # TODO: un-tested method, deploy on robot and debug
    def rollout_tf(self, dataset_name=''):
        net = self.options.tf_net
        sess = self.options.tf_sess
        if not dataset_name:
            dataset = Dataset.create_ds(self.options, prefix='learn')
        else:               
            dataset = Dataset.get_ds(self.options, dataset_name)
        
        try:
            while True:
                controls = self.c.getUpdates()
                state = self.izzy.getState()

                if self.c.override():
                    # supervised
                    self.update_gripper(controls)

                    controls = self.controls2simple(controls)
                    if not all(int(c) == 0 for c in controls):
                        frame = self.bc.read_frame(show=self.options.show, record=self.options.record, state=state)
                        dataset.stage(frame, controls, state)

                    print "supervisor: " + str(controls)
                
                else:
                    # autonomous
                    frame = self.bc.read_binary_frame(record=self.options.record, state=state)
                    frame = np.reshape(frame, (125, 125, 1))
                    net_controls = net.output(sess, frame)
                    print controls
                    controls = self.net2controls(net_controls)
                    self.update_gripper(controls)
                time.sleep(0.03)

        except KeyboardInterrupt:
            pass
        
        sess.close()
        dataset.commit()
        if self.options.record:
            self.bc.save_recording()





    def learn(self, dataset_name=''):
        
        if not dataset_name:
            dataset = Dataset.create_ds(self.options, prefix='learn')
        else:               
            dataset = Dataset.get_ds(self.options, dataset_name)

        try:
            while True:
                controls = self.c.getUpdates()
                state = self.izzy.getState()
                self.update_gripper(controls)

                controls = self.controls2simple(controls)
                if not all(int(c) == 0 for c in controls):
                        frame = self.bc.read_frame(show=self.options.show, record=self.options.record, state=state)
                        dataset.stage(frame, controls, state)
                
                print "supervisor: " + str(controls)
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            pass
        
        dataset.commit()
        if self.options.record:
            self.bc.save_recording()

    def update_gripper(self, controls):
        controls[1] = 0
        controls[3] = 0
        self.izzy.control(controls)
        self.turntable.control([controls[5]])

    def net2controls(self, net_controls):
        """ convert net output (1d nparray) to izzy controls """
        for i in range(len(net_controls)):
            net_controls[i] = (net_controls[i] - self.options.translations[i]) * self.options.scales[i] * 1.5
            if abs(net_controls[i]) < self.options.drift:
                net_controls[i] = 0.0
        return [net_controls[0], 0.0, net_controls[1], 0.0, net_controls[2], net_controls[3]]


    def controls2simple(self, controls):
        """ convert raw izzy controls to net output (no trans/scaling) """
        return [controls[0], controls[2], controls[4], controls[5]]
