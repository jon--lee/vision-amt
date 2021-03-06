import random
import numpy as np
import tensorflow as tf
import cv2

class InputData():

    def __init__(self, train_path, test_path):
        self.i = 0

        train_tups = parse(train_path, 4000) 
        test_tups = parse(test_path, 800)
        
        self.train_data = []
        for path, labels in train_tups:
            im = cv2.imread(path)
            im = im2tensor(im)
            self.train_data.append((im, labels))

        self.test_data = []
        for path, labels in test_tups:
            im = cv2.imread(path)
            im = im2tensor(im)
            self.test_data.append((im, labels))

        random.shuffle(self.train_data)
        random.shuffle(self.test_data)


    def next_train_batch(self, n):
        if self.i + n > len(self.train_data):
            self.i = 0
            random.shuffle(self.train_data)
        batch = self.train_data[self.i:n+self.i]
        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])
    
    def next_test_batch(self):
        batch = self.test_data
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])

    def all_train_batch(self):
        if self.i + n > len(self.train_data):
            self.i = 0
            random.shuffle(self.train_data)
        batch = self.train_data[self.i:]
        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])


    
def parse(filepath, stop=-1):
    """
        Parses file containing paths and labels into list
        of tupals in the form of:
        
        data =  [ 
                    (path, [label1, label2 ... ])
                    ...
                ]
    """
    f = open(filepath, 'r')
    tups = []
    lines = [ x for x in f ]
    random.shuffle(lines)
    for i, line in enumerate(lines):
        split = line.split(' ')
        path = split[0]
        labels = np.array( [ float(x) for x in split[1:] ] )
        tups.append((path, labels))
        if (not stop < 0) and i >= stop-1:
            break            
    return tups


def im2tensor(im,channels=1):
    """
        convert 3d image (height, width, 3-channel) where values range [0,255]
        to appropriate pipeline shape and values of either 0 or 1
        cv2 --> tf
    """
    shape = np.shape(im)
    h, w = shape[0], shape[1]
    zeros = np.zeros((h, w, channels))
    for i in range(channels):
        #Binary Mask
        zeros[:,:,i] = np.round(im[:,:,i] / 255.0 - .25, 0)
        #zeros[:,:,i] = np.round(im[:,:,i] / 255.0, 0)

        #Nomarlized RGB
        #zeros[:,:,i] = im[:,:,i] / 255.0

        #zeros[:,:,i] = im[:,:,i]
    return zeros

def process_out(n):
    # print "n: ", n
    out = np.zeros([4])
    n_0 = np.argmax(n[0:5])
    n_1 = np.argmax(n[5:10])
    # n_2 = np.argmax(n[10:14])
    # n_3 = np.argmax(n[15:20])    
    # print n_0
    # print n_1
    out[0] = -(n_0 - 2)/2.0
    out[1] = (4-n_1)/4.0
    # out[2] = (n_2 - 2)/2.0
    # out[3] = (n_3 - 2)/2.0

    return out


class AMTData(InputData):
    
    def __init__(self, train_path, test_path,channels=1):
        self.train_tups = parse(train_path)
        self.test_tups = parse(test_path)

        self.i = 0
        self.channels = channels

        random.shuffle(self.train_tups)
        random.shuffle(self.test_tups)

    def next_train_batch(self, n):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        if self.i + n > len(self.train_tups):
            self.i = 0
            random.shuffle(self.train_tups)
        batch_tups = self.train_tups[self.i:n+self.i]
        batch = []
        for path, labels in batch_tups:
            im = cv2.imread(path)
         
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])







    def next_test_batch(self):
        """
        read into memory on request
        :return: tuple with images in [0], labels in [1]
        """
        batch = []
        for path, labels in self.test_tups[:200]:
            im = cv2.imread(path,self.channels)
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        random.shuffle(self.test_tups)
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])
