import h5py
import numpy as np
import caffe
from PIL import Image
import parser

def reshape(img):
    """
    Given np array image in shape (width, height, channel)
    convert to same image in shape (channel, width, height) for caffe
    Any images size is okay.
    """
    b = np.zeros((len(img[0,0,:]), len(img[0,:,0]), len(img[:,0,0])))
    b[0,:,:] = img[:,:,0]
    b[1, :, :] = img[:, :, 1]
    b[2, :, :] = img[:, :, 2]
    return b


def img2hdf(filename_prefix, output_path, cutoff=-1):
    """
    Retrieve images from textfile of paths and labels
    Write images and labels to datasets

    filename_prefix - prefix name of text file to read from (exclude extension)
    [ i.e. img2hdf('train') as opposed to img2hdf('train.txt') ]
    """
    data = parser.parse(filename_prefix + '.txt', cutoff)
    paths, labels = zip(*data)
   
    images = np.array([ reshape(np.round(caffe.io.load_image(path), 0)) for path in paths ])
    labels = np.array(labels)

    f = h5py.File(filename_prefix + '.h5', 'w')    
    f.create_dataset('data', data=images)
    f.create_dataset('labels', data=labels)

    f.close()

    # show where to find .h5 file by writing to prefix_hdf.txt
    lst = open(filename_prefix + '_hdf.txt', 'w+')
    lst.write(output_path)
    lst.close()
